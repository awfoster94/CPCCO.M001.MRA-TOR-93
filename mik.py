import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from pykrige.ok import OrdinaryKriging
from skgstat import Variogram
from shapely.geometry import Point
from matplotlib.colors import LogNorm

# ─── Load Data ────────────────────────────────────────────────────────────
data_path = os.path.join("..", "02_data")
gis_path = os.path.join(data_path, "GIS")
aoi_path = os.path.join(gis_path, "shp", "OUs", "zp1_up1_outline.shp")
uumu_path = os.path.join(data_path, "HEIS_Data_Pull", "avg_2018_2020_conc_uu_mu_2.csv")
rcl_path = os.path.join("data_gap", "data_gap_RCs.csv")
grid_path = os.path.join(gis_path, "shp", "grid_274", "grid_274.shp")

aoi = gpd.read_file(aoi_path)
uumu = pd.read_csv(uumu_path)

# ─── Settings ─────────────────────────────────────────────────────────────
thresholds = [3.4, 34, 50, 100, 500, 1000, 1500]
cols = [f"ind{str(t).replace('.', '_')}" for t in thresholds]
z_mid = np.array([1.7, 18.7, 42, 75, 300, 750, 1250])  # bin midpoints

# ─── Create Interpolation Grid ────────────────────────────────────────────
aoi_bounds = aoi.total_bounds  # xmin, ymin, xmax, ymax
grid_res = 50  # adjust as needed
xgrid = np.arange(aoi_bounds[0], aoi_bounds[2], grid_res)
ygrid = np.arange(aoi_bounds[1], aoi_bounds[3], grid_res)
xx, yy = np.meshgrid(xgrid, ygrid)
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_shape = xx.shape

# ─── Create AOI Mask ──────────────────────────────────────────────────────
grid_df = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in grid_points], crs=aoi.crs)
within_mask = grid_df.within(aoi.geometry.values.union_all())
mask = within_mask.values.reshape(grid_shape)

# ─── Initialize Outputs ───────────────────────────────────────────────────
nep_stack = []
p_stack = []

variogram_params = {
    "ind3_4":   (3500, 0.35, 0.00),
    "ind34":    (3319, 0.22, 0.09),
    "ind50":    (1190, 0.24, 0.02),
    "ind100":   (1800, 0.09, 0.09),
    "ind500":   (1300, 0.02, 0.02),
    "ind1000":  (3500, 0.03, 0.01),
    "ind1500":  (3500, 0.03, 0.01),
}

# ─── 1–2. Variogram Fit & Kriging ─────────────────────────────────────────
for i, col in enumerate(cols):
    df = uumu.dropna(subset=[col])
    x, y, v = df["x"].values, df["y"].values, df[col].values

    if col in variogram_params:
        # Use manual parameters
        rng, sill, nugget = variogram_params[col]
        print(f"{col}: using manual parameters → range={rng}, sill={sill}, nugget={nugget}")

        # Optional: plot empirical variogram to check fit
        V = Variogram(np.c_[x, y], v, model='spherical', maxlag=3500, normalize=False, use_nugget=True)
        fig = V.plot(show=False)
        plt.title(f"Empirical Variogram: {col} (manual params used)")

        textstr = f"Range: {rng:.0f} \nSill: {sill:.3f}\nNugget: {nugget:.3f}"
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.show()

    else:
        # Fit variogram automatically
        V = Variogram(np.c_[x, y], v,
                      model='spherical', maxlag=3500,
                      normalize=False, use_nugget=True)
        rng, sill, nugget = V.parameters
        print(f"{col}: Fitted → range={rng:.2f}, sill={sill:.2f}, nugget={nugget:.2e}")

        fig = V.plot(show=False)
        plt.title(f"Fitted Variogram: {col}")
        plt.tight_layout()
        plt.show()

    # Kriging
    OK = OrdinaryKriging(x, y, v,
                         variogram_model='spherical',
                         variogram_parameters=[(sill+nugget), rng, nugget],
                         verbose=True, enable_plotting=True)
    z, _ = OK.execute("grid", xgrid, ygrid)
    z = np.clip(z, 0, 1)

    z_masked = np.where(mask, z, np.nan)
    nep_stack.append(z_masked)

    # Plot NEP
    plt.figure(figsize=(6, 5))
    plt.imshow(z_masked, extent=aoi_bounds, origin="lower", cmap="viridis")
    plt.title(f"NEP: {thresholds[i]} µg/L")
    plt.colorbar(label="Probability")
    plt.tight_layout()
    plt.show()

nep_stack = np.array(nep_stack)  # shape (7, rows, cols)

# ─── 3. Bin Probabilities (p₁ to p₇) ───────────────────────────────────────
p_stack = np.empty_like(nep_stack)
p_stack[0] = nep_stack[0]
for i in range(1, len(thresholds)):
    p_stack[i] = np.maximum(nep_stack[i] - nep_stack[i - 1], 0)

# Plot bin probabilities
for i in range(len(thresholds)):
    plt.figure(figsize=(6, 5))
    plt.imshow(p_stack[i], extent=aoi_bounds, origin="lower", cmap="plasma")
    plt.title(f"Bin p{i+1}")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# ─── 4. MIK Mean ──────────────────────────────────────────────────────────
mean_raster = np.nansum(p_stack * z_mid[:, None, None], axis=0)

plt.figure(figsize=(6, 5))
plt.imshow(mean_raster, extent=aoi_bounds, origin="lower", cmap="YlGnBu")
plt.title("MIK Mean Concentration")
plt.colorbar(label="µg/L")
plt.tight_layout()
plt.show()

# ─── 5. Conditional Variance ──────────────────────────────────────────────
var = np.nansum(p_stack * (z_mid[:, None, None] - mean_raster) ** 2, axis=0)
var = np.maximum(var, 0)

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import cm

# AOI extent
xmin, ymin, xmax, ymax = 563996.125, 132498.1563, 572249.9375, 138998.9219
extent = [xmin, xmax, ymin, ymax]

# ─── 1. Define MIK-CV Bins & Colormap ────────────────────────────────
cv_bins = [0, 25000, 50000, 75000, 100000, 150000, 200000, 250000,
           300000, 350000]
cmap = plt.get_cmap("coolwarm", len(cv_bins) - 1)
norm = mcolors.BoundaryNorm(boundaries=cv_bins, ncolors=cmap.N)

# ─── 2. Define concentration bin edges and labels ─────────────────────
thresholds = [3.4, 34, 50, 100, 500, 1000, 1500]
bin_labels = ["≤3.4", "3.4–34", "34–50", "50–100", "100–500",
              "500–1000", "1000–1500", ">1500"]
bins = [0] + thresholds + [np.inf]

# Define custom colors to match legend
bin_colors = [
    "black",      # ≤3.4
    "#1f78b4",    # 3.4–34 (blue)
    "#a6cee3",    # 34–50 (light blue)
    "#33a02c",    # 50–100 (green)
    "#ffff33",    # 100–500 (yellow)
    "#fb9a99",    # 500–1000 (orange-pink)
    "#e31a1c",    # 1000–1500 (red)
    "#b10026",    # >1500 (dark red)
]

# ─── 3. Prepare Data ─────────────────────────────────────────────────
data_df = uumu.dropna(subset=["avg_ctet_2018_2020", "x", "y"]).copy()
conc = data_df["avg_ctet_2018_2020"].values
bin_ids = np.digitize(conc, bins, right=True)
data_df["conc_bin"] = bin_ids
data_df["conc_bin_label"] = [bin_labels[i - 1] for i in bin_ids]

# ─── 4. Plot Raster + Points ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

# Background: MIK-CV
cv_im = ax.imshow(var, extent=extent, origin="lower",
                  cmap=cmap, norm=norm, alpha=0.7, zorder=1)
cbar = plt.colorbar(cv_im, ax=ax, ticks=cv_bins[1:-1], shrink=0.75)
cbar.set_label("MIK Conditional Variance (µg/L)²")

# Contours
cs = ax.contour(var, levels=cv_bins[1:-2], linewidths=0.7, colors='black',
                extent=extent, origin='lower')
ax.clabel(cs, fmt="%1.0f", fontsize=8)

# Plot each bin separately to control style
for i, label in enumerate(bin_labels, start=1):
    subset = data_df[data_df["conc_bin"] == i]
    if i == 1:
        # ≤3.4: plot as black Xs
        ax.scatter(subset["x"], subset["y"], marker="x", color="black",
                   label=label, zorder=3, s=60)
    else:
        ax.scatter(subset["x"], subset["y"], color=bin_colors[i - 1],
                   edgecolor="k", s=60, label=label, zorder=3)

# Legend
handles = []
for i, label in enumerate(bin_labels):
    if i == 0:
        handles.append(plt.Line2D([], [], marker='x', linestyle='None', color='black', label=label))
    else:
        handles.append(plt.Line2D([], [], marker='o', linestyle='None',
                                  markerfacecolor=bin_colors[i], markeredgecolor='k', label=label))
ax.legend(handles=handles, title="CTET Concentration (µg/L)",
          loc="upper right", fontsize=9)

# Final Touches
ax.set_title("MIK-CV Distribution with Measured CTET Concentrations", fontsize=14)
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")
plt.tight_layout()
plt.show()

# Constants
ROD_CLEANUP_LEVEL = 3.4  # µg/L

# Load RCL grid nodes
rcl_df = pd.read_csv(rcl_path)
grid_gdf = gpd.read_file(grid_path)

# Ensure correct data types
rcl_df["row"] = rcl_df["row"].astype(int)
rcl_df["col"] = rcl_df["col"].astype(int)
grid_gdf["row"] = grid_gdf["row"].astype(int)
grid_gdf["column"] = grid_gdf["column"].astype(int)

# Merge to get geometry of matching cells
rcl_gdf = pd.merge(rcl_df, grid_gdf, left_on=["row", "col"], right_on=["row", "column"])
rcl_gdf = gpd.GeoDataFrame(rcl_gdf, geometry="geometry", crs=grid_gdf.crs)

# Get centroid of each RCL grid cell
rcl_gdf["x"] = rcl_gdf.geometry.centroid.x
rcl_gdf["y"] = rcl_gdf.geometry.centroid.y

# Convert to raster index
x_idx = ((rcl_gdf["x"] - xgrid[0]) / grid_res).round().astype(int)
y_idx = ((rcl_gdf["y"] - ygrid[0]) / grid_res).round().astype(int)

# Clip to bounds
x_idx = x_idx.clip(0, var.shape[1] - 1)
y_idx = y_idx.clip(0, var.shape[0] - 1)

# Sample MIK-CV at those grid points
mik_cv_vals = var[y_idx, x_idx]
sn_vals = np.sqrt(mik_cv_vals) / ROD_CLEANUP_LEVEL

# Apply scoring per Table 4-7
def sn_score(sn):
    if np.isnan(sn):
        return np.nan
    elif sn < 30:
        return 0
    elif sn < 50:
        return 1
    elif sn < 70:
        return 2
    elif sn < 90:
        return 3
    else:
        return 4

rcl_gdf["MIK_CV"] = mik_cv_vals
rcl_gdf["S_N"] = sn_vals
rcl_gdf["S_CV"] = rcl_gdf["S_N"].apply(sn_score)

# ─── 7. Plot Score Spatially ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

score_cmap = plt.cm.get_cmap("RdYlBu_r", 5)
score_norm = plt.Normalize(vmin=0, vmax=4)

# Plot scored nodes
rcl_gdf.plot(column="S_CV", cmap=score_cmap, norm=score_norm,
             markersize=60, ax=ax, edgecolor="none", legend=True,
             legend_kwds={"label": "MIK-CV Score (S_CV)", "shrink": 0.7})

# AOI overlay for context
aoi.boundary.plot(ax=ax, color="black", linewidth=1)

ax.set_title("Scored RCL Nodes by MIK-CV (S_CV)", fontsize=14)
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")
plt.tight_layout()
plt.show()

from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import matplotlib.pyplot as plt

#MIK COV
# ─── 1. Clip mean to avoid division by 0 ───────────────────────────────────────
mean_raster_safe = np.clip(mean_raster, 1e-6, None)
mik_cov = np.sqrt(var) / mean_raster_safe

# ─── 2. Define bins for MIK-COV (%) classification ────────────────────────────
cov_bins_mikcov = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # For plotting
cov_labels = ["<1", "1.1–2", "2.1–3", "3.1–4", "4.1–5", "5.1–6", "6.1–7", ">7"]
cov_colors = [
    "#c6dbef",  # <1       - light blue
    "#9ecae1",  # 1.1–2    - sky blue
    "#6baed6",  # 2.1–3    - turquoise
    "#ffffb2",  # 3.1–4    - light yellow
    "#fecc5c",  # 4.1–5    - orange
    "#fd8d3c",  # 5.1–6    - red-orange
    "#f03b20",  # 6.1–7    - red
    "#bd0026",  # >7       - dark red
]
cmap_classified = ListedColormap(cov_colors)
norm_classified = BoundaryNorm(cov_bins_mikcov, cmap_classified.N)  # ✅ USE CORRECT BINS HERE

# ─── 3. Define bins for scoring (if needed separately) ─────────────────────────
cov_bins_score = [0, 30, 50, 70, 90, np.inf]  # For scoring
cov_scores = np.array([0, 1, 2, 3, 4])
cov_score_raster = np.digitize(mik_cov * 100, bins=cov_bins_score, right=False) - 1
cov_score_raster = np.where(np.isnan(mik_cov), np.nan, cov_scores[cov_score_raster])

fig, ax = plt.subplots(figsize=(10, 8))

# Background MIK-COV
img = ax.imshow(mik_cov, extent=extent, origin="lower",
                cmap=cmap_classified, norm=norm_classified)

# Colorbar
cbar = plt.colorbar(img, ax=ax, shrink=0.75)
tick_locs = [(cov_bins_mikcov[i] + cov_bins_mikcov[i + 1]) / 2 for i in range(len(cov_bins_mikcov) - 1)]
cbar.set_ticks(tick_locs)
cbar.set_ticklabels(cov_labels)
cbar.set_label("MIK-COV (%)")

# Overlay Measured CTET Concentrations
for i, label in enumerate(bin_labels, start=1):
    subset = data_df[data_df["conc_bin"] == i]
    if i == 1:
        ax.scatter(subset["x"], subset["y"], marker="x", color="black",
                   label=label, zorder=3, s=60)
    else:
        ax.scatter(subset["x"], subset["y"], color=bin_colors[i - 1],
                   edgecolor="k", s=60, label=label, zorder=3)

# Legend
handles = []
for i, label in enumerate(bin_labels):
    if i == 0:
        handles.append(plt.Line2D([], [], marker='x', linestyle='None',
                                  color='black', label=label))
    else:
        handles.append(plt.Line2D([], [], marker='o', linestyle='None',
                                  markerfacecolor=bin_colors[i],
                                  markeredgecolor='k', label=label))
ax.legend(handles=handles, title="CTET Concentration (µg/L)",
          loc="upper right", fontsize=9)

# Final Touches
ax.set_title("Classified MIK-COV (%) with Measured CTET Concentrations", fontsize=14)
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")
plt.tight_layout()
plt.show()

# ─── Multiply MIK-COV by 10 ────────────────────────────────────────────────────
mik_cov_x10 = mik_cov * 10  # Now on the 0–100+ scale like Table 4-8

# ─── Define MIK-COV Scoring Bins (Table 4-8) ───────────────────────────────────
score_bins = [0, 30, 50, 70, 90, 999]
score_values = np.array([0, 1, 2, 3, 4])
bin_idx = np.digitize(mik_cov_x10, bins=score_bins, right=False) - 1
bin_idx = np.clip(bin_idx, 0, len(score_values) - 1)  # Prevent index error
score_raster = np.where(np.isnan(mik_cov_x10), np.nan, score_values[bin_idx])


# ─── Define Colormap for Scores ────────────────────────────────────────────────
score_colors = [
    "#c6dbef",  # Score 0: light blue
    "#9ecae1",  # Score 1: sky blue
    "#6baed6",  # Score 2: medium blue
    "#31a354",  # Score 3: green
    "#006837",  # Score 4: dark green
]
score_labels = ["<30 → 0", "30–49 → 1", "50–69 → 2", "70–89 → 3", "≥90 → 4"]
cmap_score = ListedColormap(score_colors)
norm_score = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)

# ─── Plot the Score Raster + Measured Points ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

# Raster: MIK-COV Score
score_img = ax.imshow(score_raster, extent=extent, origin="lower",
                      cmap=cmap_score, norm=norm_score, alpha=0.6)

# Colorbar
cbar = plt.colorbar(score_img, ax=ax, shrink=0.75, ticks=[0, 1, 2, 3, 4])
cbar.set_ticklabels(score_labels)
cbar.set_ticklabels(score_labels)
cbar.set_label("MIK-COV Score (0–4)")

# Overlay: Measured CTET Concentrations
for i, label in enumerate(bin_labels, start=1):
    subset = data_df[data_df["conc_bin"] == i]
    if i == 1:
        ax.scatter(subset["x"], subset["y"], marker="x", color="black",
                   label=label, zorder=3, s=60)
    else:
        ax.scatter(subset["x"], subset["y"], color=bin_colors[i - 1],
                   edgecolor="k", s=60, label=label, zorder=3)

# Legend
handles = []
for i, label in enumerate(bin_labels):
    if i == 0:
        handles.append(plt.Line2D([], [], marker='x', linestyle='None',
                                  color='black', label=label))
    else:
        handles.append(plt.Line2D([], [], marker='o', linestyle='None',
                                  markerfacecolor=bin_colors[i],
                                  markeredgecolor='k', label=label))
ax.legend(handles=handles, title="CTET Concentration (µg/L)",
          loc="upper right", fontsize=9)

# Final Touches
ax.set_title("MIK-COV Score Raster (Table 4-8) with Measured CTET Points", fontsize=14)
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")
plt.tight_layout()
plt.show()
