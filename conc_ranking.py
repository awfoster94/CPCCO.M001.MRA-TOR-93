"""
Multi-COC Cleanup Level Exceedance (CLE) Scoring & Mapping with QA
------------------------------------------------------------------

Computes CLE scores (Table 4-6 raw scores + Eq. 3 average) for multiple COCs
(HexChromium, TC99, CTET_600y). Produces per-COC CSVs, CLE maps (full+zoom),
and QA/QC summary outputs to help reconcile with report figures.


QA options (set below):
  * write per-cell exceedance counts per subperiod
  * write TOTIM-of-maximum per subperiod
  * OU-level CLE summaries
  * optional discrete binned maps to emulate report symbology
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import flopy
from flopy.plot import PlotMapView
import geopandas as gpd
from shapely.geometry import Point

# =============================================================================
# --- flags -------------------------------------------------
# =============================================================================
WRITE_QC_COUNTS   = True    # write cntE/cntA (# of timesteps >= cleanup per subperiod)
WRITE_QC_MAXTIMES = False   # write TOTIM (days) when max occurred (bigger CSV)
WRITE_OU_SUMMARY  = True    # summarize CLE stats by OU polygon
BINNED_MAPS       = False    # plot CLE in 5 bins (0,>0-1,>1-2,>2-3,>3-4) to mimic report
PAD_CELLS_ZOOM    = 1       # # of extra rows/cols around cell list for zoom
MODEL_START_DATE  = "2015-01-01"

# =============================================================================
# Paths
# =============================================================================
FLOW_ROOT   = os.path.join("FnT_Models_from_TO-067", "Flow")
TRANS_ROOT  = os.path.join("FnT_Models_from_TO-067", "Transport")

rank_csv    = "ranks.csv"
cleanup_csv = "final_cleanup_levels.csv"
rcl_csv     = os.path.join("data_gap", "data_gap_RCs.csv")

basemap     = os.path.join("..", "02_data", "GIS", "shp")
ou_path     = os.path.join(basemap, "OUs", "ZP1_UP1.shp")
grid_path   = os.path.join(basemap, "grid_274", "grid_274.shp")

out_root    = "outputs_coc_CLE_QC"   # results root

# =============================================================================
# COC configuration
# =============================================================================
COC_SPECS = {
    "HexChromium": dict(folder="HexChromium", prefix="HexChromium",
                        cleanup_pattern="hexavalent", label="HexCr"),
    "TC99":        dict(folder="TC99",        prefix="TC99",
                        cleanup_pattern="tc-99",      label="Tc‑99"),
    "CTET_600y":   dict(folder="CTET_600y",
                        prefix="P2Rv8.3_CTET_600y",   # <<< fixed
                        cleanup_pattern="ctet",
                        label="CTET (600y)"),
}
# =============================================================================
# Layer group definitions (0-based)
# =============================================================================
LAYERS_RINGOLD_E = [2, 3, 4]  # model layers 3–5
LAYERS_RINGOLD_A = [6]        # model layer 7

# =============================================================================
# Plot styling
# =============================================================================
CMAP_CLE_CONT = "viridis"   # for continuous maps
VMIN_CLE      = 0.0
VMAX_CLE      = 4.0

# binned cmap (to mimic report) if BINNED_MAPS=True
CLE_CLASS_COLORS  = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
CLE_CLASS_BOUNDS  = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # center each integer bin
CLE_CLASS_CMAP    = mcolors.ListedColormap(CLE_CLASS_COLORS)
CLE_CLASS_NORM    = mcolors.BoundaryNorm(CLE_CLASS_BOUNDS, CLE_CLASS_CMAP.N)
CLE_CLASS_TICKS   = [0, 1, 2, 3, 4]
CLE_CLASS_LABELS  = ["0", "0–1", "1–2", "2–3", "3–4"]

# OU overlay style
OU_EDGECOLOR  = "k"
OU_LW         = 1.0
OU_FACECOLOR  = "none"
OU_ZORDER     = 10

# OU label style
OU_LABEL_FIELD = "Name"
OU_LABELSIZE   = 8
OU_LABELCOLOR  = "k"
OU_LABELZ      = OU_ZORDER + 1
OU_LABELBBOX   = dict(facecolor="white", alpha=0.4, edgecolor="none", pad=1.0)

# Optional grid shapefile overlay (QA only)
SHOW_GRID_SHP   = False
GRID_EDGECOLOR  = "0.3"
GRID_LW         = 0.25

# =============================================================================
# Helpers
# =============================================================================
def get_cleanup_threshold(cleanup_df, pattern: str) -> float:
    pat = pattern.lower()
    mask = cleanup_df["COC"].str.lower().str.contains(pat, na=False)
    if not mask.any():
        raise ValueError(f"No cleanup row matching pattern '{pattern}' in cleanup table.")
    return float(cleanup_df.loc[mask, "Final Cleanup Level"].iloc[0])

def find_ucn_file(ws: str, prefix: str) -> str:
    cand = os.path.join(ws, f"{prefix}.UCN")
    if os.path.isfile(cand):
        return cand
    ucn_list = glob.glob(os.path.join(ws, "*.UCN"))
    if not ucn_list:
        raise FileNotFoundError(f"No UCN files found in {ws}")
    print(f"  [warn] expected {prefix}.UCN not found; using {os.path.basename(ucn_list[0])}")
    return ucn_list[0]

def get_zoom_extent(mg, i_idx, j_idx, pad_cells=1):
    imin = max(0, i_idx.min() - pad_cells)
    imax = min(mg.nrow - 1, i_idx.max() + pad_cells)
    jmin = max(0, j_idx.min() - pad_cells)
    jmax = min(mg.ncol - 1, j_idx.max() + pad_cells)
    xv = mg.xvertices; yv = mg.yvertices
    xs = xv[imin:imax + 2, jmin:jmax + 2]; ys = yv[imin:imax + 2, jmin:jmax + 2]
    return (xs.min(), xs.max(), ys.min(), ys.max())

def plot_overlays(ax, ou_gdf, grid_gdf, label_ous=True):
    if ou_gdf is not None and not ou_gdf.empty:
        ou_gdf.boundary.plot(
            ax=ax, edgecolor=OU_EDGECOLOR, linewidth=OU_LW,
            facecolor=OU_FACECOLOR, zorder=OU_ZORDER,
        )
        if label_ous and OU_LABEL_FIELD in ou_gdf.columns:
            grouped = ou_gdf.dissolve(by=OU_LABEL_FIELD)
            reps = grouped.representative_point()
            for nm, pt in zip(grouped.index, reps):
                ax.text(pt.x, pt.y, str(nm), fontsize=OU_LABELSIZE,
                        color=OU_LABELCOLOR, ha="center", va="center",
                        zorder=OU_LABELZ, bbox=OU_LABELBBOX)
    if SHOW_GRID_SHP and grid_gdf is not None and not grid_gdf.empty:
        grid_gdf.boundary.plot(
            ax=ax, edgecolor=GRID_EDGECOLOR, linewidth=GRID_LW,
            facecolor="none", zorder=OU_ZORDER - 1,
        )

def bin_cle(arr):
    """Return integer bins 0..4 based on CLE_BIN_BOUNDS (right-open except last)."""
    binned = np.digitize(arr, CLE_BIN_BOUNDS, right=False) - 1  # 0..5 -> 0..4 (last bound)
    binned = np.clip(binned, 0, 4)
    return binned.astype(float)  # keep float for masked plotting; NaNs will be added separately

def cle_to_class(arr):
    """
    Map CLE values (0‑4 float) to integer classes 0,1,2,3,4 that match
    the legend buckets 0, 0‑1, 1‑2, 2‑3, 3‑4.

    0.00‑0.49 -> 0   | 0.50‑1.49 -> 1
    1.50‑2.49 -> 2   | 2.50‑3.49 -> 3
    3.50‑4.50 -> 4
    """
    bins = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    # digitize returns 1..len(bins); subtract 1 -> 0..4
    classes = np.digitize(arr, bins, right=False) - 1
    return classes.astype(float)         # keep float so NaNs stay NaN

# =============================================================================
# Per-COC processing
# =============================================================================
def process_coc(coc_key, cfg, mf, ranks_df, cleanup_df, rcl_df, ou_gdf, grid_gdf, mg, zoom_ext):
    """
    Compute CLE scores & maps for one COC (+QA).
    """
    folder          = cfg["folder"]
    prefix          = cfg["prefix"]
    cleanup_pattern = cfg["cleanup_pattern"]
    coc_label       = cfg.get("label", coc_key)

    print(f"\n=== Processing COC: {coc_key} ===")
    mt3d_ws  = os.path.join(TRANS_ROOT, folder)
    ucn_path = find_ucn_file(mt3d_ws, prefix)
    print(f"  UCN: {ucn_path}")

    ucn = flopy.utils.UcnFile(ucn_path, precision="double")
    times = np.asarray(ucn.get_times(), dtype=float)
    nt = times.size

    # TOTIM→calendar year
    t0 = pd.Timestamp(MODEL_START_DATE)
    years = np.array([(t0 + pd.Timedelta(days=float(dt))).year for dt in times], dtype=int)

    # cleanup threshold
    cleanup_val = get_cleanup_threshold(cleanup_df, cleanup_pattern)
    print(f"  Cleanup threshold ({coc_label}): {cleanup_val}")

    # cell list
    i_idx = rcl_df["i"].to_numpy()
    j_idx = rcl_df["j"].to_numpy()
    ncells = len(rcl_df)

    # accumulators for maxima & counts
    nwin = len(ranks_df)
    maxE = np.full((nwin, ncells), -np.inf, dtype=float)
    maxA = np.full((nwin, ncells), -np.inf, dtype=float)
    cntE = np.zeros((nwin, ncells), dtype=int)   # # timesteps >= cleanup
    cntA = np.zeros((nwin, ncells), dtype=int)
    if WRITE_QC_MAXTIMES:
        maxE_t = np.full((nwin, ncells), np.nan, dtype=float)  # TOTIM days
        maxA_t = np.full((nwin, ncells), np.nan, dtype=float)

    # inclusive last end-year
    end_adj = ranks_df["end_year"].to_numpy().copy()
    end_adj[-1] += 1

    # stream all timesteps
    print(f"  Streaming {nt} UCN slices ...")
    for it, (t, yr) in enumerate(zip(times, years)):
        arr = ucn.get_data(totim=t)
        arr = np.where(arr > 1e29, np.nan, arr)

        concE = np.nanmax(arr[LAYERS_RINGOLD_E, :, :], axis=0)
        concA = arr[LAYERS_RINGOLD_A[0], :, :]

        valsE = concE[i_idx, j_idx]
        valsA = concA[i_idx, j_idx]

        for w, rw in ranks_df.iterrows():
            if (rw.start_year <= yr) and (yr < end_adj[w]):
                # counts
                np.add(cntE[w], valsE >= cleanup_val, out=cntE[w])
                np.add(cntA[w], valsA >= cleanup_val, out=cntA[w])
                # maxima (track TOTIM if desired)
                gtE = valsE > maxE[w]
                if np.any(gtE):
                    maxE[w, gtE] = valsE[gtE]
                    if WRITE_QC_MAXTIMES:
                        maxE_t[w, gtE] = t
                gtA = valsA > maxA[w]
                if np.any(gtA):
                    maxA[w, gtA] = valsA[gtA]
                    if WRITE_QC_MAXTIMES:
                        maxA_t[w, gtA] = t
                break

        if it % max(1, nt // 10) == 0:
            print(f"    processed {it+1}/{nt} slices...")

    print("  Done streaming.")

    # --- raw subperiod scores ---
    score_vals = ranks_df["max_conc_greater_than_cleanup"].to_numpy(dtype=float)  # [4,3,2,1]
    rawE = np.where(maxE >= cleanup_val, score_vals[:, None], 0.0)
    rawA = np.where(maxA >= cleanup_val, score_vals[:, None], 0.0)

    # --- CLE averages (Eq.3) ---
    S_CLE_E = rawE.sum(axis=0) / nwin
    S_CLE_A = rawA.sum(axis=0) / nwin
    S_CLE_E_scaled = S_CLE_E * (4.0 / nwin)
    S_CLE_A_scaled = S_CLE_A * (4.0 / nwin)

    # --- QA summary printout ---
    print("  QA:")
    print(f"    timesteps: {nt} (min TOTIM={times.min():.1f} d, max TOTIM={times.max():.1f} d)")
    print(f"    calendar years: {years.min()}–{years.max()}")
    for w, rw in ranks_df.iterrows():
        ce = int((rawE[w] > 0).sum())
        ca = int((rawA[w] > 0).sum())
        print(f"    subperiod {rw.start_year}-{rw.end_year}: cells>=cleanup E={ce}, A={ca}")
    print(f"    overall max conc E={np.nanmax(maxE):.2g}  A={np.nanmax(maxA):.2g} (cleanup={cleanup_val})")

    # --- write CSV(s) ---
    outdir = os.path.join(out_root, coc_key)
    os.makedirs(outdir, exist_ok=True)
    out_df = rcl_df[["row", "col"]].copy()
    for w, rw in ranks_df.iterrows():
        lab = f"S_R_{rw.start_year}_{rw.end_year}"
        out_df[lab + "_E"] = rawE[w]
        out_df[lab + "_A"] = rawA[w]
        if WRITE_QC_COUNTS:
            out_df[lab + "_cntE"] = cntE[w]
            out_df[lab + "_cntA"] = cntA[w]
        if WRITE_QC_MAXTIMES:
            out_df[lab + "_maxTOTIM_E(d)"] = maxE_t[w]
            out_df[lab + "_maxTOTIM_A(d)"] = maxA_t[w]
    out_df["S_CLE_E"] = S_CLE_E
    out_df["S_CLE_A"] = S_CLE_A
    csv_path = os.path.join(outdir, f"{coc_key}_CLE_results_by_cell.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path}")

    # --- OU summary (optional) ---
    if WRITE_OU_SUMMARY:
        # cell center coords
        xc = mg.xcellcenters[i_idx, j_idx]
        yc = mg.ycellcenters[i_idx, j_idx]
        cell_gdf = gpd.GeoDataFrame(out_df.copy(),
                                    geometry=gpd.points_from_xy(xc, yc),
                                    crs=grid_gdf.crs)
        join = gpd.sjoin(cell_gdf, ou_gdf[[OU_LABEL_FIELD, "geometry"]],
                         how="left", predicate="within")
        grp = join.groupby(OU_LABEL_FIELD)
        recs = []
        for nm, g in grp:
            rec = dict(OU=nm, n_cells=len(g),
                       mean_CLE_E=g["S_CLE_E"].mean(),
                       mean_CLE_A=g["S_CLE_A"].mean(),
                       max_CLE_E=g["S_CLE_E"].max(),
                       max_CLE_A=g["S_CLE_A"].max())
            # % cells hitting each raw score level (E only; replicate for A if needed)
            for sc in [4,3,2,1]:
                rec[f"pct_cells_E_raw{sc}"] = 100.0 * (g[[c for c in g if c.startswith('S_R_') and c.endswith('_E')]].eq(sc).any(axis=1)).mean()
                rec[f"pct_cells_A_raw{sc}"] = 100.0 * (g[[c for c in g if c.startswith('S_R_') and c.endswith('_A')]].eq(sc).any(axis=1)).mean()
            recs.append(rec)
        ou_df = pd.DataFrame.from_records(recs)
        ou_csv = os.path.join(outdir, f"{coc_key}_CLE_summary_by_OU.csv")
        ou_df.to_csv(ou_csv, index=False)
        print(f"  Wrote {ou_csv}")

    # --- grid rasters for plotting ---
    # NOTE: S_CLE_*_scaled already 0..4; if BINNED_MAPS, map to class 0..4 ints
    arrE_cle = np.full((mg.nrow, mg.ncol), np.nan); arrE_cle[i_idx, j_idx] = S_CLE_E_scaled
    arrA_cle = np.full((mg.nrow, mg.ncol), np.nan); arrA_cle[i_idx, j_idx] = S_CLE_A_scaled

    if BINNED_MAPS:
        arrE_plot = np.full((mg.nrow, mg.ncol), np.nan)
        arrA_plot = np.full((mg.nrow, mg.ncol), np.nan)
        arrE_plot[i_idx, j_idx] = bin_cle(S_CLE_E_scaled)   # ← use bin_cle
        arrA_plot[i_idx, j_idx] = bin_cle(S_CLE_A_scaled)   # ← use bin_cle
    else:
        arrE_plot = arrE_cle
        arrA_plot = arrA_cle

    # --- plotting ---
    def plot_cle_map(grid_arr, title, fname, extent=None):
        masked = np.ma.masked_invalid(grid_arr)
        fig, ax = plt.subplots(figsize=(7, 6))
        pmv = PlotMapView(model=mf, ax=ax)
        if BINNED_MAPS:
            im = pmv.plot_array(masked, cmap=CLE_CLASS_CMAP, norm=CLE_CLASS_NORM)
        else:
            im = pmv.plot_array(masked, cmap=CMAP_CLE_CONT, vmin=VMIN_CLE, vmax=VMAX_CLE)
        pmv.plot_grid(lw=0.1, color="0.5")
        plot_overlays(ax, ou_gdf, grid_gdf, label_ous=True)
        if extent is not None:
            xmin, xmax, ymin, ymax = extent; ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(title)
        if BINNED_MAPS:
            cb = fig.colorbar(im, ax=ax, ticks=CLE_CLASS_TICKS, boundaries=CLE_CLASS_BOUNDS)
            cb.set_label(f"{coc_label} CLE class")
            cb.ax.set_yticklabels(CLE_CLASS_LABELS)
        else:
            cb = fig.colorbar(im, ax=ax)
            cb.set_label(f"{coc_label} Cleanup Level Exceedance Score (scaled 0–4)")
        fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)

    plot_cle_map(arrE_plot, f"{coc_label} Ringold E – CLE",        f"{coc_key}_E_CLE_full.png")
    plot_cle_map(arrA_plot, f"{coc_label} Ringold A – CLE",        f"{coc_key}_A_CLE_full.png")
    plot_cle_map(arrE_plot, f"{coc_label} Ringold E – CLE (zoom)", f"{coc_key}_E_CLE_zoom.png", extent=zoom_ext)
    plot_cle_map(arrA_plot, f"{coc_label} Ringold A – CLE (zoom)", f"{coc_key}_A_CLE_zoom.png", extent=zoom_ext)

    print(f"  Plots written to {outdir}")
    return out_df


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    os.makedirs(out_root, exist_ok=True)

    # load flow model (grid geometry)
    mf = flopy.modflow.Modflow.load(
        "P2Rv8.3_start2015_sp2024.nam",
        model_ws=FLOW_ROOT,
        version="mf2k",
        forgive=True,
        check=False,
    )

    # georef from shapefiles
    grid_gdf = gpd.read_file(grid_path)
    ou_gdf   = gpd.read_file(ou_path)

    # drop overlap OU
    if "Name" in ou_gdf.columns:
        drop_name = "Overlap: 200-ZP-1 and 200-BP-5"
        ou_gdf["Name"] = ou_gdf["Name"].astype(str).str.strip()
        ou_gdf = ou_gdf[~ou_gdf["Name"].str.fullmatch(drop_name, case=False, na=False)].copy()

    # match CRS
    if ou_gdf.crs != grid_gdf.crs:
        ou_gdf = ou_gdf.to_crs(grid_gdf.crs)

    # push georef into modelgrid (lower-left bounds; no rotation)
    llx, lly, urx, ury = grid_gdf.total_bounds
    mg = mf.modelgrid
    mg.set_coord_info(xoff=llx, yoff=lly, angrot=0.0, epsg=grid_gdf.crs.to_epsg())

    print("Model grid georeferenced.")
    print("  xoffset:", mg.xoffset, " yoffset:", mg.yoffset, " epsg:", mg.epsg)

    # ancillary tables (one copy reused for all COCs)
    ranks_df   = pd.read_csv(rank_csv).sort_values("start_year").reset_index(drop=True)
    cleanup_df = pd.read_csv(cleanup_csv)
    rcl_df     = pd.read_csv(rcl_csv)
    rcl_df["i"] = rcl_df["row"].astype(int) - 1
    rcl_df["j"] = rcl_df["col"].astype(int) - 1

    # zoom extent (same across COCs)
    zoom_ext = get_zoom_extent(mg, rcl_df["i"].to_numpy(), rcl_df["j"].to_numpy(), pad_cells=PAD_CELLS_ZOOM)

    # run all COCs
    for coc_key, cfg in COC_SPECS.items():
        process_coc(coc_key, cfg, mf, ranks_df, cleanup_df, rcl_df, ou_gdf, grid_gdf, mg, zoom_ext)

    print("\nAll COCs processed.")
