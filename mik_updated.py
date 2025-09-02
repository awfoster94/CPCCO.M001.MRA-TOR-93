import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pykrige.ok import OrdinaryKriging
from skgstat import Variogram
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
from skgstat import Variogram
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from pyproj import CRS
import shapely.geometry as sgeom
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from skgstat import models as skg_models
# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
data_path = os.path.join("..", "data")
gis_path = os.path.join("gis")
aoi_path = os.path.join(gis_path, "shp", "OUs", "zp1_up1_outline.shp")
ou_path = os.path.join(gis_path, "shp", "OUs", "ZP1_UP1.shp")
rcl_path = os.path.join("data_gap_RCs.csv")
grid_path = os.path.join(gis_path, "shp", "model_grid", "grid_274.shp")
rds_path = os.path.join(gis_path, "shp", "basemaps", "trvehrcl_buffer15m.shp")
processed_root = os.path.join(data_path, "processed_data") 
maps_root_default = os.path.join(processed_root, "maps")

# ──────────────────────────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{v:.15g}"

def _bin_representatives(thresholds: list[float]) -> np.ndarray:
    th = np.asarray(thresholds, dtype=float)
    reps = [0.5 * th[0]]  # first bin: ≤ t1
    for i in range(1, len(th)):
        reps.append(0.5 * (th[i-1] + th[i]))  # middles
    step = th[-1] - th[-2] if len(th) > 1 else th[0]  # simple extrapolation
    reps.append(th[-1] + 0.5 * step)  # open top bin: > tN
    return np.asarray(reps, dtype=float)

def infer_units_from_constituent(name: str) -> str:
    n = (name or "").strip().lower()
    if n in {"tc99", "tc-99", "tc 99", "technetium", "technetium-99"}:
        return "pCi/L"
    # default for everything else (CTET, HexCr, etc.)
    return "µg/L"

def _auto_point_colors(n_bins: int) -> list[str]:
    # colors for bins 2..n (bin 1 uses black "x")
    cmap = plt.get_cmap("tab20")
    picks = [1,3,5,7,9,11,13,15,17,19]
    cols = [mcolors.to_hex(cmap(i)) for i in picks]
    while len(cols) < (n_bins - 1):
        cols += cols
    return cols[: (n_bins - 1)]

def _sample_cmap_even(cmap="RdYlBu_r", n=5, start=0.00, end=1.00):
    """
    Return n colors sampled evenly from a matplotlib colormap.
    start/end let you clip the very ends if they’re too light/dark.
    """
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    xs = np.linspace(start, end, n)
    return [mcolors.to_hex(cm(x)) for x in xs]

def _overlay_points(ax, res, point_colors=None, point_size=20, z=6,
                    units_header=None, color_mode="discrete_even",
                    point_cmap="RdYlBu_r", cmap_span=(0.05, 0.95)):
    """
    color_mode:
      - 'discrete_even' -> one fixed color per bin, sampled evenly from `point_cmap`
      - 'continuous'    -> continuous gradient by concentration (uses point_cmap)
    """
    if "points" not in res or len(res["points"]) == 0:
        return []

    pts = res["points"].copy()
    th  = list(map(float, res.get("thresholds", [])))
    if len(th) == 0:
        return []

    bins = [-np.inf] + th + [np.inf]
    ids  = np.digitize(pts["conc"].to_numpy(float), bins, right=True)
    pts["conc_bin"] = ids
    n_bins = len(bins) - 1  # includes the ≤ first and > last bins

    def _fmt(v: float) -> str: return f"{v:.15g}"
    labels = [f"≤ {_fmt(th[0])}"] + [f"{_fmt(th[i-1])}–{_fmt(th[i])}" for i in range(1, len(th))] + [f"> {_fmt(th[-1])}"]

    handles = []
    if units_header:
        handles.append(Line2D([], [], linestyle="none", label=units_header))

    # bin 1: black "x"
    sub = pts[pts["conc_bin"] == 1]
    if not sub.empty:
        ax.scatter(sub["X"], sub["Y"], marker="x", color="black",
                   s=point_size, zorder=z)
        handles.append(Line2D([0],[0], marker="x", color="black", lw=0,
                              markersize=7, label=labels[0]))

    if color_mode == "continuous":
        # continuous gradient for bins 2..n (keeps your previous behavior)
        cm   = plt.get_cmap(point_cmap)
        vals = pts["conc"].astype(float).to_numpy()
        lo, hi = np.nanpercentile(vals, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        norm = mcolors.Normalize(vmin=lo, vmax=hi)

        for k in range(2, n_bins + 1):
            sub = pts[pts["conc_bin"] == k]
            if sub.empty: continue
            ax.scatter(sub["X"], sub["Y"],
                       c=sub["conc"].astype(float).to_numpy(),
                       cmap=cm, norm=norm, marker="o", s=point_size,
                       edgecolor="k", linewidth=0.7, zorder=z)
            # proxy color: mid of bin (use hi for the open top bin)
            lo_b, hi_b = bins[k-1], bins[k]
            rep = hi if not np.isfinite(hi_b) else 0.5*(lo_b + hi_b)
            proxy = cm(norm(rep))
            handles.append(Line2D([0],[0], marker="o", color="k", lw=0,
                                  markerfacecolor=proxy, markeredgewidth=0.7,
                                  markersize=6.5, label=labels[k-1]))
    else:
        # --- DISCRETE: evenly spaced colors across the colormap ---
        # one color per bin (bins 2..n); first bin stays 'x'
        n_colors = n_bins - 1
        if point_colors is None:
            start, end = cmap_span
            cols = _sample_cmap_even(point_cmap, n_colors, start, end)
        else:
            cols = point_colors
            if len(cols) < n_colors:
                # pad by repeating last color if too few provided
                cols = cols + [cols[-1]]*(n_colors - len(cols))

        for k in range(2, n_bins + 1):
            sub = pts[pts["conc_bin"] == k]
            if sub.empty: continue
            c = cols[k - 2]  # fixed color for this bin
            ax.scatter(sub["X"], sub["Y"], marker="o", s=point_size,
                       facecolor=c, edgecolor="k", linewidth=0.7, zorder=z)
            handles.append(Line2D([0],[0], marker="o", color="k", lw=0,
                                  markerfacecolor=c, markeredgewidth=0.7,
                                  markersize=6.5, label=labels[k-1]))
    return handles

def _left_gutter_panel(
    fig, *, handles, mappable, cb_label="µg/L",
    place_cb="left",                 # "left" or "right" of the legend
    leg_left=0.10, leg_bottom=0.12,  # legend axis (figure coords)
    leg_w=0.25, leg_h=0.26,
    cb_w=0.022,
    gap=0.030,                       # guaranteed spacing between legend & CB
    legend_pad=0.010,                # extra pad around measured legend bbox
    min_gap_to_map=0.012             # buffer from map edge
):
    # where the map starts (from subplots_adjust)
    map_left = fig.subplotpars.left

    # 1) Legend axis + legend
    leg_ax = fig.add_axes([leg_left, leg_bottom, leg_w, leg_h])
    leg_ax.axis("off")
    legend = leg_ax.legend(
        handles=handles, title="Legend", title_fontsize=10,
        loc="upper left", frameon=True, fontsize=9
    )

    # 2) Measure legend and expand by padding
    fig.canvas.draw()
    bbox = legend.get_window_extent(fig.canvas.get_renderer()) \
                 .transformed(fig.transFigure.inverted())
    x0 = bbox.x0 - legend_pad
    x1 = bbox.x1 + legend_pad

    # 3) Compute colorbar position with enforced gap and map clamp
    if place_cb == "right":
        cb_left = min(x1 + gap, map_left - cb_w - min_gap_to_map)
        tick_side = "right"   # ticks/label on side AWAY from legend
    else:  # place_cb == "left"
        cb_left = max(0.02, x0 - cb_w - gap)
        tick_side = "left"

    # 4) Create the colorbar
    cb_ax = fig.add_axes([cb_left, leg_bottom, cb_w, max(leg_h, 0.54)])
    cb = fig.colorbar(mappable=mappable, cax=cb_ax)
    cb.set_label(cb_label)

    # 5) Put ticks/label on side away from the legend
    if tick_side == "left":
        cb.ax.yaxis.set_label_position("left");  cb.ax.yaxis.tick_left()
    else:
        cb.ax.yaxis.set_label_position("right"); cb.ax.yaxis.tick_right()
    cb.ax.tick_params(pad=2)

    return leg_ax, cb_ax

def recenter_legend_in_left_gutter(fig, leg_ax, cb_ax,
                                   margin_cb=0.010, margin_map=0.012,
                                   vcenter=True):
    """
    Center the legend axis horizontally in the left gutter between the
    colorbar (right edge of cb_ax) and the map (fig.subplotpars.left).
    Keeps a small margin from both sides. If vcenter=True, vertically center too.
    """
    # measure available horizontal gutter span
    map_left = fig.subplotpars.left
    cb_right = cb_ax.get_position().x1

    usable_L = cb_right + margin_cb
    usable_R = map_left - margin_map
    span     = usable_R - usable_L

    pos = leg_ax.get_position()
    leg_w, leg_h = pos.width, pos.height

    # compute new left so legend is centered in the gutter
    new_left = usable_L + max(0.0, (span - leg_w) / 2.0)
    new_bottom = (0.5 - leg_h/2.0) if vcenter else pos.y0

    leg_ax.set_position([new_left, new_bottom, leg_w, leg_h])
    fig.canvas.draw_idle()

def _crs_is_feet(crs) -> bool:
    if crs is None:
        return False
    try:
        c = CRS.from_user_input(crs)
        for ax in c.axis_info:
            if "foot" in ax.unit_name.lower():
                return True
    except Exception:
        pass
    # fallback: string contains ft/us-ft
    try:
        if "ft" in crs.to_string().lower():
            return True
    except Exception:
        pass
    return False

def _slugify(s):
    return "".join(ch if ch.isalnum() else "_" for ch in str(s)).strip("_")

def _map_outdir(maps_root, constituent, assignment, sub=None):
    base = os.path.join(maps_root, _slugify(constituent), _slugify(assignment))
    if sub:
        base = os.path.join(base, sub)
    os.makedirs(base, exist_ok=True)
    return base

def _norm(s):
    return str(s).strip().upper() if pd.notna(s) else s

def load_points_for_mik(constituent: str, assignment: str,
                        include_characterization: bool = True,
                        processed_root: str = processed_root) -> pd.DataFrame:
    """
    Build point dataset for MIK using your processed CSVs.

    Returns a DataFrame with columns:
       NAME, Final_Assignment, TYPE, X (XCOORDS), Y (YCOORDS), conc
    """
    cdir = os.path.join(processed_root, constituent)
    f_mon = os.path.join(cdir, f"{constituent}_monitoring_avg_by_well.csv")
    f_char = os.path.join(cdir, f"{constituent}_characterization.csv")

    if not os.path.exists(f_mon):
        raise FileNotFoundError(f"Missing {f_mon}")
    df_mon = pd.read_csv(f_mon)

    # normalize & keep needed cols
    df_mon["Final_Assignment"] = df_mon["Final_Assignment"].astype(str).map(_norm)
    df_mon["NAME"] = df_mon["NAME"].astype(str)
    # avg concentration column name (we wrote it that way earlier)
    avg_col = f"{constituent}_avg_conc"
    if avg_col not in df_mon.columns:
        # fallback if needed
        if "avg_conc" in df_mon.columns:
            avg_col = "avg_conc"
        else:
            raise KeyError(f"Monitoring avg column {avg_col} not found.")

    # ensure coords exist
    for c in ["XCOORDS", "YCOORDS"]:
        if c not in df_mon.columns:
            raise KeyError(f"Expected {c} in {f_mon}")
    df_mon["X"] = pd.to_numeric(df_mon["XCOORDS"], errors="coerce")
    df_mon["Y"] = pd.to_numeric(df_mon["YCOORDS"], errors="coerce")
    mon = df_mon[["NAME", "Final_Assignment", "X", "Y", avg_col]].rename(columns={avg_col: "conc"})
    mon["TYPE"] = "MONITORING"

    parts = [mon]
    if include_characterization:
        if not os.path.exists(f_char):
            raise FileNotFoundError(f"Missing {f_char}")
        df_char = pd.read_csv(f_char)
        # normalize
        tcol = "TYPE" if "TYPE" in df_char.columns else ("Type" if "Type" in df_char.columns else None)
        if not tcol:
            raise KeyError("Characterization CSV missing TYPE/Type column")
        df_char["TYPE"] = df_char[tcol].astype(str).map(_norm)
        df_char = df_char[df_char["TYPE"] == "CHARACTERIZATION"].copy()
        df_char["Final_Assignment"] = df_char["Final_Assignment"].astype(str).map(_norm)
        for c in ["XCOORDS", "YCOORDS", "VAL"]:
            if c not in df_char.columns:
                raise KeyError(f"Expected {c} in {f_char}")
        df_char["X"] = pd.to_numeric(df_char["XCOORDS"], errors="coerce")
        df_char["Y"] = pd.to_numeric(df_char["YCOORDS"], errors="coerce")
        df_char["conc"] = pd.to_numeric(df_char["VAL"], errors="coerce")
        ch = df_char[["NAME", "Final_Assignment", "TYPE", "X", "Y", "conc"]].copy()
        parts.append(ch)

    pts = pd.concat(parts, ignore_index=True)
    pts = pts[pts["Final_Assignment"] == _norm(assignment)].copy()
    pts = pts.dropna(subset=["X", "Y", "conc"])
    return pts

def add_basemap_zoom(ax, ou_path, rds_path, pad_ft=300):
    """
    Plot Roads (underlay) and OU boundary (overlay), and zoom to OU bounds
    with a uniform pad. Returns (legend_handles, ft_units).
    """
    aoi = gpd.read_file(ou_path)          # OU boundaries
    roads = gpd.read_file(rds_path)

    if roads.crs != aoi.crs:
        roads = roads.to_crs(aoi.crs)

    ft_units = _crs_is_feet(aoi.crs)
    pad = float(pad_ft if ft_units else pad_ft * 0.3048)

    # --- ZOOM WINDOW: OU bounds + uniform pad ---
    xmin, ymin, xmax, ymax = aoi.total_bounds
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    # --- DRAW BASEMAP ---
    if not roads.empty:
        roads.plot(ax=ax, color="lightgray", linewidth=0.7, zorder=1)      # under raster
    # OU on TOP (two dashed passes)
    aoi.plot(ax=ax, facecolor="none", edgecolor="black",
             linestyle=(0, (5, 5)), linewidth=1.2, zorder=3)
    aoi.plot(ax=ax, facecolor="none", edgecolor="0.55",
             linestyle=(0, (2, 4)), linewidth=1.0, zorder=3)

    # legend handles
    handles = [
        Line2D([0], [0], color="lightgray", lw=2, label="Roads"),
        Line2D([0], [0], color="black", lw=1.5, ls=(0, (5, 5)),
               label="Groundwater Operable Unit"),
    ]
    return handles, ft_units

def _add_scale_bar(ax, ft_units=True, pad_frac=0.04, height_frac=0.01, text=""):
    """Draw a simple scalebar at bottom-left in data coords."""
    (xmin, xmax) = ax.get_xlim()
    (ymin, ymax) = ax.get_ylim()
    W = xmax - xmin
    H = ymax - ymin

    L = _nice_scale_length(W, ft_units=ft_units)     # bar length in map units
    bar_height = H * height_frac
    x0 = xmin + W * pad_frac
    y0 = ymin + H * pad_frac

    # bar (two-tone tick ends)
    ax.add_patch(mpatches.Rectangle((x0, y0), L, bar_height, fc="k", ec="k", zorder=20))
    ax.add_patch(mpatches.Rectangle((x0, y0), L/2, bar_height, fc="w", ec="k", zorder=21))

    unit_lbl = "ft" if ft_units else "m"
    ax.text(x0 + L/2, y0 + bar_height*2.2, f"{int(L)} {unit_lbl}",
            ha="center", va="bottom", fontsize=9, zorder=22)

def _nice_scale_length(width_units, ft_units=True):
    """Choose a nice scale bar length ~20–30% of axis width."""
    target = width_units * 0.25
    if ft_units:
        choices = [100, 200, 500, 1000, 2000, 5000, 10000]
    else:
        choices = [50, 100, 200, 500, 1000, 2000, 5000]
    return min(choices, key=lambda c: abs(c - target))

def plot_variograms_before_kriging(
    constituent: str,
    assignment: str,                  # "UU/MU" or "LU/CR"
    thresholds,                       # e.g. [3.4, 34, 50, 100, 500]
    model: str = "spherical",
    maxlag: float = 3500,
    n_lags: int = 15,
    use_nugget: bool = True,
    include_characterization: bool = True,
    save_plots: bool = True,
    processed_root: str = processed_root,   # same var you already use
):
    """
    Plots empirical variograms for each indicator. Returns a dict mapping
    'ind{threshold}' -> (partial_sill, range, nugget)  # <-- PyKrige order
    """
    pts = load_points_for_mik(
        constituent=constituent,
        assignment=assignment,
        include_characterization=include_characterization,
        processed_root=processed_root
    )
    if len(pts) == 0:
        raise ValueError(f"No points for {constituent} / {assignment} after filtering & cleaning.")

    labels = [f"ind{str(t).replace('.', '_')}" for t in thresholds]
    outdir = os.path.join(processed_root, constituent, "variograms", assignment.replace("/", "_"))
    if save_plots:
        os.makedirs(outdir, exist_ok=True)

    suggested = {}  # label -> (psill, range, nugget)

    print(f"\n--- Variogram fits for {constituent} / {assignment} ---")
    for t, lab in zip(thresholds, labels):
        df = pts.copy()
        # NEP indicator: 1 if conc ≤ t
        df["ind"] = (df["conc"] <= t).astype(float)
        x, y, v = df["X"].values, df["Y"].values, df["ind"].values

        V = Variogram(
            np.c_[x, y], v,
            model=model, maxlag=maxlag, n_lags=n_lags,
            normalize=False, use_nugget=use_nugget,
        )

        # scikit-gstat order:
        rng, psill, nug = V.parameters
        #psill = max(sill_total - nug, 1e-12)  # PyKrige wants partial sill

        suggested[lab] = (float(rng), float(psill), float(nug))

        # Plot + clear annotation
        fig = V.plot(show=False)
        plt.title(f"{constituent} {assignment} — Variogram ({lab})\n"
                  f"model={model}, maxlag={maxlag}, n_lags={n_lags}")
        txt = (f"scikit-gstat: sill_total={psill+nug:.4g}, range={rng:.0f}, nugget={nug:.4g}\n"
               f"PyKrige: partial_sill={psill:.4g}, range={rng:.0f}, nugget={nug:.4g}\n"
               f"N={len(v)}")
        plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes, va="top",
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
        plt.tight_layout()
        if save_plots:
            fout = os.path.join(outdir, f"{constituent}_{assignment.replace('/','_')}_{lab}_{model}.png")
            plt.savefig(fout, dpi=220)
        plt.show()

        print(f"{lab}: scikit-gstat -> (sill_total={psill+nug:.4g}, range={rng:.0f}, nugget={nug:.4g}) | "
              f"PyKrige -> (partial_sill={psill:.4g}, range={rng:.0f}, nugget={nug:.4g})  (N={len(v)})")

    print("\nPyKrige-ready params dict (copy into run_mik_for(..., variogram_params=...)):")
    print("{")
    for k, (psill, rng, nug) in suggested.items():
        print(f"    '{k}': ({psill:.6g}, {rng:.0f}, {nug:.6g}),")
    print("}")

    return suggested

def plot_variograms_with_final_params(
    constituent: str,
    assignment: str,
    thresholds: list[float],
    params_dict: dict,                 # {'ind3_4': (range, sill, nugget), ...}
    model: str = "spherical",
    maxlag: float = 3500,
    n_lags: int = 15,
    include_characterization: bool = True,
    save_plots: bool = True,
    processed_root: str = processed_root,
):
    """
    Re-makes variogram plots using your final parameters (range, sill, nugget).
    Saves to: <processed_root>/<constituent>/variograms_final/<assignment>/
    """
    # model function from scikit-gstat
    model = (model or "spherical").lower()
    model_funcs = {
        "spherical": skg_models.spherical,
        "exponential": skg_models.exponential,
        "gaussian": skg_models.gaussian,
        "stable": skg_models.stable,
        "matern": skg_models.matern,
    }
    if model not in model_funcs:
        raise ValueError(f"Unsupported model '{model}'. Pick one of {list(model_funcs)}")
    model_func = model_funcs[model]

    # load points as in your MIK workflow
    pts = load_points_for_mik(
        constituent=constituent,
        assignment=assignment,
        include_characterization=include_characterization,
        processed_root=processed_root
    )
    if len(pts) == 0:
        raise ValueError(f"No points for {constituent} / {assignment} after filtering & cleaning.")

    outdir = os.path.join(
        processed_root, constituent, "variograms_final", assignment.replace("/", "_")
    )
    if save_plots:
        os.makedirs(outdir, exist_ok=True)

    print(f"\n--- Final variograms for {constituent} / {assignment} ---")
    for t in thresholds:
        label = f"ind{str(t).replace('.', '_')}"
        if label not in params_dict:
            print(f"  [skip] {label} not found in params_dict")
            continue

        rng, sill, nug = params_dict[label]

        # indicator
        df = pts.copy()
        df["ind"] = (df["conc"] >= t).astype(float)
        x, y, v = df["X"].values, df["Y"].values, df["ind"].values

        # empirical variogram (we just use it to compute experimental points/bins)
        V = Variogram(
            np.c_[x, y], v,
            model="spherical",       # doesn't matter; we will draw our own curve
            maxlag=maxlag, n_lags=n_lags,
            normalize=False, use_nugget=True,
        )

        bins = V.bins
        gamma_exp = V.experimental

        # smooth curve of your final model across lag distances
        h = np.linspace(0.0, maxlag, 300)
        try:
            # most versions expect positional args
            gamma_fit = model_func(h, rng, sill, nug)
        except TypeError:
            # some newer versions accept a keyword
            gamma_fit = model_func(h, rng, sill, nugget=nug)

        # plot
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.plot(bins, gamma_exp, "o", ms=4.5, label="Empirical")
        ax.plot(h, gamma_fit, "-", lw=2.2, label=f"{model.title()} (final)")

        ax.set_title(f"{constituent} {assignment} — Variogram ({label})")
        ax.set_xlabel("Lag distance")
        ax.set_ylabel("Semivariance")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right")

        txt = (f"Final params:\n"
               f"range={rng:.0f}, sill={sill:.4g}, nugget={nug:.4g}\n"
               f"sill_total={sill+nug:.4g}, N={len(v)}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top",
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

        fig.tight_layout()
        if save_plots:
            fout = os.path.join(outdir, f"{constituent}_{assignment.replace('/','_')}_{label}_{model}_FINAL.png")
            fig.savefig(fout, dpi=220)
        plt.show()

def make_grid(aoi_gdf: gpd.GeoDataFrame, grid_res: float = 50):
    """Create a rectilinear grid covering the AOI polygon extent + mask inside AOI."""
    # bounds: [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = aoi_gdf.total_bounds
    xgrid = np.arange(xmin, xmax + grid_res, grid_res)
    ygrid = np.arange(ymin, ymax + grid_res, grid_res)
    xx, yy = np.meshgrid(xgrid, ygrid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_shape = xx.shape

    # AOI mask
    gpts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in grid_points], crs=aoi_gdf.crs)
    within = gpts.within(aoi_gdf.geometry.values.union_all())
    mask = within.values.reshape(grid_shape)

    extent = [xmin, xmax, ymin, ymax]  # correct order for imshow (L,R,B,T)
    return xgrid, ygrid, grid_shape, mask, extent

def _plot_nep_map(
    nep_raster, extent, *,
    title, pts_df, thresholds,
    cb_label="Probability",
    base_cmap="viridis", cmap_alpha=0.80,
    show_points=True, point_size=20,
    leg_left=0.12, leg_bottom=0.12, leg_w=0.26, leg_h=0.26,
    units="µg/L",
    # NEW
    save: bool = True,
    maps_root: str = maps_root_default,
    constituent: str | None = None,
    assignment: str | None = None,
    threshold: float | None = None
):
    """
    Styled NEP panel (like MIK-mean): basemap, left gutter colorbar+legend, scale bar.
    """
    # Make a temporary 'res' so _overlay_points can build bin labels
    res_for_points = {"points": pts_df, "thresholds": thresholds}

    # 1) Figure & map
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig.subplots_adjust(left=0.38, right=0.88, bottom=0.10, top=0.92)

    # transparent colormap so basemap peeks through; no per-artist alpha
    nep_cmap = _cmap_with_alpha(plt.get_cmap(base_cmap), alpha=cmap_alpha)
    img = ax.imshow(nep_raster, extent=extent, origin="lower",
                    cmap=nep_cmap, zorder=2, vmin=0.0, vmax=1.0)

    # 2) Basemap (roads + OU; zoomed)
    map_handles, ft_units = add_basemap_zoom(ax, ou_path=ou_path, rds_path=rds_path, pad_ft=300)

    # 3) Optional measured points (same discrete-even scheme as elsewhere)
    point_handles = []
    if show_points and pts_df is not None and len(pts_df):
        point_handles = _overlay_points(
            ax, res_for_points,
            point_size=point_size,
            units_header=f"Measured concentration ({units})",
            color_mode="discrete_even",
            point_cmap="RdYlBu_r",
            cmap_span=(0.06, 0.94),
            z=6
        )

    handles = map_handles + point_handles

    # 4) Left gutter (colorbar + legend)
    leg_ax, cb_ax = _left_gutter_panel(
        fig, handles=handles, mappable=img, cb_label=cb_label,
        place_cb="left",
        leg_left=leg_left, leg_bottom=leg_bottom, leg_w=leg_w, leg_h=leg_h,
        cb_w=0.022, gap=0.035, legend_pad=0.010, min_gap_to_map=0.012
    )
    recenter_legend_in_left_gutter(fig, leg_ax, cb_ax,
                                   margin_cb=0.010, margin_map=0.012, vcenter=True)

    # 5) Cosmetics
    _add_scale_bar(ax, ft_units=ft_units, pad_frac=0.05, height_frac=0.012)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    fig.text(0.5, 0.04, title, ha="center", va="bottom", fontsize=12)

    if save and (constituent is not None) and (assignment is not None) and (threshold is not None):
        outdir = _map_outdir(maps_root, constituent, assignment, sub="nep")
        fname  = f"{_slugify(constituent)}_{_slugify(assignment)}_NEP_at_{_fmt(threshold)}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=220, bbox_inches="tight")

    plt.show()

def run_mik_for(constituent: str,
                assignment: str,
                thresholds=None,
                variogram_params=None,
                grid_res: float = 50,
                include_characterization: bool = True,
                show_variogram: bool = False,
                save_nep_maps: bool = True,
                maps_root: str = maps_root_default):
    """
    thresholds: list of cutoffs (ascending). Example for CTET: [3.4, 34, 50, 100, 500, 1000, 1500]
    variogram_params: dict mapping threshold label -> (range, sill, nugget)
                      keys should be strings like: 'ind3_4', 'ind34', 'ind50', ...
                      If not provided for a threshold, automatic fit is used.
    """
    if thresholds is None:
        thresholds = [3.4, 34, 50, 100, 500, 1000, 1500]  # default: CTET bins
    z_mid = _bin_representatives(thresholds)
    units = infer_units_from_constituent(constituent)
    
    # 0) Load AOI + Points
    aoi = gpd.read_file(aoi_path)
    pts = load_points_for_mik(constituent, assignment,
                              include_characterization=include_characterization,
                              processed_root=processed_root)
    if len(pts) == 0:
        raise ValueError(f"No points for {constituent} / {assignment} after filtering & cleaning.")

    # 1) Grid + AOI mask
    xgrid, ygrid, grid_shape, mask, extent = make_grid(aoi, grid_res=grid_res)

    # 2) For each threshold → build indicator, variogram, OK → NEP (probability) on grid
    nep_stack = []
    cols_lbl = [f"ind{str(t).replace('.', '_')}" for t in thresholds]
    default_vario = {} if variogram_params is None else variogram_params

    for t, label in zip(thresholds, cols_lbl):
        df = pts.copy()
        # NEP indicator: 1 if conc ≤ t
        df["ind"] = (df["conc"] <= t).astype(float)

        x = df["X"].values
        y = df["Y"].values
        v = df["ind"].values

        # Variogram
        if label in default_vario:
            rng, sill, nugget = default_vario[label]
            if show_variogram:
                V = Variogram(np.c_[x, y], v, model="spherical", maxlag=3500,
                              normalize=False, use_nugget=True)
                fig = V.plot(show=False)
                plt.title(f"Empirical Variogram ({constituent} {assignment})\n{label} — using manual params")
                textstr = f"Range: {rng:.0f} \nSill: {sill:.3f}\nNugget: {nugget:.3f}"
                plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                         fontsize=10, va="top",
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
                plt.tight_layout(); plt.show()
        else:
            V = Variogram(np.c_[x, y], v, model="spherical", maxlag=3500,
                          normalize=False, use_nugget=True)
            rng, sill, nugget = V.parameters
            if show_variogram:
                fig = V.plot(show=False)
                plt.title(f"Fitted Variogram ({constituent} {assignment})\n{label}")
                plt.tight_layout(); plt.show()

        # Ordinary Kriging of indicator
        OK = OrdinaryKriging(
            x, y, v,
            variogram_model="spherical",
            variogram_parameters=[(sill + nugget), rng, nugget],  # same style you used
            verbose=False, enable_plotting=True
        )
        z, _ = OK.execute("grid", xgrid, ygrid)   # probability surface (0..1)
        z = np.clip(z, 0, 1)

        z_masked = np.where(mask, z, np.nan)
        nep_stack.append(z_masked)

        # Optional quick plot per threshold
        _plot_nep_map(
            z_masked, extent,
            title=f"{constituent} {assignment} — NEP @ {t} {units}",
            pts_df=pts, thresholds=thresholds,
            cb_label="Probability",
            base_cmap="viridis", cmap_alpha=0.80,
            show_points=True, point_size=20,
            units=units,
            # NEW
            save=save_nep_maps,
            maps_root=maps_root,
            constituent=constituent,
            assignment=assignment,
            threshold=t
        )
    nep_stack = np.array(nep_stack)  # shape (nT, rows, cols)

    # 3) Bin probabilities p_i = max(NEP_i - NEP_{i-1}, 0)
    nT = len(thresholds)
    p_stack = np.empty((nT + 1, *nep_stack[0].shape), dtype=float)
    p_stack[0] = np.clip(nep_stack[0], 0, 1)                    # ≤ t1
    for i in range(1, nT):                                      # t{i-1}–ti
        p_stack[i] = np.maximum(nep_stack[i] - nep_stack[i-1], 0.0)
    p_stack[-1] = np.maximum(1.0 - nep_stack[-1], 0.0)          # > tN

    # 4) MIK Mean (using z_mid vector)
    mean_raster = np.nansum(p_stack * z_mid[:, None, None], axis=0)

    # 5) Conditional Variance
    var = np.nansum(p_stack * (z_mid[:, None, None] - mean_raster) ** 2, axis=0)
    var = np.maximum(var, 0.0)

    results = {
        "points": pts,
        "aoi": aoi,
        "thresholds": thresholds,
        "z_mid": z_mid,                  # now N+1 long
        "xgrid": xgrid, "ygrid": ygrid,
        "extent": extent, "mask": mask,
        "nep_stack": nep_stack,
        "p_stack": p_stack,              # now N+1 bins
        "mean_raster": mean_raster,
        "var_raster": var,
        "units": units,
        "constituent": constituent,
        "assignment": assignment,
        "maps_root": maps_root,
    }
    
    return results

def _ecdf(y):
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return np.array([]), np.array([])
    x = np.sort(y)
    F = np.arange(1, x.size + 1) / x.size
    return x, F

def _prep_for_log(x):
    xp = np.asarray(x, dtype=float)
    if xp.size == 0:
        return xp
    pos = xp[xp > 0]
    if pos.size == 0:
        return xp + 1e-9
    eps = 0.5 * np.nanmin(pos)
    xp[xp <= 0] = eps
    return xp

def _summary_stats(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0, "min": np.nan, "p50": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "n": int(v.size),
        "min": float(np.min(v)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }

def plot_cdf_for_constituent(
    constituent: str,
    include_characterization: bool = True,
    logx: bool = True,
    thresholds: list[float] | None = None,                 # OLD: one list for both
    thresholds_by_assignment: dict | None = None,          # NEW: {"UU/MU":[...], "LU/CR":[...]}
    overlay: bool = True,
    save: bool = True,
    processed_root_dir: str = processed_root,
):
    """
    Plot ECDFs for a constituent, split by aquifer assignment.
    If thresholds_by_assignment is provided, it overrides thresholds and lets
    UU/MU and LU/CR use different cutoffs.

    Returns:
      {"figure": <png path or dict of paths>, "stats": DataFrame}
    """
    # --- units (Tc99 -> pCi/L; others -> µg/L) ---
    units = infer_units_from_constituent(constituent)

    # loader from your MIK step (monitoring avg + optional characterization)
    def _load(assignment):
        return load_points_for_mik(
            constituent=constituent,
            assignment=assignment,
            include_characterization=include_characterization,
            processed_root=processed_root_dir
        )

    pts_uumu = _load("UU/MU")
    pts_lucr = _load("LU/CR")

    conc_u = pts_uumu["conc"].astype(float).to_numpy()
    conc_l = pts_lucr["conc"].astype(float).to_numpy()

    x_u, F_u = _ecdf(conc_u)
    x_l, F_l = _ecdf(conc_l)
    if logx:
        x_u = _prep_for_log(x_u)
        x_l = _prep_for_log(x_l)

    # thresholds per assignment
    if thresholds_by_assignment is not None:
        th_u = thresholds_by_assignment.get("UU/MU", [])
        th_l = thresholds_by_assignment.get("LU/CR", [])
    else:
        th_u = th_l = (thresholds or [])

    outdir = os.path.join(processed_root_dir, constituent, "cdf")
    if save:
        os.makedirs(outdir, exist_ok=True)

    stats = pd.DataFrame.from_dict(
        {"UU/MU": _summary_stats(conc_u), "LU/CR": _summary_stats(conc_l)},
        orient="index",
    ).rename_axis("Assignment").reset_index()

    if overlay:
        fig, ax = plt.subplots(figsize=(8, 6))

        # ECDF curves
        if x_u.size:
            ax.step(x_u, F_u, where="post", label=f"UU/MU (n={np.isfinite(conc_u).sum()})")
        if x_l.size:
            ax.step(x_l, F_l, where="post", label=f"LU/CR (n={np.isfinite(conc_l).sum()})")

        # threshold lines (distinct styles)
        if th_u:
            for t in th_u:
                xt = max(t, 1e-12) if logx else t
                ax.axvline(xt, linestyle="--", linewidth=0.9, color="tab:blue", alpha=0.6)
        if th_l:
            for t in th_l:
                xt = max(t, 1e-12) if logx else t
                ax.axvline(xt, linestyle=":", linewidth=0.9, color="tab:orange", alpha=0.7)

        # legend bits for threshold lines (now with units)
        extra = []
        if th_u:
            extra.append(Line2D([0],[0], ls="--", color="tab:blue",
                                label=f"UU/MU thresholds ({units})"))
        if th_l:
            extra.append(Line2D([0],[0], ls=":", color="tab:orange",
                                label=f"LU/CR thresholds ({units})"))

        ax.set_title(f"{constituent} — Empirical CDF (ECDF)")
        ax.set_xlabel(f"Concentration ({units})")          # ← units applied
        ax.set_ylabel("F(x) = P(X ≤ x)")
        if logx:
            ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # combine handles
        leg1 = ax.legend(loc="lower right")
        if extra:
            ax.add_artist(leg1)
            ax.legend(handles=extra, loc="lower left")

        fpath = None
        if save:
            suffix = "_log" if logx else ""
            fpath = os.path.join(outdir, f"{constituent}_CDF_overlay{suffix}.png")
        fig.tight_layout()
        if save:
            fig.savefig(fpath, dpi=220)
        plt.show()

        return {"figure": fpath, "stats": stats}

    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # UU/MU panel
        ax = axes[0]
        if x_u.size:
            ax.step(x_u, F_u, where="post", label=f"UU/MU (n={np.isfinite(conc_u).sum()})")
        for t in th_u:
            xt = max(t, 1e-12) if logx else t
            ax.axvline(xt, linestyle="--", linewidth=0.9, color="tab:blue", alpha=0.6)
        ax.set_title(f"{constituent} — UU/MU ECDF")
        ax.set_xlabel(f"Concentration ({units})")          # ← units applied
        ax.set_ylabel("F(x) = P(X ≤ x)")
        if logx: ax.set_xscale("log")
        ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend()

        # LU/CR panel
        ax = axes[1]
        if x_l.size:
            ax.step(x_l, F_l, where="post", label=f"LU/CR (n={np.isfinite(conc_l).sum()})")
        for t in th_l:
            xt = max(t, 1e-12) if logx else t
            ax.axvline(xt, linestyle=":", linewidth=0.9, color="tab:orange", alpha=0.7)
        ax.set_title(f"{constituent} — LU/CR ECDF")
        ax.set_xlabel(f"Concentration ({units})")          # ← units applied
        if logx: ax.set_xscale("log")
        ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend()

        fpath = None
        if save:
            suffix = "_log" if logx else ""
            fpath = os.path.join(outdir, f"{constituent}_CDF_split{suffix}.png")
        fig.tight_layout()
        if save:
            fig.savefig(fpath, dpi=220)
        plt.show()

        return {"figure": fpath, "stats": stats}

def _cov_percent(res, clip=1e-6):
    """Return MIK–COV in percent: 100 * sqrt(var) / max(mean, clip)."""
    mean = np.asarray(res["mean_raster"], float)
    var  = np.asarray(res["var_raster"], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = 100.0 * np.sqrt(np.maximum(var, 0)) / np.clip(mean, clip, None)
    return cv

def plot_mik_cov(
    res,
    title_prefix="",
    bins_pct=None,
    n_classes=9,
    cmap_name="YlOrRd",
    show_points=True,
    point_size=20,
    point_cmap="RdYlBu_r",
    cmap_span=(0.06, 0.94),
    save: bool = True
):
    """
    Plot MIK-COV = sqrt(var) / mean (dimensionless if you don't multiply by 100).
    Classification here is only for display; scoring happens in plot_mik_cov_score.
    """
    extent   = res["extent"]
    units    = res.get("units", "µg/L")
    mean     = np.asarray(res["mean_raster"], dtype=float)
    var      = np.asarray(res["var_raster"],  dtype=float)

    mean_safe = np.clip(mean, 1e-6, None)
    cov_val   = np.sqrt(np.maximum(var, 0.0)) / mean_safe
    cov_val[~np.isfinite(mean)] = np.nan

    if bins_pct is None:
        bins_pct = _nice_bins(cov_val, n_classes=n_classes)
    bins_pct = np.array(bins_pct, dtype=float)
    bins_pct = np.unique(bins_pct)
    if bins_pct.size < 2:
        bins_pct = np.array([0.0, 1.0])

    cmap = plt.get_cmap(cmap_name, bins_pct.size - 1)
    norm = mcolors.BoundaryNorm(boundaries=bins_pct, ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig.subplots_adjust(left=0.38, right=0.88, bottom=0.10, top=0.92)

    img = ax.imshow(cov_val, extent=extent, origin="lower",
                    cmap=cmap, norm=norm, alpha=0.80, zorder=2)

    map_handles, ft_units = add_basemap_zoom(ax, ou_path=ou_path, rds_path=rds_path, pad_ft=300)

    point_handles = []
    if show_points:
        point_handles = _overlay_points(
            ax, res,
            point_size=point_size,
            units_header=f"Measured concentration ({units})",
            color_mode="discrete_even",
            point_cmap=point_cmap,
            cmap_span=cmap_span
        )

    all_handles = map_handles + point_handles

    leg_ax, cb_ax = _left_gutter_panel(
        fig, handles=all_handles, mappable=img,
        cb_label="MIK-COV",   # dimensionless unless you scaled by 100 upstream
        place_cb="left",
        leg_left=0.12, leg_bottom=0.12, leg_w=0.26, leg_h=0.26,
        cb_w=0.022, gap=0.035, legend_pad=0.010, min_gap_to_map=0.012
    )
    recenter_legend_in_left_gutter(fig, leg_ax, cb_ax,
                                   margin_cb=0.010, margin_map=0.012, vcenter=True)

    _add_scale_bar(ax, ft_units=ft_units, pad_frac=0.05, height_frac=0.012)
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    fig.text(0.5, 0.04, f"{title_prefix} MIK-COV", ha="center", va="bottom", fontsize=12)

    if save:
        constituent = res.get("constituent", "constituent")
        assignment  = res.get("assignment", "assignment")
        maps_root   = res.get("maps_root", maps_root_default)
        outdir = _map_outdir(maps_root, constituent, assignment, sub="cov")
        fname  = f"{_slugify(constituent)}_{_slugify(assignment)}_MIK_COV.png"
        fig.savefig(os.path.join(outdir, fname), dpi=220, bbox_inches="tight")

    plt.show()
        
def plot_mik_cov_score(
    res,
    title_prefix="",
    score_percentiles=(30, 50, 70, 90),  # percentile cut points; default Table 4-8
    scores=None,
    score_colors=None,
    show_points=True,
    point_size=20,
    point_cmap="RdYlBu_r",
    cmap_span=(0.06, 0.94),
    save: bool = True
):
    """
    Score the MIK-COV raster by *percentile ranges*:
      <p30 -> 0, p30–p50 -> 1, p50–p70 -> 2, p70–p90 -> 3, >=p90 -> 4
    """
    extent   = res["extent"]
    units    = res.get("units", "µg/L")
    mean     = np.asarray(res["mean_raster"], dtype=float)
    var      = np.asarray(res["var_raster"],  dtype=float)

    mean_safe = np.clip(mean, 1e-6, None)
    cov_val   = np.sqrt(np.maximum(var, 0.0)) / mean_safe
    cov_val[~np.isfinite(mean)] = np.nan

    # percentile cut points from the raster distribution
    cov_finite = cov_val[np.isfinite(cov_val)]
    if cov_finite.size == 0:
        cuts = np.array([np.inf, np.inf, np.inf, np.inf])
    else:
        cuts = np.nanpercentile(cov_finite, list(score_percentiles))

    if scores is None:
        scores = [0, 1, 2, 3, 4]
    n_classes = len(scores)

    # Build bins from percentiles: [-inf, p30, p50, p70, p90, +inf]
    bins = np.r_[-np.inf, cuts, np.inf]
    idx  = np.digitize(cov_val, bins=bins, right=False) - 1
    idx  = np.clip(idx, 0, n_classes - 1)
    score_raster = np.where(np.isnan(cov_val), np.nan, np.array(scores, dtype=float)[idx])

    # Colormap for scores
    if score_colors is None:
        score_colors = ["#c6dbef", "#9ecae1", "#6baed6", "#31a354", "#006837"]
    cmap = ListedColormap(score_colors[:n_classes])
    norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n_classes + 0.5, 1.0),
                                ncolors=cmap.N)

    # Labels showing numeric cut values
    def _tick_labels(pp, cuts):
        def _fmt(x):
            # compact formatting (2–3 sig figs)
            return f"{x:.3g}" if np.isfinite(x) else "—"
        p30,p50,p70,p90 = cuts
        return [f"<p30 (<{_fmt(p30)}) → 0",
                f"p30–p50 ({_fmt(p30)}–{_fmt(p50)}) → 1",
                f"p50–p70 ({_fmt(p50)}–{_fmt(p70)}) → 2",
                f"p70–p90 ({_fmt(p70)}–{_fmt(p90)}) → 3",
                f"≥p90 (≥{_fmt(p90)}) → 4"]

    tick_labels = _tick_labels(score_percentiles, cuts)

    # Figure & layout
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig.subplots_adjust(left=0.38, right=0.88, bottom=0.10, top=0.92)

    img = ax.imshow(score_raster, extent=extent, origin="lower",
                    cmap=cmap, norm=norm, alpha=0.80, zorder=2)

    # Basemap + zoom
    map_handles, ft_units = add_basemap_zoom(ax, ou_path=ou_path, rds_path=rds_path, pad_ft=300)

    # Optional measured points
    point_handles = []
    if show_points:
        point_handles = _overlay_points(
            ax, res,
            point_size=point_size,
            units_header=f"Measured concentration ({units})",
            color_mode="discrete_even",
            point_cmap=point_cmap,
            cmap_span=cmap_span
        )

    all_handles = map_handles + point_handles

    # Left gutter: legend + colorbar
    leg_ax, cb_ax = _left_gutter_panel(
        fig, handles=all_handles, mappable=img,
        cb_label="MIK-COV Score (0–4)",
        place_cb="left",
        leg_left=0.12, leg_bottom=0.12, leg_w=0.26, leg_h=0.26,
        cb_w=0.022, gap=0.035, legend_pad=0.010, min_gap_to_map=0.012
    )
    cb_ax.yaxis.set_ticks(range(n_classes))
    cb_ax.set_yticklabels(tick_labels)

    recenter_legend_in_left_gutter(fig, leg_ax, cb_ax,
                                   margin_cb=0.010, margin_map=0.012, vcenter=True)

    _add_scale_bar(ax, ft_units=ft_units, pad_frac=0.05, height_frac=0.012)
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    fig.text(0.5, 0.04, f"{title_prefix} MIK-COV Score", ha="center", va="bottom", fontsize=12)

    if save:
        constituent = res.get("constituent", "constituent")
        assignment  = res.get("assignment", "assignment")
        maps_root   = res.get("maps_root", maps_root_default)
        outdir = _map_outdir(maps_root, constituent, assignment, sub="cov_score")
        fname  = f"{_slugify(constituent)}_{_slugify(assignment)}_MIK_COV_score.png"
        fig.savefig(os.path.join(outdir, fname), dpi=220, bbox_inches="tight")

    plt.show()

def _bin_label_from_thresholds(thresholds, i, units="µg/L"):
    th = list(map(float, thresholds))
    if i == 0:
        return f"≤ {th[0]:g} {units}"
    if i < len(th):
        return f"{th[i-1]:g}–{th[i]:g} {units}"
    return f"> {th[-1]:g} {units}"

def plot_bin_probs(
    res,
    title_prefix="",
    cmap="plasma",
    show_points=True,
    point_size=20,
    point_cmap="RdYlBu_r",
    cmap_span=(0.06, 0.94),
    save: bool = True
):
    """
    Plot each bin probability for NEP bins:
      p1 = P(X ≤ t1), ..., pN = P(t{N-1} < X ≤ tN), p{N+1} = P(X > tN)
    """
    thresholds = res["thresholds"]
    units = res.get("units", "µg/L")
    p_stack    = res["p_stack"]   # shape: (N+1, rows, cols)
    extent     = res["extent"]
    cb_units   = "Probability"

    # NOTE: N+1 bins
    for i in range(len(thresholds) + 1):
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        fig.subplots_adjust(left=0.38, right=0.88, bottom=0.10, top=0.92)

        img = ax.imshow(
            p_stack[i], extent=extent, origin="lower",
            cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.80, zorder=2
        )

        map_handles, ft_units = add_basemap_zoom(ax, ou_path=ou_path, rds_path=rds_path, pad_ft=300)

        point_handles = []
        if show_points:
            point_handles = _overlay_points(
                ax, res,
                point_size=point_size,
                units_header=f"Measured concentration ({units})",
                color_mode="discrete_even",
                point_cmap=point_cmap,
                cmap_span=cmap_span
            )

        all_handles = map_handles + point_handles

        leg_ax, cb_ax = _left_gutter_panel(
            fig, handles=all_handles, mappable=img, cb_label=cb_units,
            place_cb="left",
            leg_left=0.12, leg_bottom=0.12, leg_w=0.26, leg_h=0.26,
            cb_w=0.022, gap=0.035, legend_pad=0.010, min_gap_to_map=0.012
        )
        recenter_legend_in_left_gutter(fig, leg_ax, cb_ax,
                                       margin_cb=0.010, margin_map=0.012, vcenter=True)

        _add_scale_bar(ax, ft_units=ft_units, pad_frac=0.05, height_frac=0.012)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]

        bin_label = _bin_label_from_thresholds(thresholds, i, units=units)
        fig.text(0.5, 0.04, f"{title_prefix} Bin p{i+1} — {bin_label}",
                 ha="center", va="bottom", fontsize=12)

        if save:
            constituent = res.get("constituent", "constituent")
            assignment  = res.get("assignment", "assignment")
            maps_root   = res.get("maps_root", maps_root_default)
            outdir = _map_outdir(maps_root, constituent, assignment, sub="bin_probs")
            fname  = f"{_slugify(constituent)}_{_slugify(assignment)}_bin_p{i+1}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=220, bbox_inches="tight")

        plt.show()

def plot_mik_mean(res, title_prefix="", show_points=True, point_colors=None, point_size=20,
                  save: bool = True):
    extent = res["extent"]; mean_raster = res["mean_raster"]
    cb_units = res.get("units", "µg/L")    

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig.subplots_adjust(left=0.38, right=0.88, bottom=0.10, top=0.92)

    img = ax.imshow(mean_raster, extent=extent, origin="lower",
                    cmap="YlGnBu", alpha=0.80, zorder=2)

    map_handles, ft_units = add_basemap_zoom(ax, ou_path=ou_path, rds_path=rds_path, pad_ft=300)

    point_handles = []
    if show_points:
        point_handles = _overlay_points(
            ax, res,
            point_size=20,
            units_header=f"Measured concentration ({cb_units})",
            color_mode="discrete_even",          # ← evenly spaced across the cmap
            point_cmap="RdYlBu_r",               # blue (low) → red (high)
            cmap_span=(0.06, 0.94)               # optional: clip ends to avoid extremes
        )

    all_handles = map_handles + point_handles

    # _left_gutter_panel(
    #     fig, handles=all_handles, mappable=img, cb_label=cb_units,
    #     place_cb="left", leg_left=0.12, leg_bottom=0.12, leg_w=0.26, leg_h=0.26,
    #     cb_w=0.022, gap=0.035, legend_pad=0.010, min_gap_to_map=0.012
    # )

    leg_ax, cb_ax = _left_gutter_panel(
        fig, handles=all_handles, mappable=img, cb_label=cb_units,
        place_cb="left",
        leg_left=0.12, leg_bottom=0.12, leg_w=0.26, leg_h=0.26,
        cb_w=0.022, gap=0.035, legend_pad=0.010, min_gap_to_map=0.012
    )

    # Center the legend between the colorbar and the map edge
    recenter_legend_in_left_gutter(fig, leg_ax, cb_ax,
                                margin_cb=0.010,   # gap from colorbar
                                margin_map=0.012,  # gap from map
                                vcenter=True)      # vertical center

    _add_scale_bar(ax, ft_units=ft_units, pad_frac=0.05, height_frac=0.012)
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    fig.text(0.5, 0.04, f"{title_prefix} MIK Mean Concentration", ha="center", va="bottom", fontsize=12)
    if save:
        constituent = res.get("constituent", "constituent")
        assignment  = res.get("assignment", "assignment")
        maps_root   = res.get("maps_root", maps_root_default)
        outdir = _map_outdir(maps_root, constituent, assignment, sub="mean")
        fname  = f"{_slugify(constituent)}_{_slugify(assignment)}_MIK_mean.png"
        fig.savefig(os.path.join(outdir, fname), dpi=220, bbox_inches="tight")
        
    plt.show()
    
def _nice_bins(arr, n_classes=9):
    """Return 'nice' bin edges covering the finite data range."""
    finite = np.asarray(arr, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [0, 1]  # fallback
    vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    # Use 'nice' tick-based edges (…1,2,2.5,5,10…)
    locator = MaxNLocator(nbins=n_classes-1, steps=[1, 2, 2.5, 5, 10], min_n_ticks=2)
    edges = locator.tick_values(vmin, vmax)
    return edges.tolist()

def plot_mik_variance(
    res,
    title_prefix="",
    bins=None,
    n_classes=9,
    cmap_name="coolwarm",
    show_points=True,
    point_size=20,
    point_cmap="RdYlBu_r",
    cmap_span=(0.06, 0.94),
    save: bool = True
):
    extent = res["extent"]
    units = res.get("units", "µg/L")
    var    = res["var_raster"]

    # --- classify CV into discrete classes (nice) ---
    cv_bins = np.array(bins if bins is not None else _nice_bins(var, n_classes), dtype=float)
    cv_bins = np.unique(cv_bins)
    if cv_bins.size < 2:
        cv_bins = np.array([0.0, 1.0])

    cmap = plt.get_cmap(cmap_name, cv_bins.size - 1)
    norm = mcolors.BoundaryNorm(boundaries=cv_bins, ncolors=cmap.N)

    # --- figure & map panel ---
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig.subplots_adjust(left=0.38, right=0.88, bottom=0.10, top=0.92)

    # raster first (on top; alpha lets basemap show through)
    img = ax.imshow(var, extent=extent, origin="lower",
                    cmap=cmap, norm=norm, alpha=0.80, zorder=2)

    # basemap + zoom (roads + OU)
    map_handles, ft_units = add_basemap_zoom(ax, ou_path=ou_path, rds_path=rds_path, pad_ft=300)

    # optional measured points (same look as your mean map)
    point_handles = []
    if show_points:
        point_handles = _overlay_points(
            ax, res,
            point_size=point_size,
            units_header=f"Measured concentration ({units})",
            color_mode="discrete_even",
            point_cmap=point_cmap,
            cmap_span=cmap_span
        )

    all_handles = map_handles + point_handles

    # left gutter: legend + colorbar (no overlap)
    leg_ax, cb_ax = _left_gutter_panel(
        fig, handles=all_handles, mappable=img,
        cb_label=f"MIK Conditional Variance ({units})²",
        place_cb="left",
        leg_left=0.12, leg_bottom=0.12, leg_w=0.26, leg_h=0.26,
        cb_w=0.022, gap=0.035, legend_pad=0.010, min_gap_to_map=0.012
    )
    recenter_legend_in_left_gutter(
        fig, leg_ax, cb_ax, margin_cb=0.010, margin_map=0.012, vcenter=True
    )
    # (Optional) if you want colorbar to show bin edges only:
    # cb_ax.yaxis.set_ticks(cv_bins[1:-1] if cv_bins.size > 2 else [])

    # scalebar & clean frame
    _add_scale_bar(ax, ft_units=ft_units, pad_frac=0.05, height_frac=0.012)
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]

    # bottom title
    fig.text(0.5, 0.04, f"{title_prefix} MIK Conditional Variance",
             ha="center", va="bottom", fontsize=12)

    if save:
        constituent = res.get("constituent", "constituent")
        assignment  = res.get("assignment", "assignment")
        maps_root   = res.get("maps_root", maps_root_default)
        outdir = _map_outdir(maps_root, constituent, assignment, sub="variance")
        fname  = f"{_slugify(constituent)}_{_slugify(assignment)}_MIK_variance.png"
        fig.savefig(os.path.join(outdir, fname), dpi=220, bbox_inches="tight")

    plt.show()

def _cmap_with_alpha(cmap, alpha=1.0):
    """Return a copy of `cmap` with a fixed RGBA alpha applied."""
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cols = cmap(np.linspace(0, 1, cmap.N))
    cols[:, 3] = alpha
    return ListedColormap(cols, name=f"{cmap.name}_a{int(alpha*100)}")

def score_rcl_nodes(res, rod_cleanup_level=3.4):
    """
    Score RCL nodes using percentile-based bins of s_N computed over the whole grid.
    s_N = sqrt(var_raster) / RCL. Percentile cuts: 30|50|70|90 -> scores 0..4.
    """
    xgrid = res["xgrid"]; ygrid = res["ygrid"]; var = np.asarray(res["var_raster"], float)

    # --- compute s_N over the raster & its percentile cut points (ignore NaNs) ---
    sn_field = np.sqrt(np.maximum(var, 0.0)) / float(rod_cleanup_level)
    sn_finite = sn_field[np.isfinite(sn_field)]
    if sn_finite.size == 0:
        # Degenerate case: nothing finite; fall back to zeros
        q30 = q50 = q70 = q90 = np.inf
    else:
        q30, q50, q70, q90 = np.nanpercentile(sn_finite, [30, 50, 70, 90])

    # --- load RCL nodes + grid 274 and sample s_N at node positions ---
    rcl_df  = pd.read_csv(rcl_path)
    grid_gdf = gpd.read_file(grid_path)
    rcl_df["row"] = rcl_df["row"].astype(int); rcl_df["col"] = rcl_df["col"].astype(int)
    grid_gdf["row"] = grid_gdf["row"].astype(int); grid_gdf["column"] = grid_gdf["column"].astype(int)

    rcl_gdf = pd.merge(rcl_df, grid_gdf, left_on=["row","col"], right_on=["row","column"])
    rcl_gdf = gpd.GeoDataFrame(rcl_gdf, geometry="geometry", crs=grid_gdf.crs)

    # Sample at centroids
    rcl_gdf["x"] = rcl_gdf.geometry.centroid.x
    rcl_gdf["y"] = rcl_gdf.geometry.centroid.y

    # map to raster indices
    grid_res = xgrid[1] - xgrid[0]
    x_idx = ((rcl_gdf["x"] - xgrid[0]) / grid_res).round().astype(int)
    y_idx = ((rcl_gdf["y"] - ygrid[0]) / grid_res).round().astype(int)
    x_idx = x_idx.clip(0, sn_field.shape[1] - 1)
    y_idx = y_idx.clip(0, sn_field.shape[0] - 1)

    mik_cv_vals = var[y_idx, x_idx]
    sn_vals     = sn_field[y_idx, x_idx]

    # percentile-based scoring
    def _score_from_percentiles(v):
        if not np.isfinite(v):
            return np.nan
        if v < q30: return 0
        if v < q50: return 1
        if v < q70: return 2
        if v < q90: return 3
        return 4

    rcl_gdf["MIK_CV"] = mik_cv_vals
    rcl_gdf["S_N"]    = sn_vals
    rcl_gdf["S_CV"]   = rcl_gdf["S_N"].apply(_score_from_percentiles)

    # (optional) expose the cut points for reference/debugging
    rcl_gdf["SN_Q30"] = q30
    rcl_gdf["SN_Q50"] = q50
    rcl_gdf["SN_Q70"] = q70
    rcl_gdf["SN_Q90"] = q90
    return rcl_gdf

def plot_rcl_scores(
    rcl_gdf,
    aoi=None,
    title="Scored RCL Nodes by MIK-CV",
    ou_path=ou_path,
    rds_path=rds_path,
    pad_ft=300,
    draw_mode="squares",
    rcl_size=18,
    rcl_alpha=0.70,
    cmap_name="RdYlBu_r",
    res_for_points=None,
    point_size=20,
    save: bool = True,
    maps_root: str = maps_root_default,
    constituent: str | None = None,
    assignment: str | None = None
):
    if aoi is None:
        aoi = gpd.read_file(ou_path)

    # ── CRS align ─────────────────────────────────────────────────────────────
    if rcl_gdf.crs is None:
        rcl_gdf = rcl_gdf.set_crs(aoi.crs, allow_override=True)
    elif rcl_gdf.crs != aoi.crs:
        rcl_gdf = rcl_gdf.to_crs(aoi.crs)

    rcl_plot = rcl_gdf.dropna(subset=["S_CV"]).copy()
    # Use provided x,y if present; otherwise centroid
    if {"x","y"}.issubset(rcl_plot.columns):
        x = pd.to_numeric(rcl_plot["x"], errors="coerce")
        y = pd.to_numeric(rcl_plot["y"], errors="coerce")
    else:
        geoms = rcl_plot.geometry
        if not geoms.geom_type.isin(["Point"]).all():
            geoms = geoms.centroid
        x = geoms.x; y = geoms.y
    m = np.isfinite(x) & np.isfinite(y)
    rcl_plot = rcl_plot.loc[m].copy()
    x = x.loc[m]; y = y.loc[m]
    rcl_plot["S_CV"] = rcl_plot["S_CV"].astype(int)

    # ── Discrete palette 0..4 ────────────────────────────────────────────────
    levels = np.arange(-0.5, 5.5, 1.0)
    base_cmap = plt.get_cmap(cmap_name, len(levels) - 1)
    score_cmap = _cmap_with_alpha(base_cmap, alpha=rcl_alpha)
    score_norm = mcolors.BoundaryNorm(levels, ncolors=score_cmap.N)
    mappable = cm.ScalarMappable(norm=score_norm, cmap=score_cmap); mappable.set_array([])

    # ── Figure & layout ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig.subplots_adjust(left=0.38, right=0.88, bottom=0.10, top=0.92)
    ax.set_rasterization_zorder(4.9)

    # ── RCL layer ────────────────────────────────────────────────────────────
    if draw_mode == "pcolormesh":
        xu = np.unique(np.sort(np.asarray(x))); yu = np.unique(np.sort(np.asarray(y)))
        dx = np.median(np.diff(xu)) if xu.size > 1 else 1.0
        dy = np.median(np.diff(yu)) if yu.size > 1 else 1.0
        xedges = np.r_[xu - dx/2, xu[-1] + dx/2]
        yedges = np.r_[yu - dy/2, yu[-1] + dy/2]
        Z = np.full((yu.size, xu.size), np.nan)
        xi = np.searchsorted(xu, np.asarray(x))
        yi = np.searchsorted(yu, np.asarray(y))
        Z[yi, xi] = rcl_plot["S_CV"].to_numpy()
        ax.pcolormesh(xedges, yedges, Z, cmap=score_cmap, norm=score_norm,
                      shading="flat", zorder=4.8, rasterized=True)
    else:
        ax.scatter(
            x, y,
            c=rcl_plot["S_CV"],
            cmap=score_cmap, norm=score_norm,
            marker="s", s=rcl_size,
            edgecolors="none", linewidths=0,
            antialiased=False, rasterized=True, zorder=4.8
        )

    # Basemap on top
    map_handles, ft_units = add_basemap_zoom(ax, ou_path=ou_path, rds_path=rds_path, pad_ft=pad_ft)

    # Optional measured-point overlay
    all_handles = map_handles[:]
    pt_handles = []
    if res_for_points is not None:
        units = res_for_points.get("units", "µg/L")
        pt_handles = _overlay_points(
            ax, res_for_points, point_size=point_size, z=6,
            units_header=f"Measured concentration ({units})",
            color_mode="discrete_even", point_cmap="RdYlBu_r", cmap_span=(0.06, 0.94)
        )
        all_handles += pt_handles

    # Proxy legend entry for RCL nodes
    rcl_proxy = Line2D([0],[0], marker="s", color="k", lw=0,
                       markerfacecolor="white", markeredgewidth=0.7,
                       markersize=7, label="RCL node (color = S_CV)")
    all_handles = map_handles + [rcl_proxy] + pt_handles

    # Left gutter panel
    leg_ax, cb_ax = _left_gutter_panel(
        fig, handles=all_handles, mappable=mappable,
        cb_label="MIK-CV Score", place_cb="left",
        leg_left=0.12, leg_bottom=0.12, leg_w=0.26, leg_h=0.26,
        cb_w=0.022, gap=0.035, legend_pad=0.010, min_gap_to_map=0.012
    )
    cb_ax.yaxis.set_ticks([0,1,2,3,4])

    recenter_legend_in_left_gutter(fig, leg_ax, cb_ax,
                                   margin_cb=0.010, margin_map=0.012, vcenter=True)

    _add_scale_bar(ax, ft_units=ft_units, pad_frac=0.05, height_frac=0.012)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    fig.text(0.5, 0.04, title, ha="center", va="bottom", fontsize=12)
    
    if save:
        c = constituent or (res_for_points.get("constituent") if isinstance(res_for_points, dict) else None) or "constituent"
        a = assignment  or (res_for_points.get("assignment")  if isinstance(res_for_points, dict) else None) or "assignment"
        outdir = _map_outdir(maps_root, c, a, sub="rcl_scores")
        fname  = f"{_slugify(c)}_{_slugify(a)}_RCL_SCV_{draw_mode}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=220, bbox_inches="tight")
    
    plt.show()


    
# ──────────────────────────────────────────────────────────────────────────────
# CTET UU/MU
# ──────────────────────────────────────────────────────────────────────────────

##CDF for threshold defn

thresholds_ctet_uumu = [3.4, 10, 34, 100,300]
thresholds_ctet_lucr = [3.4, 10, 34, 100, 300]

ctet_cdf = plot_cdf_for_constituent(
    constituent="CTET",
    include_characterization=False,
    logx=True,
    thresholds_by_assignment={
        "UU/MU": thresholds_ctet_uumu,
        "LU/CR": thresholds_ctet_lucr
    },
    overlay=True,   
    save=True
)
print(ctet_cdf["stats"])


ctet_uumu_params = plot_variograms_before_kriging(
    constituent="CTET",
    assignment="UU/MU",
    thresholds=thresholds_ctet_uumu,
    model="spherical",
    maxlag=3500,
    n_lags=15,
    use_nugget=True,
    include_characterization=False,  # flip to False if you want monitoring-only
    save_plots=True
)

ctet_uumu_params_updated = {
    'ind3_4': (3499.77905979792, 0.22058724944635313, 0.04812535974324434),
    'ind10': (1384.1401640073657, 0.2761858412019904, 3.2621993908099394e-15),
    'ind34': (938.0994948652194, 0.08148638776289183, 0.1144445559012011),
    'ind100': (1723.4990049606326, 0.019804405212632988, 0.019784655122452494),
    'ind300': (2063.7919015579964, 0.01, 0.001)}

plot_variograms_with_final_params(
    constituent="CTET",
    assignment="UU/MU",
    thresholds=thresholds_ctet_uumu,
    params_dict=ctet_uumu_params_updated,
    model="spherical",
    maxlag=3500,
    n_lags=15,
    include_characterization=False,
    save_plots=True
)

# 2) Paste the printed dict (or reuse ctet_uumu_params) into run_mik_for:
ctet_uumu = run_mik_for(
    constituent="CTET",
    assignment="UU/MU",
    thresholds=thresholds_ctet_uumu,
    variogram_params=ctet_uumu_params_updated,  
    grid_res=50,
    include_characterization=False,
    show_variogram=False                
)


# ──────────────────────────────────────────────────────────────────────────────
# CTET LU/CR
# ──────────────────────────────────────────────────────────────────────────────


ctet_lucr_params = plot_variograms_before_kriging(
    constituent="CTET",
    assignment="LU/CR",
    thresholds=thresholds_ctet_lucr,
    model="spherical",
    maxlag=3500,
    n_lags=15,
    use_nugget=True,
    include_characterization=False,  # flip to False if you want monitoring-only
    save_plots=True
)

ctet_lucr_params_updated = {
    'ind3_4': (3453.8016791704335, 0.28974403969600593, 4.042172460893039e-11),
    'ind10': (3453.8016791704335, 0.37358655638152855, 5.852516212190578e-11),
    'ind34': (2191.069418118981, 0.3, 0.05),
    'ind100': (1500, 0.2, 0.05),
    'ind300': (1500, 0.2, 0.05)}

plot_variograms_with_final_params(
    constituent="CTET",
    assignment="LU/CR",
    thresholds=thresholds_ctet_lucr,
    params_dict=ctet_lucr_params_updated,
    model="spherical",
    maxlag=3500,
    n_lags=15,
    include_characterization=False,
    save_plots=True
)


# 2) Paste the printed dict (or reuse ctet_uumu_params) into run_mik_for:
ctet_lucr = run_mik_for(
    constituent="CTET",
    assignment="LU/CR",
    thresholds=thresholds_ctet_lucr,
    variogram_params=ctet_lucr_params_updated,  
    grid_res=50,
    include_characterization=False,
    show_variogram=False                
)


# ──────────────────────────────────────────────────────────────────────────────
# HexCr UU/MU
# ──────────────────────────────────────────────────────────────────────────────

##CDF for threshold defn

thresholds_hexcr_uumu = [48, 75, 100, 150]
thresholds_hexcr_lucr = [48, 75, 100]

hexcr_cdf = plot_cdf_for_constituent(
    constituent="HexCr",
    include_characterization=False,
    logx=True,
    thresholds_by_assignment={
        "UU/MU": thresholds_hexcr_uumu,
        "LU/CR": thresholds_hexcr_lucr
    },
    overlay=True,   
    save=True
)
print(hexcr_cdf["stats"])


hexcr_uumu_params = plot_variograms_before_kriging(
    constituent="HexCr",
    assignment="UU/MU",
    thresholds=thresholds_hexcr_uumu,
    model="spherical",
    maxlag=3500,
    n_lags=15,
    use_nugget=True,
    include_characterization=False,  # flip to False if you want monitoring-only
    save_plots=True
)

hexcr_uumu_params_updated = {
    'ind48': (2500, 0.07, 0.02),
    'ind75': (2500, 0.05, 0.02),
    'ind100': (1500, 0.06, 0.01),
    'ind150': (2500, 0.02, 0.001)}

plot_variograms_with_final_params(
    constituent="HexCr",
    assignment="UU/MU",
    thresholds=thresholds_hexcr_uumu,
    params_dict=hexcr_uumu_params_updated,
    model="spherical",
    maxlag=3500,
    n_lags=15,
    include_characterization=False,
    save_plots=True
)

# 2) Paste the printed dict (or reuse ctet_uumu_params) into run_mik_for:
hexcr_uumu = run_mik_for(
    constituent="HexCr",
    assignment="UU/MU",
    thresholds=thresholds_hexcr_uumu,
    variogram_params=hexcr_uumu_params_updated,  
    grid_res=50,
    include_characterization=False,
    show_variogram=False                
)

# ──────────────────────────────────────────────────────────────────────────────
# HexCr LU/CR
# ──────────────────────────────────────────────────────────────────────────────


hexcr_lucr_params = plot_variograms_before_kriging(
    constituent="HexCr",
    assignment="LU/CR",
    thresholds=thresholds_hexcr_lucr,
    model="spherical",
    maxlag=3500,
    n_lags=15,
    use_nugget=True,
    include_characterization=False,  # flip to False if you want monitoring-only
    save_plots=True
)

hexcr_lucr_params_updated = {
    'ind48': (1000, 0.25, 0.01),
    'ind75': (1000, 0.25, 0.01),
    'ind100': (1000, 0.25, 0.01)}


plot_variograms_with_final_params(
    constituent="HexCr",
    assignment="LU/CR",
    thresholds=thresholds_hexcr_lucr,
    params_dict=hexcr_lucr_params_updated,
    model="spherical",
    maxlag=3500,
    n_lags=15,
    include_characterization=False,
    save_plots=True
)



# 2) Paste the printed dict (or reuse ctet_uumu_params) into run_mik_for:
hexcr_lucr = run_mik_for(
    constituent="HexCr",
    assignment="LU/CR",
    thresholds=thresholds_hexcr_lucr,
    variogram_params=hexcr_lucr_params_updated,  
    grid_res=50,
    include_characterization=False,
    show_variogram=False                
)

# ──────────────────────────────────────────────────────────────────────────────
# Tc99 UU/MU
# ──────────────────────────────────────────────────────────────────────────────

##CDF for threshold defn

thresholds_tc99_uumu = [9, 51, 90, 225, 450, 900, 5000]
thresholds_tc99_lucr = [51, 90, 225]

tc99_cdf = plot_cdf_for_constituent(
    constituent="Tc99",
    include_characterization=False,
    logx=True,
    thresholds_by_assignment={
        "UU/MU": thresholds_tc99_uumu,
        "LU/CR": thresholds_tc99_lucr
    },
    overlay=True,   
    save=True
)
print(tc99_cdf["stats"])


tc99_uumu_params = plot_variograms_before_kriging(
    constituent="Tc99",
    assignment="UU/MU",
    thresholds=thresholds_tc99_uumu,
    model="spherical",
    maxlag=3500,
    n_lags=15,
    use_nugget=True,
    include_characterization=False,  
    save_plots=True
)

tc99_uumu_params_updated = {
    'ind9': (3499.5068596874944, 0.02042347300437718, 0.0030915772023635563),
    'ind51': (3234.6091653952876, 0.1524903209644482, 0.10018158252517226),
    'ind90': (2903.311470556268, 0.10878833087822419, 0.17237759579392553),
    'ind225': (1000, 0.12303153468083267, 0.08566764129112239),
    'ind450': (1000, 0.18, 0.05),
    'ind900': (1000, 0.11, 0.05),
    'ind5000': (1500, 0.05, 0.01)}

plot_variograms_with_final_params(
    constituent="Tc99",
    assignment="UU/MU",
    thresholds=thresholds_tc99_uumu,
    params_dict=tc99_uumu_params_updated,
    model="spherical",
    maxlag=3500,
    n_lags=15,
    include_characterization=False,
    save_plots=True
)


# 2) Paste the printed dict (or reuse ctet_uumu_params) into run_mik_for:
tc99_uumu = run_mik_for(
    constituent="Tc99",
    assignment="UU/MU",
    thresholds=thresholds_tc99_uumu,
    variogram_params=tc99_uumu_params_updated,  
    grid_res=50,
    include_characterization=False,
    show_variogram=False                
)

# ──────────────────────────────────────────────────────────────────────────────
# Tc99 LU/CR
# ──────────────────────────────────────────────────────────────────────────────


tc99_lucr_params = plot_variograms_before_kriging(
    constituent="Tc99",
    assignment="LU/CR",
    thresholds=thresholds_tc99_lucr,
    model="spherical",
    maxlag=3500,
    n_lags=15,
    use_nugget=True,
    include_characterization=False,  
    save_plots=True
)

tc99_lucr_params_updated = {
    'ind90': (2338.1352785423956, 0.18664110744741697, 0.10756277751949161),
    'ind225': (3453.8016791704335, 0.1, 0.1),
    'ind450': (3453.8016791704335, 0.1, 0.01)}


# 2) Paste the printed dict (or reuse ctet_uumu_params) into run_mik_for:
tc99_lucr = run_mik_for(
    constituent="Tc99",
    assignment="LU/CR",
    thresholds=thresholds_tc99_lucr,
    variogram_params=tc99_lucr_params_updated,  
    grid_res=50,
    include_characterization=False,
    show_variogram=False                
)



# ──────────────────────────────────────────────────────────────────────────────
# Plot bin probabilities and MIK mean / variance 
# ──────────────────────────────────────────────────────────────────────────────


plot_bin_probs(ctet_uumu, title_prefix="CTET UU/MU —")
plot_mik_mean(ctet_uumu, title_prefix="CTET UU/MU —")
plot_mik_variance(ctet_uumu, title_prefix="CTET UU/MU —")

plot_bin_probs(ctet_lucr, title_prefix="CTET LU/CR —")
plot_mik_mean(ctet_lucr, title_prefix="CTET LU/CR —")
plot_mik_variance(ctet_lucr, title_prefix="CTET LU/CR —")

plot_bin_probs(hexcr_uumu, title_prefix="HexCr UU/MU —")
plot_mik_mean(hexcr_uumu, title_prefix="HexCr UU/MU —")
plot_mik_variance(hexcr_uumu, title_prefix="HexCr UU/MU —")

plot_bin_probs(hexcr_lucr, title_prefix="HexCr LU/CR —")
plot_mik_mean(hexcr_lucr, title_prefix="HexCr LU/CR —")
plot_mik_variance(hexcr_lucr, title_prefix="HexCr LU/CR —")

plot_bin_probs(tc99_uumu, title_prefix="Tc99 UU/MU —")
plot_mik_mean(tc99_uumu, title_prefix="Tc99 UU/MU —")
plot_mik_variance(tc99_uumu, title_prefix="Tc99 UU/MU —")

plot_bin_probs(tc99_lucr, title_prefix="Tc99 LU/CR —")
plot_mik_mean(tc99_lucr, title_prefix="Tc99 LU/CR —")
plot_mik_variance(tc99_lucr, title_prefix="Tc99 LU/CR —")


# ──────────────────────────────────────────────────────────────────────────────
# RCL scoring & plots 
# ──────────────────────────────────────────────────────────────────────────────

# CTET — UU/MU
rcl_ctet_uumu = score_rcl_nodes(ctet_uumu, rod_cleanup_level=3.4)
plot_rcl_scores(
    rcl_ctet_uumu, ctet_uumu["aoi"],
    title="CTET UU/MU — Scored RCL Nodes by MIK-CV",
    res_for_points=ctet_uumu,
    draw_mode="pcolormesh"
)

# CTET — LU/CR
rcl_ctet_lucr = score_rcl_nodes(ctet_lucr, rod_cleanup_level=3.4)
plot_rcl_scores(
    rcl_ctet_lucr, ctet_lucr["aoi"],
    title="CTET LU/CR — Scored RCL Nodes by MIK-CV",
    res_for_points=ctet_lucr,
    draw_mode="pcolormesh"
)

# HexCr — UU/MU
rcl_hexcr_uumu = score_rcl_nodes(hexcr_uumu, rod_cleanup_level=48)
plot_rcl_scores(
    rcl_hexcr_uumu, hexcr_uumu["aoi"],
    title="HexCr UU/MU — Scored RCL Nodes by MIK-CV (S_CV)",
    res_for_points=hexcr_uumu,
    draw_mode="pcolormesh"
)

# HexCr — LU/CR
rcl_hexcr_lucr = score_rcl_nodes(hexcr_lucr, rod_cleanup_level=48)
plot_rcl_scores(
    rcl_hexcr_lucr, hexcr_lucr["aoi"],
    title="HexCr LU/CR — Scored RCL Nodes by MIK-CV (S_CV)",
    res_for_points=hexcr_lucr,
    draw_mode="pcolormesh"
)

# Tc99 — UU/MU
rcl_tc99_uumu = score_rcl_nodes(tc99_uumu, rod_cleanup_level=9)
plot_rcl_scores(
    rcl_tc99_uumu, tc99_uumu["aoi"],
    title="Tc99 UU/MU — Scored RCL Nodes by MIK-CV (S_CV)",
    res_for_points=tc99_uumu,
    draw_mode="pcolormesh"
)

# Tc99 — LU/CR
rcl_tc99_lucr = score_rcl_nodes(tc99_lucr, rod_cleanup_level=9)
plot_rcl_scores(
    rcl_tc99_lucr, tc99_lucr["aoi"],
    title="Tc99 LU/CR — Scored RCL Nodes by MIK-CV (S_CV)",
    res_for_points=tc99_lucr,
    draw_mode="pcolormesh"
)


# ──────────────────────────────────────────────────────────────────────────────
# MIK-COV scoring & plots 
# ──────────────────────────────────────────────────────────────────────────────


plot_mik_cov(ctet_uumu, title_prefix="CTET UU/MU —")
plot_mik_cov_score(ctet_uumu, title_prefix="CTET UU/MU —")

plot_mik_cov(ctet_lucr, title_prefix="CTET LU/CR —")
plot_mik_cov_score(ctet_lucr, title_prefix="CTET LU/CR —")

plot_mik_cov(hexcr_uumu, title_prefix="HexCr UU/MU —")
plot_mik_cov_score(hexcr_uumu, title_prefix="HexCr UU/MU —")

plot_mik_cov(hexcr_lucr, title_prefix="HexCr LU/CR —")
plot_mik_cov_score(hexcr_lucr, title_prefix="HexCr LU/CR —")

plot_mik_cov(tc99_uumu, title_prefix="Tc99 UU/MU —")
plot_mik_cov_score(tc99_uumu, title_prefix="Tc99 UU/MU —")

plot_mik_cov(tc99_lucr, title_prefix="Tc99 LU/CR —")
plot_mik_cov_score(tc99_lucr, title_prefix="Tc99 LU/CR —")
