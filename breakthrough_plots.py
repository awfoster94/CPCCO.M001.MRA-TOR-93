#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Three-COC breakthrough plots for a single model cell
---------------------------------------------------
• CTET, Cr6, Tc-99 — one subplot each
• Fixed layer colours: 2=red, 3=green, 4=yellow, 5=blue, 6=orange, 7=purple
• Legend drawn inside Tc-99 panel, always shows layers 2-7 + cleanup
• Accept row/col (1-based) or projected x/y; finds nearest cell centre
"""

import os, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import flopy
import geopandas as gpd

# =============================================================================
# CONFIG ----------------------------------------------------------------------
# =============================================================================
MODEL_START_DATE = "2015-01-01"

FLOW_ROOT   = os.path.join("source_files", "flow", "source")
TRANS_ROOT  = os.path.join("source_files", "transport")

cleanup_csv = os.path.join("data", "conc_ranking", "final_cleanup_levels.csv")

basemap     = os.path.join("gis", "shp")
grid_path   = os.path.join(basemap, "model_grid", "grid_274.shp")
head_file = os.path.join(FLOW_ROOT, "P2Rv8.3.hds")

X_START = pd.Timestamp("2015-01-01")
X_END   = pd.Timestamp("2037-12-31")

COC_SPECS = {
    "CTET": dict(folder="CTET_600y", prefix="P2Rv8.3_CTET_600y",
                 cleanup_pattern="ctet", label="CTET", units="µg/L"),
    "Cr6" : dict(folder="HexChromium", prefix="HexChromium",
                 cleanup_pattern="hexavalent", label="Cr6", units="µg/L"),
    "Tc99": dict(folder="TC99", prefix="TC99",
                 cleanup_pattern="tc-99", label="Tc99", units="pCi/L"),
}

LAYER_COLOR = {2:"red", 3:"green", 4:"yellow", 5:"blue", 6:"orange", 7:"purple"}
PLOT_LAYERS = [2,3,4,5,6,7]

CLEANUP_COLOR = "k"
CLEANUP_LS    = "--"
CLEANUP_LW    = 1.0
CLEANUP_LABEL = "ROD Cleanup level"

# =============================================================================
# LOAD FLOW MODEL & GEOREFERENCE ----------------------------------------------
# =============================================================================
mf = flopy.modflow.Modflow.load("P2Rv8.3_start2015_sp2023.nam",
                                model_ws=FLOW_ROOT, version="mf2k",
                                forgive=True, check=False)

grid_gdf = gpd.read_file(grid_path)
llx, lly, _, _ = grid_gdf.total_bounds
mg = mf.modelgrid
mg.set_coord_info(xoff=llx, yoff=lly, angrot=0.0, epsg=grid_gdf.crs.to_epsg())

# =============================================================================
# HELPERS ---------------------------------------------------------------------
# =============================================================================
def _find_ucn(ws, prefix):
    cand = os.path.join(ws, f"{prefix}.UCN")
    if os.path.isfile(cand):
        return cand
    lst = glob.glob(os.path.join(ws, "*.UCN"))
    if not lst:
        raise FileNotFoundError(f"No UCN in {ws}")
    print(f"[warn] {prefix}.UCN not found; using {os.path.basename(lst[0])}")
    return lst[0]

def _cleanup_val(df, pat):
    m = df["COC"].str.lower().str.contains(pat.lower(), na=False)
    if not m.any():
        raise ValueError(f"Cleanup pattern '{pat}' not found")
    return float(df.loc[m, "Final Cleanup Level"].iloc[0])

def _nearest_rc(x, y, one_based=True):
    d2 = (mg.xcellcenters - x)**2 + (mg.ycellcenters - y)**2
    i0, j0 = np.unravel_index(np.argmin(d2), d2.shape)
    return (i0+1, j0+1) if one_based else (i0, j0)

def _ts_all_layers(ucn, row1b, col1b):
    i, j = row1b-1, col1b-1
    t   = np.asarray(ucn.get_times(), float)
    dt0 = pd.Timestamp(MODEL_START_DATE)
    dates = dt0 + pd.to_timedelta(t, unit="D")
    arr0 = ucn.get_data(totim=t[0]); nlay = arr0.shape[0]
    data = np.empty((t.size, nlay), float)
    for k, tt in enumerate(t):
        vals = ucn.get_data(totim=tt)[:, i, j]
        data[k] = np.where(vals>=1e29, np.nan, vals)
    cols = [f"L{n+1}" for n in range(nlay)]
    df = pd.DataFrame(data, index=dates, columns=cols)
    df.insert(0, "totim_days", t)
    df.index.name="date"
    return df

def _ts_heads(hds, row1b, col1b):
    i, j = row1b-1, col1b-1
    t   = np.asarray(hds.get_times(), float)
    dt0 = pd.Timestamp(MODEL_START_DATE)
    dates = dt0 + pd.to_timedelta(t, unit="D")
    arr0 = hds.get_data(totim=t[0]); nlay = arr0.shape[0]
    botm = mf.dis.botm.array
    data = np.empty((t.size, nlay), float)
    for k, tt in enumerate(t):
        heads = hds.get_data(totim=tt)[:, i, j]
        for n in range(nlay):
            h = heads[n]
            b = botm[n, i, j] if botm.ndim == 3 else botm[n]
            data[k, n] = 0 if (h > 1e29 or np.isnan(h) or h < b) else h
    cols = [f"L{n+1}" for n in range(nlay)]
    df = pd.DataFrame(data, index=dates, columns=cols)
    return df

def _plot_one(ax, df, label, units, cleanup, xrot, head_df=None):
    for ln in PLOT_LAYERS:
        col = f"L{ln}"
        if col in df and np.isfinite(df[col]).any():
            ax.plot(df.index, df[col], marker="o", ms=2.5, lw=1.25,
                    color=LAYER_COLOR[ln], label=f"layer {ln}")

    ax.axhline(cleanup, color=CLEANUP_COLOR, ls=CLEANUP_LS, lw=CLEANUP_LW,
               label=CLEANUP_LABEL)

    if head_df is not None:
        ax2 = ax.twinx()
        col = "L3"
        if col in head_df and np.isfinite(head_df[col]).any():
            ax2.plot(head_df.index, head_df[col], marker="x", ms=2.5, lw=1.0,
                     color="black", linestyle="--", alpha=0.8, label="Layer 3 head")
        ax2.set_ylabel("Head (ft)")

    # --- aesthetic to match previous ECF --------------------------------------
    ax.set_facecolor("#f0f0f0")       # light grey panel background
    ax.set_axisbelow(True)            # draw grid underneath the lines
    ax.grid(True, which="major",
            color="white", linestyle="-", linewidth=0.8, alpha=1.0)

    ax.set_ylabel(units)
    ax.set_title(label, fontsize=11, fontweight="bold")
    # ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim(X_START, X_END)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=xrot, ha="right")

# =============================================================================
# MAIN FUNCTION ---------------------------------------------------------------
# =============================================================================
def plot_breakthrough_3coc(*, row=None, col=None, x=None, y=None,
                           title=None, xrot=45, outfile=None, show=True, plot_head=False):
    if (row is None or col is None):
        if x is None or y is None:
            raise ValueError("Provide row+col or x+y.")
        row, col = _nearest_rc(x, y)
        print(f"(x={x:.1f}, y={y:.1f}) -> row {row}, col {col}")

    clean_df = pd.read_csv(cleanup_csv)
    dfs, cln = {}, {}
    for key, cfg in COC_SPECS.items():
        ucn = flopy.utils.UcnFile(_find_ucn(os.path.join(TRANS_ROOT, cfg["folder"]),
                                            cfg["prefix"]), precision="double")
        dfs[key] = _ts_all_layers(ucn, row, col)
        dfs[key] = _ts_all_layers(ucn, row, col).apply(lambda s: s.clip(lower=0)) #negative concs will plot as zero
        cln[key] = _cleanup_val(clean_df, cfg["cleanup_pattern"])

    head_df = None
    if plot_head:
        hds = flopy.utils.HeadFile(head_file)
        head_df = _ts_heads(hds, row, col)

    fig, axes = plt.subplots(1, 3, figsize=(12,4), sharex=True)
    for ax, key in zip(axes, ["CTET", "Cr6", "Tc99"]):
        _plot_one(ax, dfs[key], COC_SPECS[key]["label"], COC_SPECS[key]["units"],
                  cln[key], xrot, head_df)

    for ax in axes:
        # ax.set_ylim(bottom=0)                 # y-axis starts at 0
        ax.tick_params(axis='x', labelsize=7) # smaller year labels

    handles = [mlines.Line2D([], [], color=LAYER_COLOR[ln], marker="o",
                             ms=2, lw=1, label=f"layer {ln}")
               for ln in PLOT_LAYERS]
    handles.append(mlines.Line2D([], [], color=CLEANUP_COLOR, ls=CLEANUP_LS,
                                 lw=CLEANUP_LW, label=CLEANUP_LABEL))
    if plot_head:
        handles.append(mlines.Line2D([], [], color="black", linestyle="--",
                                    marker="x", lw=1, label="Layer 3 head"))
    axes[-1].legend(handles=handles, loc="upper right",
                    fontsize=6, frameon=True, framealpha=0.9)

    if title is None:
        title = f"Simulated concentrations at r{row} c{col}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98, c="navy")
    fig.tight_layout(rect=(0,0.02,1,0.95))

    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print("Saved", outfile)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dfs, cln, row, col

def export_breakthrough_every5(*,
                               outdir="BTC_PW_every5",
                               step=5,
                               row_start=39,
                               col_start=32,
                               start_pw_num=1,
                               show=False,
                               out_csv=True):
    """
    Iterate the full structured grid, taking every `step`-th row & column.
    For each sampled cell, export a 3-panel BTC figure and (optionally)
    write an output CSV mapping PW IDs to row/col and xy coords.

    Parameters
    ----------
    outdir : str
        Output folder for PNGs and optional output CSV.
    step : int
        Subsampling stride (every Nth row and column).
    row_start, col_start : int
        1-based starting row/col (use 1 for classic "1, 6, 11, ...").
    start_pw_num : int
        PW index to start numbering at (PW-0001, PW-0006, ...).
    show : bool
        Whether to show each figure interactively (False  for batch).
    out_csv : bool
        If True, writes `output.csv` with corresponding metadata for each PW.
    """
    os.makedirs(outdir, exist_ok=True)

    # structured grid sizes
    nrow = int(mf.dis.nrow)
    ncol = int(mf.dis.ncol)

    # cache cell-center arrays for XY lookup
    xc = mf.modelgrid.xcellcenters
    yc = mf.modelgrid.ycellcenters

    # --- read subset CSV and build (r,c) list ---------------------------------
    idx = pd.read_csv(os.path.join("data", "data_gap_RCs.csv"))
    idx["row"] = idx["row"].astype(int)
    idx["col"] = idx["col"].astype(int)

    # now take every `step`-th LINE from the CSV starting at that line
    subset = idx.iloc[0::step].copy()

    # build pairs in CSV order (no modulo, no re-sorting)
    pairs = list(zip(subset["row"].astype(int).to_list(), subset["col"].astype(int).to_list()))

    # --- export loop -----------------------------------------------------------
    csv = []
    pw = int(start_pw_num)

    # for r in range(row_start, nrow + 1, step):
    #     for c in range(col_start, ncol + 1, step):
    for (r, c) in pairs:
        pw_id = f"PW-{pw:04d}"
        print(pw_id, (r, c))
        title = f"Simulated concentrations at {pw_id}"
        outfile = os.path.join(outdir, f"{pw_id}_r{r:04d}_c{c:04d}.png")
        try:
            dfs, cln, rr, cc = plot_breakthrough_3coc(
                row=r, col=c,
                title=title,
                xrot=45,
                outfile=outfile,
                show=show,
                plot_head=False
            )
            # collect manifest row
            x = float(xc[r-1, c-1])
            y = float(yc[r-1, c-1])
            csv.append(
                dict(pw_id=pw_id, row=r, col=c, x=x, y=y, outfile=os.path.relpath(outfile))
            )
            print(f"[ok] {pw_id} -> r{r} c{c}  -> {os.path.basename(outfile)}")
        except Exception as e:
            print(f"[skip] {pw_id} r{r} c{c}: {e}")

        pw += step

    if out_csv and csv:
        mdf = pd.DataFrame(csv)
        mdf.to_csv(os.path.join(outdir, "PW_output.csv"), index=False)
        print(f"Saved output CSV with {len(mdf)} entries -> {os.path.join(outdir,'PW_output.csv')}")

    print("Done.")

# =============================================================================
# EXAMPLE ---------------------------------------------------------------------
# =============================================================================
if __name__ == "__main__":

    # EASTING  = 568300   # update
    # NORTHING = 138240   # update
    # dfs, cln, r1b, c1b = plot_breakthrough_3coc(
    #     x=EASTING, y=NORTHING,
    #     title="Simulated concentrations at XY point",
    #     xrot=45,
    #     outfile="BTC_xy_point_2.png",
    #     show=True,
    # )

    export_breakthrough_every5(
        outdir=os.path.join("data", "breakthroughcurves"),
        step=5,
        row_start=1,          # 1,6,11,... rows
        col_start=1,          # 1,6,11,... cols
        start_pw_num=1,       # PW-0001 starts at (1,1)
        show=False,
        out_csv=True,
    )
