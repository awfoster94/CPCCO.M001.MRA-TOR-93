import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

data_path = os.path.join("..", "02_data", "HEIS_Data_Pull")
ctet = os.path.join(data_path, "ft6_dataset4kriging_add_ij_hsu_CTET2022-2024.csv")
hexcr = os.path.join(data_path, "ft6_dataset4kriging_add_ij_hsu_HexCr2022-2024.csv")
tc99 = os.path.join(data_path, "ft6_dataset4kriging_add_ij_hsu_Tc992022-2024.csv")
aq = os.path.join(data_path, "2025RedundancyGapAnalysis_WellHSUList.csv")
gis_path = os.path.join("..", "02_data", "GIS")
basemap_gdf = gpd.read_file(os.path.join(gis_path, "shp", "OUs", "ZP1_UP1.shp")) 


ctet_df = pd.read_csv(ctet)
hexcr_df = pd.read_csv(hexcr)
tc99_df = pd.read_csv(tc99)
aq_df = pd.read_csv(aq)

# -----------------------
# Helpers
# -----------------------
def _norm_str(s):
    return str(s).strip().upper() if pd.notna(s) else s

def process_constituent(
    df: pd.DataFrame,
    aq_df: pd.DataFrame,
    basemap_gdf: gpd.GeoDataFrame,
    constituent: str,
    outroot: str = "figs",
    xcol: str = "XCOORDS",   # force XCOORDS
    ycol: str = "YCOORDS",   # force YCOORDS
):
    """
    Builds per-constituent outputs in outroot/<CONSTITUENT>/:
      - <constituent>_monitoring_avg_by_well.csv  (avg from VAL)
      - <constituent>_characterization.csv
      - <constituent>_extraction.csv
      - <constituent>_UUMU_monitoringAvg_plusCharacterization.png
      - <constituent>_LUCR_monitoringAvg_plusCharacterization.png
    Returns dict with dataframes and paths.
    """
    os.makedirs(outroot, exist_ok=True)
    outdir = os.path.join(outroot, str(constituent))
    os.makedirs(outdir, exist_ok=True)

    # --- Normalize TYPE (handles TYPE/Type)
    d = df.copy()
    type_col = "TYPE" if "TYPE" in d.columns else ("Type" if "Type" in d.columns else None)
    if type_col is None:
        raise KeyError("No TYPE/Type column found in input dataframe.")
    d["TYPE"] = d[type_col].astype(str).map(_norm_str)

    # --- Normalize aquifer + join key
    aq2 = aq_df.copy()
    if not {"NAME", "Final_Assignment"}.issubset(aq2.columns):
        raise KeyError("aq_df must contain 'NAME' and 'Final_Assignment' columns.")
    aq2["Final_Assignment"] = aq2["Final_Assignment"].astype(str).map(_norm_str)
    d["NAME_norm"] = d["NAME"].astype(str).map(_norm_str)
    aq2["NAME_norm"] = aq2["NAME"].astype(str).map(_norm_str)

    # --- Force coords to XCOORDS/YCOORDS and coerce numeric
    if xcol not in d.columns or ycol not in d.columns:
        raise KeyError(f"Expected coordinate columns '{xcol}' and '{ycol}' not found.")
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")

    # --- Split by TYPE
    monitoring_df = d.loc[d["TYPE"] == "MONITORING"].copy()
    characterization_df = d.loc[d["TYPE"] == "CHARACTERIZATION"].copy()
    extraction_df = d.loc[d["TYPE"] == "EXTRACTION"].copy()

    # --- Merge aquifer onto all rows (by normalized name)
    d_w_aq = d.merge(aq2[["NAME_norm", "Final_Assignment"]], on="NAME_norm", how="left")
    d_w_aq["Final_Assignment"] = d_w_aq["Final_Assignment"].astype(str).map(_norm_str)

    # --- All-points GeoDataFrame (for plotting overlays + exports)
    pts = d_w_aq.dropna(subset=[xcol, ycol]).copy()
    all_points_gdf = gpd.GeoDataFrame(
        pts, geometry=gpd.points_from_xy(pts[xcol], pts[ycol]), crs=basemap_gdf.crs
    )

    # --- Compute monitoring averages (from VAL) if any monitoring rows exist
    if len(monitoring_df) > 0:
        monitoring_df[xcol] = pd.to_numeric(monitoring_df[xcol], errors="coerce")
        monitoring_df[ycol] = pd.to_numeric(monitoring_df[ycol], errors="coerce")

        val_num = pd.to_numeric(monitoring_df["VAL"], errors="coerce")
        monitoring_df = monitoring_df.assign(VAL_num=val_num)

        agg = (
            monitoring_df.groupby("NAME_norm", dropna=False)
            .agg(
                avg_conc=("VAL_num", "mean"),   # average from VAL
                **{
                    xcol: (xcol, "first"),
                    ycol: (ycol, "first"),
                },
            )
            .reset_index()
        )
        # Map back original NAME + aquifer
        name_map = monitoring_df.groupby("NAME_norm")["NAME"].first().reset_index()
        agg = agg.merge(name_map, on="NAME_norm", how="left")
        agg = agg.merge(aq2[["NAME_norm", "Final_Assignment"]], on="NAME_norm", how="left")
        agg["Final_Assignment"] = agg["Final_Assignment"].astype(str).map(_norm_str)
        agg = agg.dropna(subset=[xcol, ycol])

        monitoring_avg_gdf = gpd.GeoDataFrame(
            agg, geometry=gpd.points_from_xy(agg[xcol], agg[ycol]), crs=basemap_gdf.crs
        )
    else:
        monitoring_avg_gdf = gpd.GeoDataFrame(
            columns=["NAME_norm", "NAME", "avg_conc", "Final_Assignment", xcol, ycol, "geometry"],
            geometry="geometry",
            crs=basemap_gdf.crs,
        )

    # -----------------------
    # Exports (tables)
    # -----------------------
    df_monitoring_avg_csv = monitoring_avg_gdf.drop(columns="geometry", errors="ignore").rename(
        columns={"avg_conc": f"{constituent}_avg_conc"}  # keep same header name pattern
    )
    char_df = d_w_aq[d_w_aq["TYPE"] == "CHARACTERIZATION"].copy()
    extr_df = d_w_aq[d_w_aq["TYPE"] == "EXTRACTION"].copy()

    f_monitoring = os.path.join(outdir, f"{constituent}_monitoring_avg_by_well.csv")
    f_char = os.path.join(outdir, f"{constituent}_characterization.csv")
    f_extr = os.path.join(outdir, f"{constituent}_extraction.csv")

    df_monitoring_avg_csv.to_csv(f_monitoring, index=False)
    char_df.to_csv(f_char, index=False)
    extr_df.to_csv(f_extr, index=False)

    # -----------------------
    # Maps (UU/MU and LU/CR)
    # -----------------------
    def _plot_assignment(label):
        label_norm = _norm_str(label)
        label_clean = label_norm.replace("/", "")

        gm = monitoring_avg_gdf[monitoring_avg_gdf["Final_Assignment"] == label_norm].copy()
        ch = all_points_gdf[
            (all_points_gdf["TYPE"] == "CHARACTERIZATION") &
            (all_points_gdf["Final_Assignment"] == label_norm)
        ].copy()

        fig, ax = plt.subplots(figsize=(9, 9))
        basemap_gdf.plot(ax=ax, alpha=0.2, linewidth=0.8, edgecolor="k")

        # Monitoring averages — colored by avg_conc (if any)
        if len(gm) > 0:
            sc = ax.scatter(
                gm[xcol], gm[ycol],
                c=gm["avg_conc"], s=70, alpha=0.9, marker="o",
            )
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label(f"{constituent} average concentration (from VAL)")
        else:
            ax.text(0.02, 0.98, "No monitoring averages in this assignment",
                    transform=ax.transAxes, va="top", ha="left")

        # Characterization wells — triangles
        ax.scatter(
            ch[xcol], ch[ycol],
            s=60, alpha=0.9, marker="^", label="Characterization",
        )

        # Force view to data extent to avoid blank-looking maps
        if len(gm) + len(ch) > 0:
            x_all = pd.concat([gm[xcol], ch[xcol]], ignore_index=True)
            y_all = pd.concat([gm[ycol], ch[ycol]], ignore_index=True)
            ax.set_xlim(x_all.min() - 50, x_all.max() + 50)
            ax.set_ylim(y_all.min() - 50, y_all.max() + 50)

        ax.set_title(f"{constituent} — {label_norm}: Monitoring avg (colored) + Characterization (triangles)")
        ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        ax.legend(loc="upper right")
        fig.tight_layout()

        fname = os.path.join(outdir, f"{constituent}_{label_clean}_monitoringAvg_plusCharacterization.png")
        fig.savefig(fname, dpi=220); plt.close(fig)
        return fname

    f_uumu = _plot_assignment("UU/MU")
    f_lucr = _plot_assignment("LU/CR")

    # Summary print
    print({
        "constituent": constituent,
        "rows_monitoring": int(len(monitoring_df)),
        "rows_characterization": int(len(characterization_df)),
        "rows_extraction": int(len(extraction_df)),
        "n_monitoring_wells_avg": int(len(monitoring_avg_gdf)),
        "csv_monitoring_avg": f_monitoring,
        "csv_characterization": f_char,
        "csv_extraction": f_extr,
        "png_UUMU": f_uumu,
        "png_LUCR": f_lucr,
    })

    return {
        "monitoring_df": monitoring_df,
        "characterization_df": characterization_df,
        "extraction_df": extraction_df,
        "monitoring_avg_gdf": monitoring_avg_gdf,
        "all_points_gdf": all_points_gdf,
        "used_xcol": xcol,
        "used_ycol": ycol,
        "paths": {
            "dir": outdir,
            "csv_monitoring_avg": f_monitoring,
            "csv_characterization": f_char,
            "csv_extraction": f_extr,
            "png_UUMU": f_uumu,
            "png_LUCR": f_lucr,
        },
    }

# -----------------------
# Run for all three constituents (writes to ../02_data/processed_data/<constituent>/)
# -----------------------
outroot = os.path.join("..", "02_data", "processed_data")

out_ctet = process_constituent(ctet_df, aq_df, basemap_gdf, constituent="CTET", outroot=outroot)
out_hexcr = process_constituent(hexcr_df, aq_df, basemap_gdf, constituent="HexCr", outroot=outroot)
out_tc99 = process_constituent(tc99_df, aq_df, basemap_gdf, constituent="Tc99", outroot=outroot)