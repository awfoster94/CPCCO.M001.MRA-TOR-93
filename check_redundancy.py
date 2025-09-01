"""
Check redundancy analysis
@author: rspinti

CONDA: use conda environment gstools

"""
import os
import numpy as np
import pandas as pd
import gstools as gs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error, r2_score
import rasterio
from rasterio.transform import from_origin
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import cKDTree
import logging
from joblib import Parallel, delayed
from datetime import datetime
import time
import gc
import argparse

def plot_variogram(data_subset, var_params, wdir):
    analyte, transform, stype, ran, hor, va, sill, nugget, var, len3 = var_params.iloc[0]

    print("Plotting Variogram for ", analyte)
    required_cols = ['XCOORDS', 'YCOORDS', 'ELE_m', 'VAL', 'NAME']
    if not all(col in data_subset.columns for col in required_cols):
        # logger.error(f"Missing columns: {required_cols}")
        data_subset.to_csv(f"{wdir}/invalid_data_subset.csv", index=False)
        return None
    if data_subset[required_cols].isnull().any().any():
        # logger.error(f"NaN/None values in data for {analyte}")
        data_subset.to_csv(f"{wdir}/invalid_data_subset.csv", index=False)
        return None
    if len(data_subset) < 3:
        # logger.error(f"Insufficient data points ({len(data_subset)}) for kriging")
        data_subset.to_csv(f"{wdir}/insufficient_data_subset.csv", index=False)
        return None

    coords = data_subset[['XCOORDS', 'YCOORDS', 'ELE_m']].values
    unique_coords, indices = np.unique(coords, axis=0, return_index=True)
    if len(unique_coords) < len(coords):
        # logger.warning(f"Found {len(coords) - len(unique_coords)} duplicate coordinates")
        data_subset = data_subset.iloc[indices].copy()

    x = data_subset['XCOORDS'].values
    y = data_subset['YCOORDS'].values
    z = data_subset['ELE_m'].fillna(0).values
    values = data_subset['VAL'].values
    values[values <= 0] = 1e-6

    if transform == 'log':
        log_values = np.log10(values)
        if np.any(np.isnan(log_values)) or np.any(np.isinf(log_values)):
            invalid_data = data_subset[np.isnan(log_values) | np.isinf(log_values)].copy()
            invalid_data['log_values'] = log_values[np.isnan(log_values) | np.isinf(log_values)]
            return None
        bin_center, gamma = gs.vario_estimate((x, y, z), log_values)
    else:
        q_values = np.log10(values)
        if analyte not in ['Iodine-129', 'Tritium']:
            if np.any(np.isnan(log_values)) or np.any(np.isinf(log_values)):
                invalid_data = data_subset[np.isnan(log_values) | np.isinf(log_values)].copy()
                invalid_data['log_values'] = log_values[np.isnan(log_values) | np.isinf(log_values)]
                return None
        bin_center, gamma = gs.vario_estimate((x, y, z), q_values)

    

    if stype == 'exponential':
        model = gs.Exponential(dim=3, var=var, len_scale=[ran, ran, len3], nugget=nugget)
    else:
        model = gs.Spherical(dim=3, var=var, len_scale=[ran, ran, len3], nugget=nugget)


    # Plot the variogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(bin_center, gamma, label='Experimental Variogram', color='blue')
    model.plot(ax=ax, label='Fitted Spherical Model', color='red')
    ax.set_xlabel('Lag Distance (m)')
    ax.set_ylabel('Semi-variogram')
    ax.set_title(f'Check Variogram for {analyte} (Baseline)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    variogram_file = f"{wdir}/variogram/variogram_baseline_check.png"
    plt.savefig(variogram_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print('Variance: ', var, 'Range: ', ran, 'Nugget: ', nugget)



# Run functions
cocs = ['Carbon tetrachloride', 'Chromium', 'Hexavalent Chromium', 'Iodine-129', 
        'Nitrate', 'Technetium-99', 'Trichloroethene', 'Tritium', 'Uranium']
special_ls = ['Chromium', 'Iodine-129']
var_csv = pd.read_csv(r'C:\Project_work\Hanford\Data_gap\01_plumemapping\inputs\inputs_csv\VariogramParameters.csv')
outputs = "C:/Project_work/Hanford/Data_gap/01_plumemapping/outputs/"

for c in cocs:
    wdir = os.path.join(outputs, c)
    data = pd.read_csv(os.path.join(wdir, "final_dataset_for_plume_mapping.csv"))
    # if c not in special_ls:
    #     data = pd.read_csv(os.path.join(wdir, "csv/final_dataset_for_plume_mapping.csv"))
    # elif c == special_ls[0]:
    #     data = pd.read_csv(os.path.join(wdir,"csv/ft6_dataset4kriging_add_ij_hsu_with_finalassignments_withHexCr.csv"))
    # else:
    #     data = pd.read_csv(os.path.join(wdir,"csv/ft6_dataset4kriging_add_ij_hsu_with_finalassignments_MODIFIED_FloorI129.csv"))
    
    var_params = var_csv[var_csv['analyte']==c]

    plot_variogram(data, var_params, wdir)
