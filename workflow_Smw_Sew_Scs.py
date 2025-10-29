# -*- coding: utf-8 -*-
"""
Created July 2025, Updated October 2025 for additional 200-UP-1 WMAs S & SX SY

@author: afoster
"""

##############################################################################
##############################################################################

########################## ECF-200ZP1-25-0092 ################################
############################ python workflow #################################

#### This workflow supports calculation of 3 scoring criteria in Eq (6)  ####
#### of ECF-200ZP1-0098 Rev.0 in Section 4.2.7 Total Score (Originally). ####
#### For each potential well location, the existing extraction and       ####
#### monitoring wells identified in the redundancy analysis are used:    ####
#### 1. To calculate, Smw (proximal distance to closest monitoring well) ####
#### 2. To calculate, Sew (proximal distance to closest extraction well) ####
#### 3. To calculate, Sp-cs (proximal distance continuing source)        ####
#### Note. Sp-cs uses particle tracking (mp3du) as a basis for the calc  ####


##############################################################################
##############################################################################

# note make sure to create the virtual environments from the .yml file 
# mp3du-env-env.yml

# import necessary python packages and libraries
import os
import glob
import subprocess
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import shapefile
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import LineString
from osgeo import gdal
from datetime import datetime
start_time = datetime.now()
import shapefile
from math import sin, cos, pi
import numpy as np
import time
import json
from scipy.spatial import cKDTree
from functools import reduce

# define some global variables
ecf_name = 'ECF-200ZP1-25-0092'

# collect the local working directory
cwd = os.getcwd()

# define modflow executable
mf_executable = 'mf2k-mst-cpcc09dpv.exe'

# define the mp3du executables & post porcessing exectuables
mp3du_executable = 'mp3du.exe'
gsf_executable = 'writep3dgsf.exe'
writep3doutput_executable = 'writep3doutput.exe'

exe_list = [mf_executable, mp3du_executable, gsf_executable, writep3doutput_executable]

# define list of subfolders of flow+particle tracks
constituent_list = ['cr', 'tec-99']

exe_d = os.path.join(cwd, 'bin', 'win')
flow_source_d = os.path.join(cwd, 'source_files', 'flow', 'source')
transport_source_d = os.path.join(cwd, 'source_files', 'transport')
ptrk_calc_d = os.path.join(cwd, 'calcs', 'ptrack')
ptrk_ssx_calc_d = os.path.join(cwd, 'calcs', 'ptrack-ssx')
gis_d = os.path.join(cwd, 'gis')
fig_d = os.path.join(cwd, 'figs')

# global boolen to turn the ECF workflow on, must be turned to call helper functions in the main()
flag_new_ecf = True

# booleans to turn on each calc function incrementally that are called in main()
# perform sequentially by turning each boolen on, running function, then turning off before running the next function. 
# Please note, figures will need to be closed as they open up for the workflow to continue forward

################################################################################################
# FIRST CODE BLOCK TO RUN. Can be run in series, turn off after successfuly completion.
################################################################################################
# Smw calc - scoring for distance/proximity calcs monitoring wells
flag_proximal_distance_to_mws = False

# Sew, calc - scoring distance/proximity calcs extraction wells 
flag_proximal_distance_to_ews = False
################################################################################################

################################################################################################
# SECOND CODE BLOCK TO RUN. Can be run in series, turn off after successfuly completion.
################################################################################################
# Sp-cs particle tracking calculations for continuing sources of cr & tec99 for scoring
flag_create_ptrk_folder = False
flag_copy_transport_props = False
flag_run_modflow = False
flag_write_gsf_json_input = False
flag_run_gsfwriter = False
flag_modify_nam_file_with_new_package_mp3du = False
flag_write_p3d_mp3du = False
flag_generate_part_start_locs = False
flag_mp3du_json_input = False
flag_run_mp3du = False
flag_write_p3doutput_json_input_path = False
flag_write_p3doutput_json_input_endpts = False
flag_run_writep3doutput = False
flag_generate_pathlines_map = False
flag_calc_relative_path_count = False
flag_relcount_pathlines_map = False
flag_generate_endpoints_map = False
flag_generate_pathlines_endpoints_map = False
################################################################################################

################################################################################################
# THIRD CODE BLOCK TO RUN. Can be run in series, turn off after successfuly completion.
################################################################################################
# Sp-cs continued generate bounding polygons and respective centerlines for continuing source areas
flag_parse_source_zones = False

# please note, the next two function calls will require user input on the figure for continuing source site
# Step 1. expand, maximize figure window 
# Step 2. draw bounding polygon with clicking around particle pathlines & respective endpoints
# Step 3. close figure manually
# Step 4. expand, maximize figure window
# Step 5. draw centerline through plume center mass through drawn polygon by clicking to make a polyline
# Step 6. close figure

# Step 7. repeat Steps 1-6 for each source area site (2 chromium, 2 technetium-99)
flag_generate_bounding_and_centerline = False # Turn this boolean on, if new bounding and centerline generation is desired, otherwise will keep existing shapefiles.

if flag_generate_bounding_and_centerline:
    flag_generate_bounding_polygon = True # don't edit this, boolean above will provide direction
    flag_generate_centerline = True # don't edit this, boolean above will provide direction
else:
    flag_generate_bounding_polygon = False # don't edit this, boolean above will provide direction
    flag_generate_centerline = False # don't edit this, boolean above will provide direction

# plot results of bounding polygon and centerline
flag_generate_bounding_centerline_map_all = False

# calculate scoring
flag_centerline_to_points = False
flag_potential_wells_in_bounding = False

# all constituents combined
flag_calculate_continuous_source_score_all = False
flag_generate_continuous_source_score_map_all = False

# chromium only
flag_calculate_continuous_source_score_chromium = False
flag_generate_continuous_source_score_map_chromium = False

# tec-99 only
flag_calculate_continuous_source_score_tec99 = False
flag_generate_continuous_source_score_map_tec99 = False

# ctet only (no continuing source area, so all zeros!)
flag_calculate_continuous_source_score_ctet = False

# combine scores with details for Smw, Sew, Scs
flag_combine_all_scores_detailed = False
# combine scores only for Smw, Sew, Scs
flag_combine_all_scores_only = False
################################################################################################

# define helper functions for the workflow calculations

# this function calculates the proximal distance to nearest monitoring well for each potential well cell
def calc_proximal_distance_to_mws(flag, gis_d, fig_d, data_gap_wells_flag=[]):
    
    if flag: 
        # create output directory for shapefile summary
        outdir = os.path.join(gis_d, 'shp', 'scores')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # list of HSUs
        hsu_list = ['UU/MU', 'LU/CR']
        for hsu in hsu_list:
            print(hsu)
            if hsu == 'UU/MU':
                hsu_tag = 'uu_mu'
            if hsu == 'LU/CR':
                hsu_tag = 'lu_cr'

            # load in potential wells shapefile as geopandas gdf & store crs for reference
            potential_wells_gdf = gpd.read_file(os.path.join('gis','shp','data_gap_wells', 'potential_wells.shp'))
            crs_ref = potential_wells_gdf.crs

            # define pandas padataframe for looping through each potential well cell
            potential_wells_df = pd.DataFrame(columns=('row', 'col', 'EASTING', 'NORTHING'))
            potential_wells_df['row'] = potential_wells_gdf['row']
            potential_wells_df['col'] = potential_wells_gdf['col']
            potential_wells_df['EASTING'] = potential_wells_gdf['x']
            potential_wells_df['NORTHING'] = potential_wells_gdf['y']
            potential_wells_df = potential_wells_df.reset_index(drop=True)


            if data_gap_wells_flag == 'updated':

                # load in updated analysis well locations
                all_2025_wells = pd.read_csv(os.path.join('gis', 'xlsx', 'ECF-200-ZP1-25-0092', 'Table-A-2-Candidate-Wells', 'Table-A-2-Candidate-Wells.csv'))
                all_2025_wells = all_2025_wells[all_2025_wells['Final_Assignment'] == hsu]
                all_2025_wells['WELL_NAME'] = all_2025_wells['NAME']
                print(all_2025_wells)
                # load in recent hwis data pull well location information
                hwis_pull_csv = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93.csv'), encoding="cp1252") # may need to modify or remove coding, if loading error occurs
                hwis_pull_csv_reduced = hwis_pull_csv[['WELL_ID', 'WELL_NAME', 'NORTHING', 'EASTING']]
                hwis_pull_csv_reduced['NORTHING_HWIS'] = hwis_pull_csv_reduced['NORTHING']
                hwis_pull_csv_reduced['EASTING_HWIS'] = hwis_pull_csv_reduced['EASTING']
                

                all_2025_wells_merged = all_2025_wells.merge(hwis_pull_csv_reduced, how='left', on='WELL_NAME', suffixes=("", "_new"))

                # load in recent heis data pull well location information
                Manual_GW_CY2025_GWSR_TO93_txt = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HEIS_Data_Pull', 'Manual_GW_CY2025_GWSR_TO93.txt'), delimiter=',', header=0)
                Manual_GW_CY2025_GWSR_TO93_txt_reduced = Manual_GW_CY2025_GWSR_TO93_txt[['WELL_NAME', 'NORTHING', 'EASTING']]
                Manual_GW_CY2025_GWSR_TO93_txt_reduced = Manual_GW_CY2025_GWSR_TO93_txt_reduced.drop_duplicates(subset='WELL_NAME')
                all_2025_wells_merged_2 = all_2025_wells_merged.merge(Manual_GW_CY2025_GWSR_TO93_txt_reduced, how='left', on='WELL_NAME', suffixes=(" ", "_GWSR"))

                # load if necessary
                #qryAWLN_SSPA_TO93_txt = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HEIS_Data_Pull', 'qryAWLN_SSPA_TO93.txt'), delimiter=',', header=0)
                #qryAWLN2021_Present_TO93_txt = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HEIS_Data_Pull', 'qryAWLN2021_Present_TO93.txt'), delimiter=',', header=0)
                
                qryMANHEIS_TO93_txt = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HEIS_Data_Pull', 'qryMANHEIS_TO93.txt'), delimiter=',', header=0)
                qryMANHEIS_TO93_txt_reduced = qryMANHEIS_TO93_txt[['WELL_NAME', 'NORTHING', 'EASTING']]
                qryMANHEIS_TO93_txt_reduced['NORTHING_MANHEIS'] = qryMANHEIS_TO93_txt_reduced['NORTHING']
                qryMANHEIS_TO93_txt_reduced['EASTING_MANHEIS'] = qryMANHEIS_TO93_txt_reduced['EASTING']
                qryMANHEIS_TO93_txt_reduced = qryMANHEIS_TO93_txt_reduced[['WELL_NAME', 'NORTHING_MANHEIS', 'EASTING_MANHEIS']]
                qryMANHEIS_TO93_txt_reduced = qryMANHEIS_TO93_txt_reduced.drop_duplicates(subset='WELL_NAME')
                all_2025_wells_merged_3 = all_2025_wells_merged_2.merge(qryMANHEIS_TO93_txt_reduced, how='left', on='WELL_NAME', suffixes=(" ", "_HEIS"))
                all_2025_wells_merged_3['NORTHING'] = all_2025_wells_merged_3['NORTHING '].fillna(all_2025_wells_merged_3['NORTHING_MANHEIS'])
                all_2025_wells_merged_3['EASTING'] = all_2025_wells_merged_3['EASTING '].fillna(all_2025_wells_merged_3['EASTING_MANHEIS'])
                all_2025_wells_merged_3['NORTHING'] = all_2025_wells_merged_3['NORTHING'].fillna(all_2025_wells_merged_3['northing_m_check'])
                all_2025_wells_merged_3['EASTING'] = all_2025_wells_merged_3['EASTING'].fillna(all_2025_wells_merged_3['easting_m_check'])
                
                all_2025_wells_merged_3['Well_Name'] = all_2025_wells_merged_3['WELL_NAME']
                #all_2025_wells_merged_3 = all_2025_wells_merged_3[['Well_Name', 'NAME', 'GWIA', 'easting_m_check', 'northing_m_check', 'Final_Assignment', 'TYPE_scripts', 'HSU_source', 'WELL_NAME', 'WELL_ID', 'NORTHING', 'EASTING']]
                all_2025_wells_merged_3 = all_2025_wells_merged_3[['Well_Name', 'NAME', 'GWIA', 'Final_Assignment', 'TYPE_scripts', 'HSU_source', 'WELL_ID', 'NORTHING', 'EASTING']]

                # determine extractors & monitoring wells
                ext_wells_list = ['EXTRACTION']
                candidate_wells_ext = all_2025_wells_merged_3[all_2025_wells_merged_3['TYPE_scripts'].isin(ext_wells_list)]

                mw_wells_list = ['MONITORING'] #'CHARACTERIZATION']
                candidate_wells_mw = all_2025_wells_merged_3[all_2025_wells_merged_3['TYPE_scripts'].isin(mw_wells_list)]

            if data_gap_wells_flag == 'existing':

                # load in recent hwis pull
                hwis_pull_csv = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93.csv'), encoding="cp1252") # may need to modify or remove coding, if loading error occurs
                
                # in the recent hwis pull well name 299-W19-113 is not identified as extraction, but in the previous (existing) ECF-200-ZP1-22-0098 it is..
                hwis_pull_csv.loc[hwis_pull_csv['WELL_NAME'] == '299-W19-113', 'WELL_PROJECT_PURPOSES'] = 'EXTRACTION'
                
                ext_wells_list = ['DATA REPORT, EXTRACTION', 'EXTRACTION', 'EXTRACTION, GROUNDWATER SAMPLE']
                hwis_pull_csv_ext = hwis_pull_csv[hwis_pull_csv['WELL_PROJECT_PURPOSES'].isin(ext_wells_list)]

                # load exsiting wells dataset from previous (existing) ECF-200-ZP1-22-0098 Table A-2 Candidate Wells 
                existing_candidate_wells = pd.read_csv(os.path.join(gis_d, 'xlsx', 'ECF-200-ZP1-22-0098', 'Table-A-2-Candidate-Wells', 'Table-A-2-Candidate-Wells_A20-A40.csv'))

                # define candidate extraction wells        
                candidate_wells_ext = existing_candidate_wells[existing_candidate_wells['Well_Name'].isin(hwis_pull_csv_ext['WELL_NAME'].to_list())]

                # define candidate monitoring wells
                candidate_wells_mw = existing_candidate_wells[~existing_candidate_wells['Well_Name'].isin(candidate_wells_ext['Well_Name'].to_list())]
                
            # calculate proximity/distances for each data gap grid cell for extraction wells
            mw_proximity_dist_m = []
            for i in range(0, len(potential_wells_df)):
                #print(f"i is: {i}")
                #print(f"the Easting (m), Northing (m) is: {potential_wells_df.iloc[i]['EASTING']}, {potential_wells_df.iloc[i]['NORTHING']}")
                dist_mw_m = []
                for j in range(0, len(candidate_wells_mw)):

                    # calc distance to each potential well location for each monitoring well
                    dist_m = np.sqrt(((candidate_wells_mw.iloc[j]['EASTING'] - potential_wells_df.iloc[i]['EASTING']) ** 2) + ((candidate_wells_mw.iloc[j]['NORTHING'] - potential_wells_df.iloc[i]['NORTHING']) ** 2))
                    dist_mw_m.append((candidate_wells_mw.iloc[j]['Well_Name'], dist_m))

                dist_mw_m_df = pd.DataFrame(dist_mw_m, columns=('Well_Name', 'Distance_meters'))
                prox_mw_df = dist_mw_m_df.loc[dist_mw_m_df['Distance_meters'].idxmin()]
                mw_proximity_dist_m.append((potential_wells_df.iloc[i]['row'], potential_wells_df.iloc[i]['col'], potential_wells_df.iloc[i]['EASTING'], potential_wells_df.iloc[i]['NORTHING'], prox_mw_df['Well_Name'], prox_mw_df['Distance_meters']))
            mw_proximity_dist_m_df = pd.DataFrame(mw_proximity_dist_m, columns=('row', 'col', 'EASTING_m', 'NORTHING_m', 'Mw_Well_Name', 'Distance_meters'))
            
            # calculate monitoring well scoring based on distances
            mw_proximity_dist_m_df['score_mw'] = -9999
            for k in range(0, len(mw_proximity_dist_m_df)):

                dist_m = mw_proximity_dist_m_df.at[k, 'Distance_meters']

                # assign scoring
                if dist_m < 200:
                    mw_proximity_dist_m_df.at[k, 'score_mw'] = 0
                elif 200 <= dist_m < 400:
                    mw_proximity_dist_m_df.at[k, 'score_mw'] = 1
                elif 400 <= dist_m < 600:
                    mw_proximity_dist_m_df.at[k, 'score_mw'] = 2
                elif 600 <= dist_m < 800:
                    mw_proximity_dist_m_df.at[k, 'score_mw'] = 3
                elif dist_m >= 800:
                    mw_proximity_dist_m_df.at[k, 'score_mw'] = 4

            # add row_col_id
            mw_proximity_dist_m_df['row_col_id'] = 10000000+(10000*mw_proximity_dist_m_df['row'])+mw_proximity_dist_m_df['col']
            
            # create and export shapefile of results
            mw_proximity_dist_m_gdf = gpd.GeoDataFrame(mw_proximity_dist_m_df, geometry = gpd.points_from_xy(mw_proximity_dist_m_df.EASTING_m, mw_proximity_dist_m_df.NORTHING_m), crs=crs_ref)
            mw_proximity_dist_m_gdf.to_file(os.path.join(outdir, f'score_monitoring_well_proximity_{hsu_tag}.shp'))

            # create geodrataframes with geopandas to plot
            candidate_wells_ext_gdf = gpd.GeoDataFrame(candidate_wells_ext, geometry = gpd.points_from_xy(candidate_wells_ext.EASTING, candidate_wells_ext.NORTHING), crs=crs_ref)
            candidate_wells_mw_gdf = gpd.GeoDataFrame(candidate_wells_mw, geometry = gpd.points_from_xy(candidate_wells_mw.EASTING, candidate_wells_mw.NORTHING), crs=crs_ref)

            # export candidate monitoring wells
            candidate_wells_mw_gdf.to_file(os.path.join(outdir, f'candidate_mws_{hsu_tag}.shp'))

            # plot the results

            # Define the color bands and corresponding colors
            color_bands = [(0.0000000001, 0.1), (0.1, 1.01), (1.01, 2.01), (2.01, 3.01), (3.01, 4.01)]
            colors = ['lightyellow', 'yellow', 'darkorange', 'orangered', 'red']
                
            fig,ax = plt.subplots(figsize=(10,10), dpi=400)
            potential_wells_gdf.plot(ax=ax, color='grey', markersize=2, label='potential well locations')
            colorflood_legend_elements = []
            # plot proximity score here
            for (low, high), color in zip(color_bands, colors):
                subset = mw_proximity_dist_m_gdf[(mw_proximity_dist_m_gdf['score_mw'] >= low) & (mw_proximity_dist_m_gdf['score_mw'] < high)]
                subset.plot(ax=ax, marker='o', edgecolor=color, facecolor=color, alpha=1, label=f'{low}-{high}')
                # Add corresponding legend patch
                colorflood_legend_elements.append(Patch(facecolor=color, edgecolor='black', alpha=1, label=f'{low}-{high}'))
            #candidate_wells_ext_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells')
            candidate_wells_mw_gdf.plot(ax=ax, facecolor='lightgreen', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells')
            
            plt.legend()
            plt.title(f'Score - Monitoring Well Proximal Distance \n {hsu}')
            plt.ylabel('Northing (m)')
            plt.xlabel('Easting (m)')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_d, f'score_monitoring_well_proximity_{hsu_tag}.png'), dpi=400)
            plt.show()
            plt.close()

    else:
        print('calc_proximal_distance_to_mws function selected to NOT run...')

# this function calculates the proximal distance to nearest extraction well for each potential well cell
def calc_proximal_distance_to_ews(flag, gis_d, fig_d, data_gap_wells_flag=[]):

    if flag: 
        print('calculating proximal distance to extraction wells...')
        # create output directory for shapefile summary
        outdir = os.path.join(gis_d, 'shp', 'scores')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # list of HSUs
        hsu_list = ['UU/MU', 'LU/CR']
        for hsu in hsu_list:
            print(hsu)
            if hsu == 'UU/MU':
                hsu_tag = 'uu_mu'
            if hsu == 'LU/CR':
                hsu_tag = 'lu_cr'

            # load in potential wells shapefile as geopandas gdf & store crs for reference
            potential_wells_gdf = gpd.read_file(os.path.join('gis','shp','data_gap_wells', 'potential_wells.shp'))
            crs_ref = potential_wells_gdf.crs

            # define pandas padataframe for looping through each potential well cell
            potential_wells_df = pd.DataFrame(columns=('row', 'col', 'EASTING', 'NORTHING'))
            potential_wells_df['row'] = potential_wells_gdf['row']
            potential_wells_df['col'] = potential_wells_gdf['col']
            potential_wells_df['EASTING'] = potential_wells_gdf['x']
            potential_wells_df['NORTHING'] = potential_wells_gdf['y']
            potential_wells_df = potential_wells_df.reset_index(drop=True)


            if data_gap_wells_flag == 'updated':

                # load in updated analysis well locations
                all_2025_wells = pd.read_csv(os.path.join('gis', 'xlsx', 'ECF-200-ZP1-25-0092', 'Table-A-2-Candidate-Wells', 'Table-A-2-Candidate-Wells.csv'))
                all_2025_wells = all_2025_wells[all_2025_wells['Final_Assignment'] == hsu]
                all_2025_wells['WELL_NAME'] = all_2025_wells['NAME']
                print(all_2025_wells)
                # load in recent hwis data pull well location information
                hwis_pull_csv = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93.csv'), encoding="cp1252") # may need to modify or remove coding, if loading error occurs
                hwis_pull_csv_reduced = hwis_pull_csv[['WELL_ID', 'WELL_NAME', 'NORTHING', 'EASTING']]
                hwis_pull_csv_reduced['NORTHING_HWIS'] = hwis_pull_csv_reduced['NORTHING']
                hwis_pull_csv_reduced['EASTING_HWIS'] = hwis_pull_csv_reduced['EASTING']
                

                all_2025_wells_merged = all_2025_wells.merge(hwis_pull_csv_reduced, how='left', on='WELL_NAME', suffixes=("", "_new"))

                # load in recent heis data pull well location information
                Manual_GW_CY2025_GWSR_TO93_txt = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HEIS_Data_Pull', 'Manual_GW_CY2025_GWSR_TO93.txt'), delimiter=',', header=0)
                Manual_GW_CY2025_GWSR_TO93_txt_reduced = Manual_GW_CY2025_GWSR_TO93_txt[['WELL_NAME', 'NORTHING', 'EASTING']]
                Manual_GW_CY2025_GWSR_TO93_txt_reduced = Manual_GW_CY2025_GWSR_TO93_txt_reduced.drop_duplicates(subset='WELL_NAME')
                all_2025_wells_merged_2 = all_2025_wells_merged.merge(Manual_GW_CY2025_GWSR_TO93_txt_reduced, how='left', on='WELL_NAME', suffixes=(" ", "_GWSR"))

                # load if necessary
                #qryAWLN_SSPA_TO93_txt = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HEIS_Data_Pull', 'qryAWLN_SSPA_TO93.txt'), delimiter=',', header=0)
                #qryAWLN2021_Present_TO93_txt = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HEIS_Data_Pull', 'qryAWLN2021_Present_TO93.txt'), delimiter=',', header=0)
                
                qryMANHEIS_TO93_txt = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HEIS_Data_Pull', 'qryMANHEIS_TO93.txt'), delimiter=',', header=0)
                qryMANHEIS_TO93_txt_reduced = qryMANHEIS_TO93_txt[['WELL_NAME', 'NORTHING', 'EASTING']]
                qryMANHEIS_TO93_txt_reduced['NORTHING_MANHEIS'] = qryMANHEIS_TO93_txt_reduced['NORTHING']
                qryMANHEIS_TO93_txt_reduced['EASTING_MANHEIS'] = qryMANHEIS_TO93_txt_reduced['EASTING']
                qryMANHEIS_TO93_txt_reduced = qryMANHEIS_TO93_txt_reduced[['WELL_NAME', 'NORTHING_MANHEIS', 'EASTING_MANHEIS']]
                qryMANHEIS_TO93_txt_reduced = qryMANHEIS_TO93_txt_reduced.drop_duplicates(subset='WELL_NAME')
                all_2025_wells_merged_3 = all_2025_wells_merged_2.merge(qryMANHEIS_TO93_txt_reduced, how='left', on='WELL_NAME', suffixes=(" ", "_HEIS"))

                all_2025_wells_merged_3['NORTHING'] = all_2025_wells_merged_3['NORTHING '].fillna(all_2025_wells_merged_3['NORTHING_MANHEIS'])
                all_2025_wells_merged_3['EASTING'] = all_2025_wells_merged_3['EASTING '].fillna(all_2025_wells_merged_3['EASTING_MANHEIS'])
                all_2025_wells_merged_3['NORTHING'] = all_2025_wells_merged_3['NORTHING'].fillna(all_2025_wells_merged_3['northing_m_check'])
                all_2025_wells_merged_3['EASTING'] = all_2025_wells_merged_3['EASTING'].fillna(all_2025_wells_merged_3['easting_m_check'])
                all_2025_wells_merged_3['Well_Name'] = all_2025_wells_merged_3['WELL_NAME']
                #all_2025_wells_merged_3 = all_2025_wells_merged_3[['Well_Name', 'NAME', 'GWIA', 'easting_m_check', 'northing_m_check', 'Final_Assignment', 'TYPE_scripts', 'HSU_source', 'WELL_NAME', 'WELL_ID', 'NORTHING', 'EASTING']]
                all_2025_wells_merged_3 = all_2025_wells_merged_3[['Well_Name', 'NAME', 'GWIA', 'Final_Assignment', 'TYPE_scripts', 'HSU_source', 'WELL_ID', 'NORTHING', 'EASTING']]

                # determine extractors & monitoring wells
                ext_wells_list = ['EXTRACTION']
                candidate_wells_ext = all_2025_wells_merged_3[all_2025_wells_merged_3['TYPE_scripts'].isin(ext_wells_list)]

                mw_wells_list = ['MONITORING'] #'CHARACTERIZATION']
                candidate_wells_mw = all_2025_wells_merged_3[all_2025_wells_merged_3['TYPE_scripts'].isin(mw_wells_list)]

            if data_gap_wells_flag == 'existing':

                # load in recent hwis pull
                hwis_pull_csv = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93.csv'), encoding="cp1252") # may need to modify or remove coding, if loading error occurs
                
                # in the recent hwis pull well name 299-W19-113 is not identified as extraction, but in the previous (existing) ECF-200-ZP1-22-0098 it is..
                hwis_pull_csv.loc[hwis_pull_csv['WELL_NAME'] == '299-W19-113', 'WELL_PROJECT_PURPOSES'] = 'EXTRACTION'
                
                ext_wells_list = ['DATA REPORT, EXTRACTION', 'EXTRACTION', 'EXTRACTION, GROUNDWATER SAMPLE']
                hwis_pull_csv_ext = hwis_pull_csv[hwis_pull_csv['WELL_PROJECT_PURPOSES'].isin(ext_wells_list)]

                # load wells dataset from previous (existing) ECF-200-ZP1-22-0098 Table A-2 Candidate Wells 
                candidate_wells = pd.read_csv(os.path.join(gis_d, 'xlsx', 'ECF-200-ZP1-22-0098', 'Table-A-2-Candidate-Wells', 'Table-A-2-Candidate-Wells_A20-A40.csv'))
                candidate_wells = candidate_wells[candidate_wells['HSU_Assignment'] == hsu]
                # define candidate extraction wells        
                candidate_wells_ext = candidate_wells[candidate_wells['Well_Name'].isin(hwis_pull_csv_ext['WELL_NAME'].to_list())]

                # define candidate monitoring wells
                candidate_wells_mw =candidate_wells[~candidate_wells['Well_Name'].isin(candidate_wells_ext['Well_Name'].to_list())]
                
            # calculate proximity/distances for each data gap grid cell for extraction wells
            ext_proximity_dist_m = []
            for i in range(0, len(potential_wells_df)):
                #print(f"i is: {i}")
                #print(f"the Easting (m), Northing (m) is: {potential_wells_df.iloc[i]['EASTING']}, {potential_wells_df.iloc[i]['NORTHING']}")
                dist_calc_ext = []
                dist_ext_m = []
                for j in range(0, len(candidate_wells_ext)):
                    # calc distance to each potential well location for each monitoring well
                    dist_m = np.sqrt(((candidate_wells_ext.iloc[j]['EASTING'] - potential_wells_df.iloc[i]['EASTING']) ** 2) + ((candidate_wells_ext.iloc[j]['NORTHING'] - potential_wells_df.iloc[i]['NORTHING']) ** 2))

                    dist_ext_m.append((candidate_wells_ext.iloc[j]['Well_Name'], dist_m))

                dist_ext_m_df = pd.DataFrame(dist_ext_m, columns=('Well_Name', 'Distance_meters'))
                prox_ext_df = dist_ext_m_df.loc[dist_ext_m_df['Distance_meters'].idxmin()]
                ext_proximity_dist_m.append((potential_wells_df.iloc[i]['row'], potential_wells_df.iloc[i]['col'], potential_wells_df.iloc[i]['EASTING'], potential_wells_df.iloc[i]['NORTHING'], prox_ext_df['Well_Name'], prox_ext_df['Distance_meters']))
            ext_proximity_dist_m_df = pd.DataFrame(ext_proximity_dist_m, columns=('row', 'col', 'EASTING_m', 'NORTHING_m', 'Ext_Well_Name', 'Distance_meters'))
            
            # calculate extraction well scoring based on distances
            ext_proximity_dist_m_df['score_ext'] = -9999
            for k in range(0, len(ext_proximity_dist_m_df)):

                dist_m = ext_proximity_dist_m_df.at[k, 'Distance_meters']

                # assign scoring
                if dist_m < 50:
                    ext_proximity_dist_m_df.at[k, 'score_ext'] = 4
                elif 50 <= dist_m < 100:
                    ext_proximity_dist_m_df.at[k, 'score_ext'] = 3
                elif 100 <= dist_m < 200:
                    ext_proximity_dist_m_df.at[k, 'score_ext'] = 2
                elif 200 <= dist_m < 300:
                    ext_proximity_dist_m_df.at[k, 'score_ext'] = 1
                elif dist_m >= 300:
                    ext_proximity_dist_m_df.at[k, 'score_ext'] = 0

            # add row_col_id
            ext_proximity_dist_m_df['row_col_id'] = 10000000+(10000*ext_proximity_dist_m_df['row'])+ext_proximity_dist_m_df['col']

            # create and export shapefile of results
            ext_proximity_dist_m_gdf = gpd.GeoDataFrame(ext_proximity_dist_m_df, geometry = gpd.points_from_xy(ext_proximity_dist_m_df.EASTING_m, ext_proximity_dist_m_df.NORTHING_m), crs=crs_ref)
            ext_proximity_dist_m_gdf.to_file(os.path.join(outdir, f'score_extraction_well_proximity_{hsu_tag}.shp'))

            # create geodrataframes with geopandas to plot
            candidate_wells_ext_gdf = gpd.GeoDataFrame(candidate_wells_ext, geometry = gpd.points_from_xy(candidate_wells_ext.EASTING, candidate_wells_ext.NORTHING), crs=crs_ref)
            candidate_wells_mw_gdf = gpd.GeoDataFrame(candidate_wells_mw, geometry = gpd.points_from_xy(candidate_wells_mw.EASTING, candidate_wells_mw.NORTHING), crs=crs_ref)

            # export candidate extraction wells
            candidate_wells_ext_gdf.to_file(os.path.join(outdir, f'candidate_ews_{hsu_tag}.shp'))

            # plot the results

            # Define the color bands and corresponding colors
            color_bands = [(0.0000000001, 0.1), (0.1, 1.01), (1.01, 2.01), (2.01, 3.01), (3.01, 4.01)]
            colors = ['lightyellow', 'yellow', 'darkorange', 'orangered', 'red']
                
            fig,ax = plt.subplots(figsize=(10,10), dpi=400)
            potential_wells_gdf.plot(ax=ax, color='grey', markersize=2, label='potential well locations')
            colorflood_legend_elements = []
            # plot proximity score here
            for (low, high), color in zip(color_bands, colors):
                subset = ext_proximity_dist_m_gdf[(ext_proximity_dist_m_gdf['score_ext'] >= low) & (ext_proximity_dist_m_gdf['score_ext'] < high)]
                subset.plot(ax=ax, marker='o', edgecolor=color, facecolor=color, alpha=1, label=f'{low}-{high}')
                # Add corresponding legend patch
                colorflood_legend_elements.append(Patch(facecolor=color, edgecolor='black', alpha=1, label=f'{low}-{high}'))
            candidate_wells_ext_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label=f'candidate extraction wells {hsu}')
            #candidate_wells_mw_gdf.plot(ax=ax, facecolor='lightgreen', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells')
            
            plt.legend()
            plt.title(f'Score - Extraction Well Proximal Distance \n {hsu}')
            plt.ylabel('Northing (m)')
            plt.xlabel('Easting (m)')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_d, f'score_extraction_well_proximity_{hsu_tag}.png'), dpi=400)
            plt.show()
            plt.close()
    else:
        print('calc_proximal_distance_to_ews function selected NOT to run...')

# this function creates a particle tracking folder for the calculations
def create_ptrk_folder(flag, constituent, flow_source_d, ptrk_calc_d, exe_d, exe_list):
    if flag:
        print('creating particle tracking folder...')
        if not os.path.exists(os.path.join(ptrk_calc_d, constituent)):
            os.makedirs(os.path.join(ptrk_calc_d, constituent))

            # copy original directory to template directory
        for filename in os.listdir(flow_source_d):
            flow_source_fpth = os.path.join(flow_source_d, filename)
            ptrk_calc_fpth = os.path.join(ptrk_calc_d, constituent, filename)
        
            # copy all the files to the new folder
            if os.path.isfile(flow_source_fpth):
                shutil.copy(flow_source_fpth, ptrk_calc_fpth)
            
            for exe in exe_list:
                exe_ptrk_fpth = os.path.join(ptrk_calc_d, constituent, exe)
                shutil.copy(os.path.join(exe_d, exe), os.path.join(ptrk_calc_d, constituent))
    else:
        print('create_ptrk_folder selected NOT to run...')

# this function copies transport parameter source files into the ptrk calc directory
def copy_transport_props(flag, transport_source_d, ptrk_calc_d):
    if flag:
        print('copying shared transport parameters to particle tracking folder...')
        if not os.path.exists(os.path.join(ptrk_calc_d, 'transport', 'SharedFiles')):
            os.makedirs(os.path.join(ptrk_calc_d, 'transport', 'SharedFiles'))

            # copy original directory to template directory
        for filename in os.listdir(os.path.join(transport_source_d, 'SharedFiles')):
            transport_source_fpth = os.path.join(transport_source_d, 'SharedFiles', filename)
            ptrk_calc_fpth = os.path.join(ptrk_calc_d, 'transport', 'SharedFiles', filename)
        
            # copy all the files to the new folder
            if os.path.isfile(transport_source_fpth):
                shutil.copy(transport_source_fpth, ptrk_calc_fpth)
    else:
        print('copy_transport_props function selected NOT to run...')

# this function runs mf2k-mst-cpcc09dpv.exe modflow flow model
def run_modflow(flag, constituent, folder_path, executable, input_file):
    if flag:
        print('running mf2k-mst-cpcc09dpv.exe...\nnote model time is in: days\nmodel length is in: meters')
        try:
            # store current directory
            cwd = os.getcwd()
            # change working directory to CreateSubGrid folder
            os.chdir(os.path.join(folder_path, constituent))
            # Construct the command
            command = [executable, input_file, 'colorcode']

            # Run the command
            result = subprocess.run(command, shell=False, text=True, capture_output=True)
            # Check for successful execution
            if result.returncode == 0:
                print("Execution completed successfully!")
            else:
                print(f"Error occurred:\n{result.stderr}")
            # return back to the original working directory
            os.chdir(cwd)
        except FileNotFoundError:
            print("Error: Executable not found. Please check the path.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print('run_modflow function selected NOT to run')

# this function writes the gsf file input json
def write_gsf_json_input(flag, constituent, ptrk_calc_d, xoff, yoff, rot, flow_fnm, dis_type, gsf_nm, gsf_json_nm):
    if flag:
        print('creating gsf json...')
        data = {
            "TRANSFORMATION": {
                "XOFF": xoff,
                "YOFF": yoff,
                "ROT": rot
            },
            "FLOW_MODEL_TYPE": {
                "MODFLOW": {
                    "NAME_FILE": flow_fnm,
                    "GSF_FILE": {
                        "TYPE": dis_type
                    }
                }
            },
            "OUTPUT_FILENAME": gsf_nm
        }
        text = json.dumps(data, indent=4)
        ### output phreeqc input file
        with open(os.path.join(ptrk_calc_d, constituent, str(gsf_json_nm)+".json"), "w") as file:
            file.write(text)
    else:
        print('write_gsf_json_input Function selected NOT to run')

# this function runs the write3dgsf.exe 
def run_gsfwriter(flag, constituent, folder_path, executable, input_file):
    if flag:
        print('running writep3dgsf.exe...')
        try:
            # store current directory
            cwd = os.getcwd()
            # change working directory to CreateSubGrid folder
            os.chdir(os.path.join(folder_path, constituent))
            # Construct the command
            command = [executable, str(input_file)+'.json', 'colorcode']

            # Run the command
            result = subprocess.run(command, shell=False, text=True, capture_output=True)
            # Check for successful execution
            if result.returncode == 0:
                print("Execution completed successfully!")
            else:
                print(f"Error occurred:\n{result.stderr}")
            # return back to the original working directory
            os.chdir(cwd)
        except FileNotFoundError:
            print("Error: Executable not found. Please check the path.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print('run_gsfwriter function selected NOT to run')

# function to modify MODFLOW nam file with newly written input package & name for mp3du
def modify_nam_file_with_new_package_mp3du(flag, constituent, ptrk_calc_d, mf_inp_fnm, binary_print_ID, package_name, package_type):
    if flag:
        print('writing updated mf.nam file for mp3du purposes...')
        # read in name file & update with new package line
        with open(os.path.join(ptrk_calc_d, constituent, str(mf_inp_fnm)), 'r') as file:
            original_nam_lines = file.readlines()

        find_package = "ORT 23 P2Rv8.3.ort\n"
        add_package = str(package_type) + " " + str(binary_print_ID) + " " + str(package_name) + "\n"
        updated_nam_file_lines = []
        for line in original_nam_lines:
            updated_nam_file_lines.append(line)
            if line.strip() == find_package.strip():
                updated_nam_file_lines.append(add_package)

        # write the updated nam file back to the .nam file
        with open(os.path.join(ptrk_calc_d, constituent, str(mf_inp_fnm)+'.mp3du'), 'w') as file:
            file.writelines(updated_nam_file_lines)
    else:
        print('modify_nam_file_with_new_package Function selected to not run')

# this functions writes a starter p3d input file for mp3du
def write_p3d_mp3du(flag, constituent, ptrk_calc_d, p3d_fnm):
    if flag:
        print('writing initial p3d file for mp3du...')

        text = f"""# PATH3D Input File
        
CONSTANT               3        (FREE)        -1 VELOCITY METHOD LAYER 1
CONSTANT               3        (FREE)        -1 VELOCITY METHOD LAYER 2
CONSTANT               3        (FREE)        -1 VELOCITY METHOD LAYER 3
CONSTANT               3        (FREE)        -1 VELOCITY METHOD LAYER 4
CONSTANT               3        (FREE)        -1 VELOCITY METHOD LAYER 5
CONSTANT               3        (FREE)        -1 VELOCITY METHOD LAYER 6
CONSTANT               3        (FREE)        -1 VELOCITY METHOD LAYER 7
CONSTANT               0.15     (FREE)        -1 POROSITY L1
CONSTANT               0.15     (FREE)        -1 POROSITY L2
CONSTANT               0.15     (FREE)        -1 POROSITY L3
CONSTANT               0.15     (FREE)        -1 POROSITY L4
CONSTANT               0.15     (FREE)        -1 POROSITY L5
CONSTANT               0.15     (FREE)        -1 POROSITY L6
CONSTANT               0.15     (FREE)        -1 POROSITY L7
CONSTANT               1.0      (FREE)        -1 RETARDATION L1
CONSTANT               1.0      (FREE)        -1 RETARDATION L2
CONSTANT               1.0      (FREE)        -1 RETARDATION L3
CONSTANT               1.0      (FREE)        -1 RETARDATION L4
CONSTANT               1.0      (FREE)        -1 RETARDATION L5
CONSTANT               1.0      (FREE)        -1 RETARDATION L6
CONSTANT               1.0      (FREE)        -1 RETARDATION L7
OPEN/CLOSE  ../transport/SharedFiles/dsp1.ref    1.0     (FREE)  -1 DISPH L1
OPEN/CLOSE  ../transport/SharedFiles/dsp2.ref    1.0     (FREE)  -1 DISPH L2
OPEN/CLOSE  ../transport/SharedFiles/dsp3.ref    1.0     (FREE)  -1 DISPH L3
OPEN/CLOSE  ../transport/SharedFiles/dsp4.ref    1.0     (FREE)  -1 DISPH L4
OPEN/CLOSE  ../transport/SharedFiles/dsp5.ref    1.0     (FREE)  -1 DISPH L5
OPEN/CLOSE  ../transport/SharedFiles/dsp6.ref    1.0     (FREE)  -1 DISPH L6
OPEN/CLOSE  ../transport/SharedFiles/dsp7.ref    1.0     (FREE)  -1 DISPH L7
OPEN/CLOSE  ../transport/SharedFiles/dsp1.ref    0.2     (FREE)  -1 DISPT L1
OPEN/CLOSE  ../transport/SharedFiles/dsp2.ref    0.2     (FREE)  -1 DISPT L2
OPEN/CLOSE  ../transport/SharedFiles/dsp3.ref    0.2     (FREE)  -1 DISPT L3
OPEN/CLOSE  ../transport/SharedFiles/dsp4.ref    0.2     (FREE)  -1 DISPT L4
OPEN/CLOSE  ../transport/SharedFiles/dsp5.ref    0.2     (FREE)  -1 DISPT L5
OPEN/CLOSE  ../transport/SharedFiles/dsp6.ref    0.2     (FREE)  -1 DISPT L6
OPEN/CLOSE  ../transport/SharedFiles/dsp7.ref    0.2     (FREE)  -1 DISPT L7
OPEN/CLOSE  ../transport/SharedFiles/dsp1.ref    0.0     (FREE)  -1 DISPV L1
OPEN/CLOSE  ../transport/SharedFiles/dsp2.ref    0.0     (FREE)  -1 DISPV L2
OPEN/CLOSE  ../transport/SharedFiles/dsp3.ref    0.0     (FREE)  -1 DISPV L3
OPEN/CLOSE  ../transport/SharedFiles/dsp4.ref    0.0     (FREE)  -1 DISPV L4
OPEN/CLOSE  ../transport/SharedFiles/dsp5.ref    0.0     (FREE)  -1 DISPV L5
OPEN/CLOSE  ../transport/SharedFiles/dsp6.ref    0.0     (FREE)  -1 DISPV L6
OPEN/CLOSE  ../transport/SharedFiles/dsp7.ref    0.0     (FREE)  -1 DISPV L7

        """

        # write the p3d file to the folder
        with open(os.path.join(ptrk_calc_d, constituent, p3d_fnm), 'w') as file:
            file.writelines(text)
    else:
        print('write_p3d_mp3du function selected NOT to run...')

# this function generates particle starting locations from shapefile references
def generate_part_start_locs(flag, constituent, ptrk_calc_d, pstrt_fnm, gis_d):
    if flag: 
        print(f'generating particle starting locations for the {constituent} source areas...')

        # load in reference gis files for particle starting locations
        if constituent == 'cr':
            if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
                source_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX.shp'))
            else:
                source_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
        
        if constituent == 'tec-99':
            if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
                source_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX.shp'))
            else:
                source_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))
        
        # load in model shapefile
        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))
        model_grid_gdf['row_col'] = model_grid_gdf['row'].astype(int).astype(str) + '_' + model_grid_gdf['column'].astype(int).astype(str)
        
        # store model crs
        model_grid_crs = model_grid_gdf.crs

        # intersect source areas with model grid
        intersect = gpd.overlay(source_gdf, model_grid_gdf, how='intersection')
        
        # load in gsf file
        gsf_file = pd.read_csv(os.path.join(gis_d, 'xlsx', 'gsf', 'gsf_cells.csv'))
        gsf_file['P3D_CellID'] = (gsf_file['layer']-1)*(gsf_file['row'].max()*gsf_file['col'].max())+(gsf_file['row']-1)*gsf_file['col'].max()+gsf_file['col']
                
        # merge select row cols with row cols in gsf dataframe
        merged = pd.merge(intersect, gsf_file, on='row_col')
        # create point shapefile one particle per cell per layer to start
        prt_strt_locs_df = merged[['x_meters', 'y_meters', 'cellid','P3D_CellID', 'layer', 'row', 'col', 'lay_row_col', 'geometry']]
        prt_strt_locs_df['TIME_ATTR'] = 0 # release at beginning of model simulation
        prt_strt_locs_df['ZLOC_ATTR'] = 0.5 # relative z cell elevation for releasing particles

        # define coordinate locations
        prt_strt_locs_gdf = gpd.GeoDataFrame(prt_strt_locs_df, geometry='geometry', crs=model_grid_crs)
        prt_strt_locs_df_geo_cellcents = prt_strt_locs_gdf.geometry.centroid

        prt_strt_locs_df = prt_strt_locs_gdf[['x_meters', 'y_meters', 'cellid', 'layer', 'row', 'col', 'lay_row_col', 'P3D_CellID', 'TIME_ATTR', 'ZLOC_ATTR']]
        
        # only keep particle starting locations if in layers 3 or 4 or 5 or 7 for respective aquifer units
        lays_of_interest = [3, 4, 5, 7]
        print(prt_strt_locs_df['layer'].unique())
        prt_strt_locs_df = prt_strt_locs_df[prt_strt_locs_df['layer'].isin(lays_of_interest)]
        print(prt_strt_locs_df['layer'].unique())

        # export particle starting locations
        prt_strt_locs_gdf_points = gpd.GeoDataFrame(prt_strt_locs_df, geometry=prt_strt_locs_df_geo_cellcents, crs=model_grid_crs)
        prt_strt_locs_gdf_points.to_file(os.path.join(ptrk_calc_d, constituent, pstrt_fnm))

        # create rings of particles around centroids of source areas to use, if desired
        def generate_ring_points(center, radius, num_points):
            angle_disc = 360 / num_points
            ring_pts = []
            for angle in np.arange(0, 360, angle_disc):
                theta_rad = np.deg2rad(angle)
                dx = radius * np.cos(theta_rad)
                dy = radius * np.sin(theta_rad)
                new_pt = Point(center.x + dx, center.y + dy)
                ring_pts.append(new_pt)
            return ring_pts
        
        def create_ring_geodataframe(gdf, radius, num_points):
            all_prt_strt_ring_pts = []
            p3d_cellids = []
            for _, row in gdf.iterrows():
                center = row.geometry
                p3d_cellid = row['P3D_CellID']
                prt_strt_ring_pts = generate_ring_points(center, radius=radius, num_points=num_points)
                all_prt_strt_ring_pts.extend(prt_strt_ring_pts)
                p3d_cellids.extend([p3d_cellid]*len(prt_strt_ring_pts))
            ring_gdf = gpd.GeoDataFrame({'P3D_CellID': p3d_cellids, 'geometry': all_prt_strt_ring_pts}, crs=gdf.crs)

            return ring_gdf
        
        radius_m = 99.99/2 # cell discretization halved (meters)
        num_points = 36 
        ring_gdf = create_ring_geodataframe(prt_strt_locs_gdf_points, radius_m, num_points)  # Adjust radius based on your CRS
        ring_gdf['TIME_ATTR'] = 0 # release at beginning of model simulation
        ring_gdf['ZLOC_ATTR'] = 0.5 # relative z cell elevation for releasing particles
        
        # export to shapefile
        ring_gdf.to_file(os.path.join(ptrk_calc_d, constituent, pstrt_fnm+'_ring'))
        
    else:
        print('generate_part_start_locs function selected NOT to run...')

# this function writes the json input file for mp3du
def write_mp3du_json_input(flag, constituent, ptrk_calc_d, gsf_nm, flow_fnm, mp3du_json_nm, pstrt_fnm, pthlin_nm, part_type):
    if flag:
        print('creating gsf json...')
        if part_type == 'ring':
            pstrt_fpth = 'particle_starting_locations_ring/'+pstrt_fnm+'_ring.shp'
        else:
            pstrt_fpth = 'particle_starting_locations/'+pstrt_fnm+'.shp'
        data = {
            "FLOW_MODEL_TYPE": {
                "MODFLOW": {
                    "NAME_FILE": flow_fnm+'.mp3du',
                    "GSF_FILE": {
                        "TYPE": "GSF_V.1.1.0",
                        "FILE_NAME": gsf_nm
                    },
                    "OUTPUT_PRECISION": "DOUBLE",
                    "IFACE": [ 
                        { "MNW2": 0 }, 
                        { "RIV": 2 },
                        { "CHD": 2 }, 
                        { "RCH": 6 }
                    ],
                    "THREAD_COUNT": 10
                }
            },
            "SIMULATIONS": [
                {
                    "ENDPOINT": {
                        "NAME": constituent+"",
                        "DIRECTION": "FORWARD",
                        "THREAD_COUNT": 10,
                        "INITIAL_STEPSIZE": 0.1,
                        "MAX_STEPSIZE": 1.0e6,
                        "STAGNATION_DT": 1.0e-15,
                        "EULER_DT": 1.0e-4,
                        "ADAPTIVE_STEP_ERROR": 1.0e-6,
                        "SIMULATION_END_TIME" : 8309.0,
                        "CAPTURE_RADIUS": 10.0,
                        "OPTIONS": [
                            "DISPERSION",
                            "RETARDATION",
                            #"TERMINATION"
                        ],
                        "PARTICLE_START_LOCATIONS": {
                            "REPEAT": 1,
                            "REPEAT_DT": 10000,
                            "SHAPEFILE": {
                                "FILE_NAME": pstrt_fpth,
                                "CELLID_ATTR": "P3D_CellID",
                                "TIME_ATTR": "TIME_ATTR",
                                "ZLOC_ATTR": "ZLOC_ATTR",
                            }
                        }
                    }
                },
                {
                    "PATHLINE": {
                        "NAME": constituent+"",
                        "DIRECTION": "FORWARD",
                        "THREAD_COUNT": 10,
                        "INITIAL_STEPSIZE": 0.1,
                        "MAX_STEPSIZE": 1.0e6,
                        "STAGNATION_DT": 1.0e-15,
                        "EULER_DT": 1.0e-4,
                        "ADAPTIVE_STEP_ERROR": 1.0e-6,
                        "SIMULATION_END_TIME" : 8309.0,
                        "CAPTURE_RADIUS": 10.0,
                        "OPTIONS": [
                            "DISPERSION",
                            "RETARDATION",
                            #"TERMINATION"
                        ],
                        "PARTICLE_START_LOCATIONS": {
                            "REPEAT": 1,
                            "REPEAT_DT": 10000,
                            "SHAPEFILE": {
                                "FILE_NAME": pstrt_fpth,
                                "CELLID_ATTR": "P3D_CellID",
                                "TIME_ATTR": "TIME_ATTR",
                                "ZLOC_ATTR": "ZLOC_ATTR"
                            }
                        }
                    }
                }
            ]
        }

        text = json.dumps(data, indent=4)
        ### output phreeqc input file
        with open(os.path.join(ptrk_calc_d, constituent, str(mp3du_json_nm)+".json"), "w") as file:
            file.write(text)
    else:
        print('write_gsf_json_input Function selected NOT to run')

# this function runs modpath3du particle tracking analyses
def run_mp3du(flag, constituent, folder_path, executable, input_file):
    if flag:
        print('running mp3du.exe for source areas: ' + constituent + '...')
        try:
            # store current directory
            cwd = os.getcwd()
            # change working directory to CreateSubGrid folder
            os.chdir(os.path.join(folder_path, constituent))
            # Construct the command
            command = [executable, input_file, 'colorcode']

            # Run the command
            result = subprocess.run(
                command,
                shell=False,
                text=True,  # Use text mode to get strings instead of bytes
                capture_output=True,
                encoding='utf-8',
                errors='replace'  # This avoids decoding crashes
            )

            # Return to original directory
            os.chdir(cwd)

            ## Check for successful execution
            if result.returncode == 0:
                print("Execution completed successfully!")
                #if result.stdout:
                #    print("Output:", result.stdout.strip())
            else:
                print("Execution failed with return code:", result.returncode)
                print("STDERR:", result.stderr.strip())
        except FileNotFoundError:
            print("Error: Executable not found. Please check the path.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    else:
        print('run_mp3du function selected NOT to run')

# this function generates a json file that goes into writep3d output files from bin mp3du outputs for pathlines
def write_p3doutput_json_input_path(flag, constituent, ptrk_calc_d, p3doutput_json_nm):
    if flag:
        print(f'creating p3doutput json for {constituent} for pathlines...')

        data = {
            "MP3DU_BIN": constituent+"_PATHLINE.bin",
            "OUTPUTS": [
                {
                    "SUMMARY": {}
                },
                {
                    "ASCII_TABLE": {
                        "FILE_NAME": constituent+"_pathlines.dat"
                    }
                },
                {
                    "DBF_TABLE": {
                        "FILE_NAME": constituent+"_pathlines.dbf"
                    }
                },
                {
                    "PATHLINE_WHOLE": {
                        "FILE_NAME": constituent+"_pathlines.shp",
                        "MAX_TIME": 400000,
                        "MIN_TIME": 0
                    }
                }
            ]
        }

        # Convert to JSON-formatted string
        text = json.dumps(data, indent=4)

        ### output phreeqc input file
        with open(os.path.join(ptrk_calc_d, constituent, str(p3doutput_json_nm)+".json"), "w") as file:
            file.write(text)
    else:
        print('write_gsf_json_input_path Function selected NOT to run')

# this function generates a json file that goes into writep3d output files from bin mp3du outputs for endpoints
def write_p3doutput_json_input_endpts(flag, constituent, ptrk_calc_d, p3doutput_json_nm):
    if flag:
        print(f'creating p3doutput json for {constituent} endpoints...')

        data = {
            "MP3DU_BIN": constituent+"_ENDPOINT.bin",
            "OUTPUTS": [
                {
                    "SUMMARY": {}
                },
                {
                    "ASCII_TABLE": {
                        "FILE_NAME": constituent+"_endpoints.dat"
                    }
                },
                {
                    "DBF_TABLE": {
                        "FILE_NAME": constituent+"_endpoints.dbf"
                    }
                },
                {
                    "ENDPOINT": {
                        "FILE_NAME": constituent+"_endpoints.shp",
                    }
                }
            ]
        }

        # Convert to JSON-formatted string
        text = json.dumps(data, indent=4)

        ### output phreeqc input file
        with open(os.path.join(ptrk_calc_d, constituent, str(p3doutput_json_nm)+".json"), "w") as file:
            file.write(text)
    else:
        print('write_gsf_json_input_endpts Function selected NOT to run')

# this function runs writep3doutput post-processing of particle-tracking results
def run_writep3doutput(flag, constituent, folder_path, executable, input_file):
    if flag:
        print('running writep3doutput.exe for source areas: ' + constituent + '...')
        try:
            # store current directory
            cwd = os.getcwd()
            # change working directory to CreateSubGrid folder
            os.chdir(os.path.join(folder_path, constituent))
            # Construct the command
            command = [executable, input_file, 'colorcode']

            # Run the command
            result = subprocess.run(command, shell=False, text=True, capture_output=True)
            # Check for successful execution
            if result.returncode == 0:
                print("Execution completed successfully!")
            else:
                print(f"Error occurred:\n{result.stderr}")
            # return back to the original working directory
            os.chdir(cwd)
        except FileNotFoundError:
            print("Error: Executable not found. Please check the path.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print('run_writep3doutput function selected NOT to run')

# this function generates a pathlines map of mp3du particle tracking results
def generate_pathlines_map(flag, gis_d, fig_d, ptrk_calc_d, part_type):
    if flag: 
        print('generating a pathlines map of particle tracking mp3du results...')
        
        # create output directory for figs
        if not os.path.exists(fig_d):
            os.makedirs(fig_d)
        
        # load in reference gis files for particle starting locations
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX.shp'))
        else:
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))
        
        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))
        
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))
        
        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))

        # load in particle starting locations and pathlines for each source area type
        if part_type == 'ring':
            cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp'))
            tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp')) 
        else:
            cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations', 'particle_starting_locations.shp'))
            tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations', 'particle_starting_locations.shp'))
        
        cr_pathlines = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'cr_pathlines.shp'))
        tec99_pathlines = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'tec-99_pathlines.shp'))

        # check model grid crs
        mdgrd_crs = model_grid_gdf.crs
        crsrc_crs = cr_source_gpf.crs

        if not mdgrd_crs == crsrc_crs:
            print('updating the crs of the model grid for consistency...')
            model_grid_gdf.to_crs(crsrc_crs)
        else:
            print('the crs of the model grid is not being updated for consistency...')

        # define crs of particle pathlines
        cr_pathlines = cr_pathlines.set_crs(mdgrd_crs)
        tec99_pathlines = tec99_pathlines.set_crs(mdgrd_crs)

        # load in data gap locations
        data_gap_locs = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))

        # load in HWIS Data Pull 2025
        hwis_data_df = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93_reduced.csv'))
        
        # filter by status of entry
        status_list = ['IN-USE', 'CANDIDATE FOR DECOMMISSIONING', 'AWAITING DRILLING']
        hwis_data_df_red = hwis_data_df[hwis_data_df['STATUS'].isin(status_list)]

        # create geopandas dataframe object and export to shapefile
        hwis_data_df_red_geometry = [Point(xy) for xy in zip(hwis_data_df_red['EASTING'], hwis_data_df_red['NORTHING'])]
        hwis_data_gdf_red = gpd.GeoDataFrame(hwis_data_df_red, geometry=hwis_data_df_red_geometry, crs=mdgrd_crs)
        hwis_data_gdf_red.to_file(os.path.join(gis_d, 'shp', 'misc', 'hwis_data_gdf_reduced.shp'))

        # load in existing candidate monitoring wells and extraction wells
        existing_candidate_wells_mw_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'existing_candidate_mws.shp'))
        existing_candidate_wells_ew_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'existing_candidate_ews.shp'))

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)

        cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
        te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
        wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
        
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA S & WMA SX-SY')
        else:
            wma_T_wma_txty_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA T & WMA TX-TY')

        model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
        cr_pathlines.plot(ax=ax, linewidth=0.2, color='chocolate', zorder=1, alpha=0.5, label='cr mp3du pathlines')
        tec99_pathlines.plot(ax=ax, linewidth=0.2, color='fuchsia', zorder=1, alpha=0.5, label='tec-99 mp3du pathlines')
        cr_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='chocolate', zorder=1, alpha=1, label='cr particle starting locations')
        tec99_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='fuchsia', zorder=1, alpha=1, label='tec-99 particle starting locations')
        data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
        hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
        existing_candidate_wells_ew_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells')
        existing_candidate_wells_mw_gdf.plot(ax=ax, facecolor='lightgreen', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells')

        # manually define legend items
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA S & WMA SX-SY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], color='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du pathlines'),
                Line2D([0], [0], color='fuchsia', linewidth=0.2, alpha=0.5, label='tec-99 mp3du pathlines'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='cr particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='tec-99 particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells'),
                Line2D([0], [0], marker='o', markerfacecolor='lightgreen', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells')
            ]
        else:
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], color='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du pathlines'),
                Line2D([0], [0], color='fuchsia', linewidth=0.2, alpha=0.5, label='tec-99 mp3du pathlines'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='cr particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='tec-99 particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells'),
                Line2D([0], [0], marker='o', markerfacecolor='lightgreen', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells')
            ]

        ax.legend(handles=legend_elements, loc='upper right')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([133300+y_axis_offset, 135250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        else:
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])

        plt.title('Pathlines from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'mp3du_pathlines_map_rings_ssxsy.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'mp3du_pathlines_map_centroids_ssxsy.png'), dpi=400)

        else:
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'mp3du_pathlines_map_rings.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'mp3du_pathlines_map_centroids.png'), dpi=400)

        plt.show()
        plt.close()

    else:
        print('generate_pathlines_map function selected NOT to run...')

# this function generates an endpoints map of mp3du particle tracking results
def generate_endpoints_map(flag, gis_d, fig_d, ptrk_calc_d, part_type):
    if flag: 
        print('generating an endpoints map of particle tracking mp3du results...')
        
        # create output directory for figs
        if not os.path.exists(fig_d):
            os.makedirs(fig_d)
        
        # load in reference gis files for particle starting locations
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX.shp'))
        else:
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))

        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))

        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))

        # load in particle starting locations and endpoints for each source area type
        if part_type == 'ring':
            cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp'))
            tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp')) 
        else:
            cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations', 'particle_starting_locations.shp'))
            tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations', 'particle_starting_locations.shp'))
        
        cr_endpoints = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'cr_endpoints.shp'))
        tec99_endpoints = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'tec-99_endpoints.shp'))

        # check model grid crs
        mdgrd_crs = model_grid_gdf.crs
        crsrc_crs = cr_source_gpf.crs

        if not mdgrd_crs == crsrc_crs:
            print('updating the crs of the model grid for consistency...')
            model_grid_gdf.to_crs(crsrc_crs)
        else:
            print('the crs of the model grid is not being updated for consistency...')

        # define crs of particle endpoints
        cr_endpoints = cr_endpoints.set_crs(mdgrd_crs)
        tec99_endpoints = tec99_endpoints.set_crs(mdgrd_crs)

        # load in data gap locations
        data_gap_locs = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))

        # load in HWIS Data Pull 2025
        hwis_data_df = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93_reduced.csv'))
        
        # filter by status of entry
        status_list = ['IN-USE', 'CANDIDATE FOR DECOMMISSIONING', 'AWAITING DRILLING']
        hwis_data_df_red = hwis_data_df[hwis_data_df['STATUS'].isin(status_list)]

        # create geopandas dataframe object and export to shapefile
        hwis_data_df_red_geometry = [Point(xy) for xy in zip(hwis_data_df_red['EASTING'], hwis_data_df_red['NORTHING'])]
        hwis_data_gdf_red = gpd.GeoDataFrame(hwis_data_df_red, geometry=hwis_data_df_red_geometry, crs=mdgrd_crs)
        hwis_data_gdf_red.to_file(os.path.join(gis_d, 'shp', 'misc', 'hwis_data_gdf_reduced.shp'))

        # load in existing candidate monitoring wells and extraction wells
        existing_candidate_wells_mw_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'existing_candidate_mws.shp'))
        existing_candidate_wells_ew_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'existing_candidate_ews.shp'))

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)

        cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
        te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
        wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
        
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA S & WMA SX-SY')
        else:
            wma_T_wma_txty_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA T & WMA TX-TY')
        
        model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
        cr_endpoints.plot(ax=ax, edgecolor='chocolate', zorder=1, facecolor='None', alpha=0.5, label='cr mp3du endpoints')
        tec99_endpoints.plot(ax=ax, edgecolor='fuchsia', zorder=1, facecolor='None', alpha=0.5, label='tec-99 mp3du endpoints')
        cr_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='chocolate', zorder=1, alpha=1, label='cr particle starting locations')
        tec99_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='fuchsia', zorder=1, alpha=1, label='tec-99 particle starting locations')
        data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
        hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
        existing_candidate_wells_ew_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells')
        existing_candidate_wells_mw_gdf.plot(ax=ax, facecolor='lightgreen', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells')

        # manually define legend items
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA S & WMA SX-SY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du endpoints'),
                Patch(facecolor='None', edgecolor='fuchsia', linewidth=0.2, alpha=0.5, label='tec-99 mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='cr particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='tec-99 particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells'),
                Line2D([0], [0], marker='o', markerfacecolor='lightgreen', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells')
            ]

        else:
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du endpoints'),
                Patch(facecolor='None', edgecolor='fuchsia', linewidth=0.2, alpha=0.5, label='tec-99 mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='cr particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='tec-99 particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells'),
                Line2D([0], [0], marker='o', markerfacecolor='lightgreen', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells')
            ]

        ax.legend(handles=legend_elements, loc='upper right')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([133300+y_axis_offset, 135250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        else:
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])

        plt.title('Endpoints from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'mp3du_endpoints_map_rings_ssxsy.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'mp3du_endpoints_map_centroids_ssxsy.png'), dpi=400)
        else:
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'mp3du_endpoints_map_rings.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'mp3du_endpoints_map_centroids.png'), dpi=400)

        plt.show()
        plt.close()

    else:
        print('generate_endpoints_map function selected NOT to run...')

# this function generates a pathlines and endpoints map of mp3du particle tracking results
def generate_pathlines_endpoints_map(flag, gis_d, fig_d, ptrk_calc_d, part_type):
    if flag: 
        print('generating a pathlines and endpoints map of particle tracking mp3du results...')
        
        # create output directory for figs
        if not os.path.exists(fig_d):
            os.makedirs(fig_d)
        
        # load in reference gis files for particle starting locations
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX.shp'))
        else:
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))

        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))

        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))

        # load in particle starting locations and endpoints for each source area type
        if part_type == 'ring':
            cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp'))
            tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp')) 
        else:
            cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations', 'particle_starting_locations.shp'))
            tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations', 'particle_starting_locations.shp'))
        
        cr_endpoints = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'cr_endpoints.shp'))
        tec99_endpoints = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'tec-99_endpoints.shp'))

        cr_pathlines = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'cr_pathlines.shp'))
        tec99_pathlines = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'tec-99_pathlines.shp'))

        # check model grid crs
        mdgrd_crs = model_grid_gdf.crs
        crsrc_crs = cr_source_gpf.crs

        if not mdgrd_crs == crsrc_crs:
            print('updating the crs of the model grid for consistency...')
            model_grid_gdf.to_crs(crsrc_crs)
        else:
            print('the crs of the model grid is not being updated for consistency...')

        # define crs of particle pathlines
        cr_pathlines = cr_pathlines.set_crs(mdgrd_crs)
        tec99_pathlines = tec99_pathlines.set_crs(mdgrd_crs)

        # define crs of particle endpoints
        cr_endpoints = cr_endpoints.set_crs(mdgrd_crs)
        tec99_endpoints = tec99_endpoints.set_crs(mdgrd_crs)

        # load in data gap locations
        data_gap_locs = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))

        # load in HWIS Data Pull 2025
        hwis_data_df = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93_reduced.csv'))
        
        # filter by status of entry
        status_list = ['IN-USE', 'CANDIDATE FOR DECOMMISSIONING', 'AWAITING DRILLING']
        hwis_data_df_red = hwis_data_df[hwis_data_df['STATUS'].isin(status_list)]

        # create geopandas dataframe object and export to shapefile
        hwis_data_df_red_geometry = [Point(xy) for xy in zip(hwis_data_df_red['EASTING'], hwis_data_df_red['NORTHING'])]
        hwis_data_gdf_red = gpd.GeoDataFrame(hwis_data_df_red, geometry=hwis_data_df_red_geometry, crs=mdgrd_crs)
        hwis_data_gdf_red.to_file(os.path.join(gis_d, 'shp', 'misc', 'hwis_data_gdf_reduced.shp'))

        # load in existing candidate monitoring wells and extraction wells
        existing_candidate_wells_mw_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'existing_candidate_mws.shp'))
        existing_candidate_wells_ew_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'existing_candidate_ews.shp'))

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)

        cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
        te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
        wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA S & WMA SX-SY')
        else:
            wma_T_wma_txty_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA T & WMA TX-TY')

        model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
        cr_endpoints.plot(ax=ax, edgecolor='chocolate', zorder=1, facecolor='None', alpha=0.5, label='cr mp3du endpoints')
        tec99_endpoints.plot(ax=ax, edgecolor='fuchsia', zorder=1, facecolor='None', alpha=0.5, label='tec-99 mp3du endpoints')
        cr_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='chocolate', zorder=1, alpha=1, label='cr particle starting locations')
        tec99_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='fuchsia', zorder=1, alpha=1, label='tec-99 particle starting locations')
        data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
        hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
        cr_pathlines.plot(ax=ax, linewidth=0.2, color='chocolate', zorder=1, alpha=0.5, label='cr mp3du pathlines')
        tec99_pathlines.plot(ax=ax, linewidth=0.2, color='fuchsia', zorder=1, alpha=0.5, label='tec-99 mp3du pathlines')
        existing_candidate_wells_ew_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells')
        existing_candidate_wells_mw_gdf.plot(ax=ax, facecolor='lightgreen', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells')

        # manually define legend items
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA S & WMA SX-SY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du endpoints'),
                Patch(facecolor='None', edgecolor='fuchsia', linewidth=0.2, alpha=0.5, label='tec-99 mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='cr particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='tec-99 particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], color='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du pathlines'),
                Line2D([0], [0], color='fuchsia', linewidth=0.2, alpha=0.5, label='tec-99 mp3du pathlines'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells'),
                Line2D([0], [0], marker='o', markerfacecolor='lightgreen', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells')
            ]
        else:
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du endpoints'),
                Patch(facecolor='None', edgecolor='fuchsia', linewidth=0.2, alpha=0.5, label='tec-99 mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='cr particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='tec-99 particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], color='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du pathlines'),
                Line2D([0], [0], color='fuchsia', linewidth=0.2, alpha=0.5, label='tec-99 mp3du pathlines'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells'),
                Line2D([0], [0], marker='o', markerfacecolor='lightgreen', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells')
            ]

        ax.legend(handles=legend_elements, loc='upper right')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([133300+y_axis_offset, 135250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        else:
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])

        plt.title('Pathlines & Endpoints from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'mp3du_pathlines_endpoints_map_rings_ssxsy.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'mp3du_pathlines_endpoints_map_centroids_ssxsy.png'), dpi=400)
        else:
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'mp3du_pathlines_endpoints_map_rings.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'mp3du_pathlines_endpoints_map_centroids.png'), dpi=400)

        plt.show()

    else:
        print('generate_pathlines_endpoints_map function selected NOT to run...')

# this function calculates the relative pathline count per model cell for the data gap assessment
def calc_relative_path_count(flag, gis_d, ptrk_calc_d):
    if flag:
        print('calculating relative path count per data gap model cell locations...')

        # create outfpth
        out_fpth = os.path.join(gis_d, 'shp', 'pathline_count')
        if not os.path.exists(out_fpth):
            os.makedirs(out_fpth)

        # load in model grid 
        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))
        model_grid_gdf['row_col'] = model_grid_gdf['row'].astype(int).astype(str) + '_' + model_grid_gdf['column'].astype(int).astype(str)
        model_grid_gdf['ID'] = model_grid_gdf['row_col']

        # store model grid crs
        mdgrd_crs = model_grid_gdf.crs

        # load in cr source area & tec-99 source area pathlines
        cr_pathlines = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'cr_pathlines.shp'))
        tec99_pathlines = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'tec-99_pathlines.shp'))

        # define crs of particle pathlines
        cr_pathlines = cr_pathlines.set_crs(mdgrd_crs)
        tec99_pathlines = tec99_pathlines.set_crs(mdgrd_crs)

        # combine both source area pathlines
        pathlines_gdf = gpd.GeoDataFrame(pd.concat([cr_pathlines, tec99_pathlines], ignore_index=True), crs=mdgrd_crs)

        # export combined source area pathlines
        pathlines_fpth = os.path.join(gis_d, 'shp', 'pathlines')
        if not os.path.exists(pathlines_fpth):
            os.makedirs(pathlines_fpth)
        
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            fnm_pathlines_gdf = 'pathlines_both_source_areas_ssxsy.shp'
        else:
            fnm_pathlines_gdf = 'pathlines_both_source_areas.shp'
        pathlines_gdf.to_file(os.path.join(pathlines_fpth, fnm_pathlines_gdf)) 

        # spatial join combined source area pathlines with data gap model cell locations
        mdgrd_sjoin = gpd.sjoin(model_grid_gdf, pathlines_gdf, how='left', predicate='intersects')
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            mdgrd_sjoin.to_file(os.path.join(out_fpth, 'mdgrd_sjoin_ssxsy.shp'))
        else:
            mdgrd_sjoin.to_file(os.path.join(out_fpth, 'mdgrd_sjoin.shp'))

        # count the total number of pathlines intersecting each model cell
        join_counts = mdgrd_sjoin.groupby("ID")["PID"].nunique().reset_index()

        # Rename column to 'join_count'
        join_counts.rename(columns={"PID": "pid_count"}, inplace=True)
        
        # Merge the count back into the original GeoDataFrame
        mdgrd_sjoin = mdgrd_sjoin.merge(join_counts, on="ID", how="left")
        
        # Fill NaN values with 0 (if there are any missing values)
        mdgrd_sjoin["path_count"] = mdgrd_sjoin["pid_count"].fillna(0).astype(int)
        
        # Save the updated shapefile
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            mdgrd_sjoin.to_file(os.path.join(out_fpth, "mdgrd_sjoin_with_counts_ssxsy.shp"))
        else:
            mdgrd_sjoin.to_file(os.path.join(out_fpth, "mdgrd_sjoin_with_counts.shp"))

        # calculate the relative path count per data gap model cell location & normalize to the max, keep unique
        unique_mdgrd_gdf_sjoin = mdgrd_sjoin.drop_duplicates(subset=["ID"]).reset_index(drop=True)

        print('calculating relative detectability...')
        # calculate relative detectability (rd) by ECF-200PO1-21-0021 Section 6.2 -> rd (decimal %) = N/MNP
        unique_mdgrd_gdf_sjoin['rel_det'] = unique_mdgrd_gdf_sjoin['path_count'] / unique_mdgrd_gdf_sjoin['path_count'].max()

        # export the relative normalization to a shapefile
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            unique_mdgrd_gdf_sjoin.to_file(os.path.join(out_fpth, 'mp3du_relative_detectability_ssxsy.shp'))
        else:
            unique_mdgrd_gdf_sjoin.to_file(os.path.join(out_fpth, 'mp3du_relative_detectability.shp'))

    else:
        print('calc_relative_path_count function selected NOT to run...')

# this function generates a relative count of pathlines mapped to the model grid cells for the data gap assessment using the mp3du particle tracking results
def generate_relcount_pathlines_map(flag, gis_d, fig_d, ptrk_calc_d, part_type):
    if flag: 
        print('generating a relative count of pathlines mapped to model grid cells for the data gap assessment...')
        
        # create output directory for figs
        if not os.path.exists(fig_d):
            os.makedirs(fig_d)

        # Define the color bands and corresponding colors
        color_bands = [(0.0000000001, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                       (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0000000001)]
        colors = ['midnightblue', 'royalblue', 'cornflowerblue', 'lightsteelblue', 'lightyellow', 'gold', 'yellow',
                  'darkorange', 'orangered', 'red']
        
        # load in reference gis files for particle starting locations
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX.shp'))
        else:
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))
            
        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))
        
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))
        
        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))

        # load in particle starting locations and pathlines for each source area type
        if part_type == 'ring':
            cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp'))
            tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp')) 
        else:
            cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations', 'particle_starting_locations.shp'))
            tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations', 'particle_starting_locations.shp'))

        # check model grid crs
        mdgrd_crs = model_grid_gdf.crs
        crsrc_crs = cr_source_gpf.crs

        if not mdgrd_crs == crsrc_crs:
            print('updating the crs of the model grid for consistency...')
            model_grid_gdf.to_crs(crsrc_crs)
        else:
            print('the crs of the model grid is not being updated for consistency...')

        # load in relative pathline shapefile here
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            relcount_path_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'pathline_count', 'mp3du_relative_detectability_ssxsy.shp'))
        else:
            relcount_path_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'pathline_count', 'mp3du_relative_detectability.shp'))
                                          
        # load in data gap locations
        data_gap_locs = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))

        # load in HWIS Data Pull 2025
        hwis_data_gdf_red = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'hwis_data_gdf_reduced.shp'))

        # load in existing candidate monitoring wells and extraction wells
        existing_candidate_wells_mw_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'existing_candidate_mws.shp'))
        existing_candidate_wells_ew_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'existing_candidate_ews.shp'))

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)

        cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
        te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
        wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA S & WMA SX-SY')
        else:
            wma_T_wma_txty_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA T & WMA TX-TY')
        
        model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
        cr_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='chocolate', zorder=1, alpha=1, label='cr particle starting locations')
        tec99_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='fuchsia', zorder=1, alpha=1, label='tec-99 particle starting locations')
        data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
        hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
        existing_candidate_wells_ew_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells')
        existing_candidate_wells_mw_gdf.plot(ax=ax, facecolor='lightgreen', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells')

        colorflood_legend_elements = []
        # plot relative pathline count shapefile here
        for (low, high), color in zip(color_bands, colors):
            subset = relcount_path_gdf[(relcount_path_gdf['rel_det'] >= low) & (relcount_path_gdf['rel_det'] < high)]
            subset.plot(ax=ax, color=color, alpha=0.75, label=f'{low}-{high}')
            # Add corresponding legend patch
            colorflood_legend_elements.append(Patch(facecolor=color, edgecolor='black', alpha=0.75, label=f'{low}-{high}'))


        # manually define legend items
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA S & WMA SX-SY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='cr particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='tec-99 particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells'),
                Line2D([0], [0], marker='o', markerfacecolor='lightgreen', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells')
            ]

        else:
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='cr particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='tec-99 particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells'),
                Line2D([0], [0], marker='o', markerfacecolor='lightgreen', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells')
            ]

        combined_legend_elements = legend_elements + colorflood_legend_elements

        ax.legend(handles=combined_legend_elements, loc='upper right')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([133300+y_axis_offset, 135250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        else:
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])

        plt.title('Color Flood of Relative Pathline Counts from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'mp3du_relative_count_pathlines_map_rings_ssxsy.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'mp3du_relative_count_pathlines_map_centroids_ssxsy.png'), dpi=400)
        else:
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'mp3du_relative_count_pathlines_map_rings.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'mp3du_relative_count_pathlines_map_centroids.png'), dpi=400)

        plt.show()
        plt.close()

    else:
        print('generate_relcount_pathlines_map function selected NOT to run...')

# this function parses out each individual continuing source zone
def parse_source_zones(flag, gis_d, fig_d, ptrk_calc_d, part_type):
    if flag:
        print('parsing out individual continuous source zones for cr & tec99 into shps...')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            # parse cr continuing source zones 
            cr_source_zones = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX.shp'))
            
            cr_source_zone_north = cr_source_zones.iloc[[0]]
            cr_source_zone_north.to_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX_North.shp'))

            cr_source_zone_south = cr_source_zones.iloc[[1]]
            cr_source_zone_south.to_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX_South.shp'))

            # parse tec99 continuing source zones
            tec99_source_zones = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX.shp'))

            tec99_source_zone_north = tec99_source_zones.iloc[[0]]
            tec99_source_zone_north.to_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX_North.shp'))

        else:

            # parse cr continuing source zones 
            cr_source_zones = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
            
            cr_source_zone_north = cr_source_zones.iloc[[0]]
            cr_source_zone_north.to_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_North.shp'))

            cr_source_zone_south = cr_source_zones.iloc[[1]]
            cr_source_zone_south.to_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_South.shp'))

            # parse tec99 continuing source zones
            tec99_source_zones = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))

            tec99_source_zone_north = tec99_source_zones.iloc[[0]]
            tec99_source_zone_north.to_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_North.shp'))

            tec99_source_zone_south = tec99_source_zones.iloc[[1]]
            tec99_source_zone_south.to_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_South.shp'))

    else:
        print('parse_source_zones function selected NOT to run...')

# this function generates polygon bounding map of mp3du particle tracking results for continuing source areas
def generate_bounding_polygon(flag, gis_d, fig_d, ptrk_calc_d, part_type, source_area):
    if flag: 
        print(f'generating {source_area} bounding polygons of particle tracking mp3du results...')
        
        # create output directory for figs
        if not os.path.exists(fig_d):
            os.makedirs(fig_d)
        
        # load in reference gis files for particle starting locations
        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))
        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))

        # load in particle starting locations, pathlines, and endpoints for each source area type
        if source_area == 'Chromium_Source_North' or source_area == 'Chromium_Source_South' or source_area == 'Chromium_Source_S-SX_North' or source_area == 'Chromium_Source_S-SX_South':
            source_area_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', source_area+'.shp'))
            if part_type == 'ring':
                cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp'))
            else:
                cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations', 'particle_starting_locations.shp'))

            cr_endpoints = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'cr_endpoints.shp'))

            cr_pathlines = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'cr_pathlines.shp'))

            part_starts = gpd.sjoin(cr_part_starts, source_area_gdf, how = 'inner', predicate='intersects')
            part_pathlines = gpd.sjoin(cr_pathlines, source_area_gdf,how = 'inner', predicate='intersects')
            part_endpoints = cr_endpoints

            # load in source area polygon
            if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
                source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX.shp'))
            else:
                source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
        
        if source_area == 'Technetium_Source_North' or source_area == 'Technetium_Source_South' or source_area == 'Technetium_Source_S-SX_North' or source_area == 'Technetium_Source_S-SX_South':
            source_area_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', source_area+'.shp'))
            if part_type == 'ring':
                tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp')) 
            else:
                tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations', 'particle_starting_locations.shp'))
            
            tec99_endpoints = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'tec-99_endpoints.shp'))

            tec99_pathlines = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'tec-99_pathlines.shp'))
        
            part_starts = gpd.sjoin(tec99_part_starts, source_area_gdf, how = 'inner', predicate='intersects')
            part_pathlines = gpd.sjoin(tec99_pathlines, source_area_gdf,how = 'inner', predicate='intersects')
            part_endpoints = tec99_endpoints

            # load in source area polygon
            if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
                source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX.shp'))
            else:
                source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))

        # check model grid crs
        mdgrd_crs = model_grid_gdf.crs
        src_crs = source_gpf.crs

        if not mdgrd_crs == src_crs:
            print('updating the crs of the model grid for consistency...')
            model_grid_gdf.to_crs(src_crs)
        else:
            print('the crs of the model grid is not being updated for consistency...')

        # define crs of particle pathlines
        part_pathlines = part_pathlines.set_crs(mdgrd_crs)

        # define crs of particle endpoints
        part_endpoints = part_endpoints.set_crs(mdgrd_crs)

        # load in data gap locations
        data_gap_locs = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))

        # load in HWIS Data Pull 2025
        hwis_data_df = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93_reduced.csv'))
        
        # filter by status of entry
        status_list = ['IN-USE', 'CANDIDATE FOR DECOMMISSIONING', 'AWAITING DRILLING']
        hwis_data_df_red = hwis_data_df[hwis_data_df['STATUS'].isin(status_list)]

        # create geopandas dataframe object and export to shapefile
        hwis_data_df_red_geometry = [Point(xy) for xy in zip(hwis_data_df_red['EASTING'], hwis_data_df_red['NORTHING'])]
        hwis_data_gdf_red = gpd.GeoDataFrame(hwis_data_df_red, geometry=hwis_data_df_red_geometry, crs=mdgrd_crs)
        hwis_data_gdf_red.to_file(os.path.join(gis_d, 'shp', 'misc', 'hwis_data_gdf_reduced.shp'))

        # load in candidate monitoring wells and extraction wells
        candidate_wells_mw_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_uu_mu.shp'))
        candidate_wells_ew_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_uu_mu.shp'))
        candidate_wells_mw_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_lu_cr.shp'))
        candidate_wells_ew_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_lu_cr.shp'))

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)
        if source_area == 'Chromium_Source_North' or source_area == 'Chromium_Source_South' or source_area == 'Chromium_Source_S-SX_North' or source_area == 'Chromium_Source_S-SX_South':
            source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='cr source zones')
        if source_area == 'Technetium_Source_North' or source_area == 'Technetium_Source_South' or source_area == 'Technetium_Source_S-SX_North' or source_area == 'Technetium_Source_S-SX_South':
            source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
        wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))        
        model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
        if source_area == 'Chromium_Source_North' or source_area == 'Chromium_Source_South' or source_area == 'Chromium_Source_S-SX_North' or source_area == 'Chromium_Source_S-SX_South':
            part_endpoints.plot(ax=ax, edgecolor='chocolate', markersize=4, zorder=2, facecolor='None', alpha=0.5, label='cr mp3du endpoints')
            part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='chocolate', zorder=1, alpha=1, label='cr particle starting locations')
        if source_area == 'Technetium_Source_North' or source_area == 'Technetium_Source_South' or source_area == 'Technetium_Source_S-SX_North' or source_area == 'Technetium_Source_S-SX_South':
            part_endpoints.plot(ax=ax, edgecolor='fuchsia', markersize=4, zorder=2, facecolor='None', alpha=0.5, label='tec99 mp3du endpoints')
            part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='fuchsia', zorder=3, alpha=1, label='tec99 particle starting locations')
        data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
        hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
        if source_area == 'Chromium_Source_North' or source_area == 'Chromium_Source_South' or source_area == 'Chromium_Source_S-SX_North' or source_area == 'Chromium_Source_S-SX_South':
            part_pathlines.plot(ax=ax, linewidth=0.2, color='chocolate', zorder=2, alpha=0.5, label='cr p3du pathlines')
        if source_area == 'Technetium_Source_North' or source_area == 'Technetium_Source_South' or source_area == 'Technetium_Source_S-SX_North' or source_area == 'Technetium_Source_S-SX_South':
            part_pathlines.plot(ax=ax, linewidth=0.2, color='fuchsia', zorder=2, alpha=0.5, label='tec99 p3du pathlines')
        candidate_wells_ew_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells UU/MU')
        candidate_wells_mw_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells UU/MU')
        candidate_wells_ew_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='grey', marker='^', markersize=20, label='candidate extraction wells LU/CR')
        candidate_wells_mw_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='lightgreen', marker='o', markersize=15, label='candidate monitoring wells LU/CR')

        # manually define legend items
        if source_area == 'Chromium_Source_North' or source_area == 'Chromium_Source_South':
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='cr source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='chocolate', linewidth=0.2, alpha=0.5, label='mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], color='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du pathlines'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR')
            ]

        if source_area == 'Technetium_Source_North' or source_area == 'Technetium_Source_South':
            legend_elements = [
                Patch(facecolor='fuchsia', edgecolor='black', alpha=1, label='tec99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='fuchsia', linewidth=0.2, alpha=0.5, label='mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], color='fuchsia', linewidth=0.2, alpha=0.5, label='tec99 mp3du pathlines'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR')
            ]

        if source_area == 'Chromium_Source_S-SX_North' or source_area == 'Chromium_Source_S-SX_South':
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='cr source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA S & WMA SX-SY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='chocolate', linewidth=0.2, alpha=0.5, label='mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], color='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du pathlines'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR')
            ]

        if source_area == 'Technetium_Source_S-SX_North' or source_area == 'Technetium_Source_S-SX_South':
            legend_elements = [
                Patch(facecolor='fuchsia', edgecolor='black', alpha=1, label='tec99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA S & WMA SX-SY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='fuchsia', linewidth=0.2, alpha=0.5, label='mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], color='fuchsia', linewidth=0.2, alpha=0.5, label='tec99 mp3du pathlines'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR')
            ]

        #ax.legend(handles=legend_elements, loc='upper right')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([133300+y_axis_offset, 135250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        else:
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])

        plt.title('Pathlines & Endpoints from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        # now create bounding polygons by drawing using mouse clicking
        polygon_points = []

        def click(event):
            if event.inaxes != ax:
                return

            x, y = event.xdata, event.ydata
            polygon_points.append((x, y))

            # Plot the point
            ax.plot(x, y, 'ro')

            # Draw lines as user clicks
            if len(polygon_points) > 1:
                x_vals, y_vals = zip(*polygon_points[-2:])
                ax.plot(x_vals, y_vals, 'k-')

            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', click)
        plt.show()

        # Close the polygon if enough points are given
        if len(polygon_points) >= 3:
            polygon_points.append(polygon_points[0])  # Close loop
            polygon = Polygon(polygon_points)
            gdf = gpd.GeoDataFrame({'geometry':[polygon]}, crs=mdgrd_crs)
            gdf.to_file(os.path.join(gis_d, 'shp', 'pathlines', f'{source_area}_bounding_raw.shp'))
            print(f'Polygon created for {source_area}')
        else:
            polygon = None
            print(f'Not enough points to create a polygon for {source_area}...')

        # Smooth the generated polygon using Chaikin's Algorithm
        def chaikin_algo(coords, iterations = 3):
            for _ in range(iterations):
                new_coords=[]
                for i in range(len(coords)-1):
                    p1 = coords[i]
                    p2 = coords[i+1]
                    Q = (0.75 * p1[0] + 0.25 * p2[0], 0.75 * p1[1] + 0.25 * p2[1])
                    R = (0.25 * p1[0] + 0.75 * p2[0], 0.25 * p1[1] + 0.75 * p2[1])
                    new_coords.extend([Q,R])
                new_coords.append(new_coords[0])
                coords = new_coords
            return coords
        
        raw_polygon_coords = polygon_points
        chaikin_polygon_coords = chaikin_algo(raw_polygon_coords, iterations=3)
        chaikin_polygon = Polygon(chaikin_polygon_coords)
        gdf_final = gpd.GeoDataFrame({'geometry': [chaikin_polygon]}, crs=mdgrd_crs)
        gdf_final.to_file(os.path.join(gis_d, 'shp', 'pathlines', f'{source_area}_bounding_final.shp'))

    else:
        print('generate_bounding_polygon function selected NOT to run...')

# this function generates centerline for each bounding map of mp3du particle tracking results for continuing source areas
def generate_centerline(flag, gis_d, fig_d, ptrk_calc_d, part_type, source_area):
    if flag: 
        print(f'generating {source_area} centerlines for bounding polygons of particle tracking mp3du results...')
        
        # create output directory for figs
        if not os.path.exists(fig_d):
            os.makedirs(fig_d)
        
        # load in reference gis files for particle starting locations
        cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
        #te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))
        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))
        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))

        # load in particle starting locations, pathlines, and endpoints for each source area type
        if source_area == 'Chromium_Source_North' or source_area == 'Chromium_Source_South' or source_area == 'Chromium_Source_S-SX_North' or source_area == 'Chromium_Source_S-SX_South':
            source_area_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', source_area+'.shp'))
            if part_type == 'ring':
                cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp'))
            else:
                cr_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'particle_starting_locations', 'particle_starting_locations.shp'))

            cr_endpoints = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'cr_endpoints.shp'))

            cr_pathlines = gpd.read_file(os.path.join(ptrk_calc_d, 'cr', 'cr_pathlines.shp'))

            part_starts = gpd.sjoin(cr_part_starts, source_area_gdf, how = 'inner', predicate='intersects')
            part_pathlines = gpd.sjoin(cr_pathlines, source_area_gdf,how = 'inner', predicate='intersects')
            part_endpoints = cr_endpoints
            bounding_polygon = gpd.read_file(os.path.join('gis', 'shp', 'pathlines', f'{source_area}_bounding_final.shp'))
        
        if source_area == 'Technetium_Source_North' or source_area == 'Technetium_Source_South' or source_area == 'Technetium_Source_S-SX_North' or source_area == 'Technetium_Source_S-SX_South':
            source_area_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', source_area+'.shp'))
            if part_type == 'ring':
                tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations_ring', 'particle_starting_locations_ring.shp')) 
            else:
                tec99_part_starts = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'particle_starting_locations', 'particle_starting_locations.shp'))
            
            tec99_endpoints = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'tec-99_endpoints.shp'))

            tec99_pathlines = gpd.read_file(os.path.join(ptrk_calc_d, 'tec-99', 'tec-99_pathlines.shp'))
        
            part_starts = gpd.sjoin(tec99_part_starts, source_area_gdf, how = 'inner', predicate='intersects')
            part_pathlines = gpd.sjoin(tec99_pathlines, source_area_gdf,how = 'inner', predicate='intersects')
            part_endpoints = tec99_endpoints
            bounding_polygon = gpd.read_file(os.path.join('gis', 'shp', 'pathlines', f'{source_area}_bounding_final.shp'))

        # check model grid crs
        mdgrd_crs = model_grid_gdf.crs
        src_crs = source_area_gdf.crs

        if not mdgrd_crs == src_crs:
            print('updating the crs of the model grid for consistency...')
            model_grid_gdf.to_crs(src_crs)
        else:
            print('the crs of the model grid is not being updated for consistency...')

        # define crs of particle pathlines
        part_pathlines = part_pathlines.set_crs(mdgrd_crs)

        # define crs of particle endpoints
        part_endpoints = part_endpoints.set_crs(mdgrd_crs)

        # load in data gap locations
        data_gap_locs = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))

        # load in HWIS Data Pull 2025
        hwis_data_df = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93_reduced.csv'))
        
        # filter by status of entry
        status_list = ['IN-USE', 'CANDIDATE FOR DECOMMISSIONING', 'AWAITING DRILLING']
        hwis_data_df_red = hwis_data_df[hwis_data_df['STATUS'].isin(status_list)]

        # create geopandas dataframe object and export to shapefile
        hwis_data_df_red_geometry = [Point(xy) for xy in zip(hwis_data_df_red['EASTING'], hwis_data_df_red['NORTHING'])]
        hwis_data_gdf_red = gpd.GeoDataFrame(hwis_data_df_red, geometry=hwis_data_df_red_geometry, crs=mdgrd_crs)
        hwis_data_gdf_red.to_file(os.path.join(gis_d, 'shp', 'misc', 'hwis_data_gdf_reduced.shp'))

        # load in candidate monitoring wells and extraction wells
        candidate_wells_mw_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_uu_mu.shp'))
        candidate_wells_ew_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_uu_mu.shp'))
        candidate_wells_mw_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_lu_cr.shp'))
        candidate_wells_ew_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_lu_cr.shp'))

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)
        if source_area == 'Chromium_Source_North' or source_area == 'Chromium_Source_South' or source_area == 'Chromium_Source_S-SX_North' or source_area == 'Chromium_Source_S-SX_South':
            source_area_gdf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='cr source zones')
        if source_area == 'Technetium_Source_North' or source_area == 'Technetium_Source_South' or source_area == 'Technetium_Source_S-SX_North' or source_area == 'Technetium_Source_S-SX_South':
            source_area_gdf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
        wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))         
        model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
        if source_area == 'Chromium_Source_North' or source_area == 'Chromium_Source_South' or source_area == 'Chromium_Source_S-SX_North' or source_area == 'Chromium_Source_S-SX_South':
            part_endpoints.plot(ax=ax, edgecolor='chocolate', markersize=4, zorder=2, facecolor='None', alpha=0.5, label='cr mp3du endpoints')
            part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='chocolate', zorder=1, alpha=1, label='cr particle starting locations')
        if source_area == 'Technetium_Source_North' or source_area == 'Technetium_Source_South' or source_area == 'Technetium_Source_S-SX_North' or source_area == 'Technetium_Source_S-SX_South':
            part_endpoints.plot(ax=ax, edgecolor='fuchsia', markersize=4, zorder=2, facecolor='None', alpha=0.5, label='tec99 mp3du endpoints')
            part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='fuchsia', zorder=3, alpha=1, label='tec99 particle starting locations')
        data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
        hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
        if source_area == 'Chromium_Source_North' or source_area == 'Chromium_Source_South' or source_area == 'Chromium_Source_S-SX_North' or source_area == 'Chromium_Source_S-SX_South':
            part_pathlines.plot(ax=ax, linewidth=0.2, color='chocolate', zorder=2, alpha=0.5, label='cr p3du pathlines')
        if source_area == 'Technetium_Source_North' or source_area == 'Technetium_Source_South' or source_area == 'Technetium_Source_S-SX_North' or source_area == 'Technetium_Source_S-SX_South':
            part_pathlines.plot(ax=ax, linewidth=0.2, color='fuchsia', zorder=2, alpha=0.5, label='tec99 p3du pathlines')
        candidate_wells_ew_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells UU/MU')
        candidate_wells_mw_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells UU/MU')
        candidate_wells_ew_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='grey', marker='^', markersize=20, label='candidate extraction wells LU/CR')
        candidate_wells_mw_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='lightgreen', marker='o', markersize=15, label='candidate monitoring wells LU/CR')
        bounding_polygon.plot(ax=ax, facecolor='salmon', edgecolor='black', alpha=0.25, label='bounding polygon for source area')

        # manually define legend items
        if source_area == 'Chromium_Source_North' or source_area == 'Chromium_Source_South':
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='cr particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], color='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du pathlines'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='salmon', edgecolor='black', alpha=0.25, label='bounding polygon for source area')
            ]
        if source_area == 'Technetium_Source_North' or source_area == 'Technetium_Source_South':
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='fuchsia', linewidth=0.2, alpha=0.5, label='tec99 mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='tec99 particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], color='fuchsia', linewidth=0.2, alpha=0.5, label='tec99 mp3du pathlines'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells'),
                Line2D([0], [0], marker='o', markerfacecolor='lightgreen', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='salmon', edgecolor='black', alpha=0.25, label='bounding polygon for source area')
            ]
        if source_area == 'Chromium_Source_S-SX_North' or source_area == 'Chromium_Source_S-SX_South':
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA S & WMA SX-SY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='cr particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], color='chocolate', linewidth=0.2, alpha=0.5, label='cr mp3du pathlines'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='salmon', edgecolor='black', alpha=0.25, label='bounding polygon for source area')
            ]
        if source_area == 'Technetium_Source_S-SX_North' or source_area == 'Technetium_Source_S-SX_South':
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA S & WMA SX-SY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Patch(facecolor='None', edgecolor='fuchsia', linewidth=0.2, alpha=0.5, label='tec99 mp3du endpoints'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='tec99 particle starting locations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], color='fuchsia', linewidth=0.2, alpha=0.5, label='tec99 mp3du pathlines'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells'),
                Line2D([0], [0], marker='o', markerfacecolor='lightgreen', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='salmon', edgecolor='black', alpha=0.25, label='bounding polygon for source area')
            ]
        #ax.legend(handles=legend_elements, loc='upper right')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([133300+y_axis_offset, 135250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        else:
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])

        plt.title('Pathlines & Endpoints from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        # now create centerlines for bounding polygons by drawing using mouse clicking
        centerline_points = []

        def click(event):
            if event.inaxes != ax:
                return

            x, y = event.xdata, event.ydata
            centerline_points.append((x, y))

            # Plot the point
            ax.plot(x, y, 'ro')

            # Draw lines as user clicks
            if len(centerline_points) > 1:
                x_vals, y_vals = zip(*centerline_points[-2:])
                ax.plot(x_vals, y_vals, 'k-')

            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', click)
        plt.show()

        # Close the centerline polyline if enough points are given
        if len(centerline_points) >= 2:
            centerline = LineString(centerline_points)
            gdf = gpd.GeoDataFrame({'geometry':[centerline]}, crs=mdgrd_crs)
            gdf.to_file(os.path.join(gis_d, 'shp', 'pathlines', f'{source_area}_centerline_final.shp'))
            print(f'Centerline created for {source_area}')
        else:
            polygon = None
            print(f'Not enough points to create a centerline for {source_area}...')

    else:
        print('generate_centerline function selected NOT to run...')

# this function generates a bounding polygon and centerlines map of mp3du particle tracking results
def generate_bounding_centerline_map_all(flag, gis_d, fig_d, ptrk_calc_d, part_type):
    if flag: 
        print('generating bounding polygon and centerline map of particle tracking mp3du results for all constituents...')
        
        # create output directory for figs
        if not os.path.exists(fig_d):
            os.makedirs(fig_d)
        
        # load in reference gis files for particle starting locations
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX.shp'))
        else:
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))

        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))
        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))

        # check model grid crs
        mdgrd_crs = model_grid_gdf.crs
        crsrc_crs = cr_source_gpf.crs

        if not mdgrd_crs == crsrc_crs:
            print('updating the crs of the model grid for consistency...')
            model_grid_gdf.to_crs(crsrc_crs)
        else:
            print('the crs of the model grid is not being updated for consistency...')

        # load in data gap locations
        data_gap_locs = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))

        # load in HWIS Data Pull 2025
        hwis_data_df = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93_reduced.csv'))
        
        # filter by status of entry
        status_list = ['IN-USE', 'CANDIDATE FOR DECOMMISSIONING', 'AWAITING DRILLING']
        hwis_data_df_red = hwis_data_df[hwis_data_df['STATUS'].isin(status_list)]

        # create geopandas dataframe object and export to shapefile
        hwis_data_df_red_geometry = [Point(xy) for xy in zip(hwis_data_df_red['EASTING'], hwis_data_df_red['NORTHING'])]
        hwis_data_gdf_red = gpd.GeoDataFrame(hwis_data_df_red, geometry=hwis_data_df_red_geometry, crs=mdgrd_crs)
        hwis_data_gdf_red.to_file(os.path.join(gis_d, 'shp', 'misc', 'hwis_data_gdf_reduced.shp'))

        # load in candidate monitoring wells and extraction wells
        candidate_wells_mw_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_uu_mu.shp'))
        candidate_wells_ew_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_uu_mu.shp'))
        candidate_wells_mw_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_lu_cr.shp'))
        candidate_wells_ew_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_lu_cr.shp'))

        # load in bounding polygons and centerlines
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_North_bounding_final.shp'))
            cr_south_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_South_bounding_final.shp'))
            tec99_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_S-SX_North_bounding_final.shp'))
            cr_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_North_centerline_final.shp'))
            cr_south_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_South_centerline_final.shp'))
            tec99_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_S-SX_North_centerline_final.shp'))
        else:
            cr_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_North_bounding_final.shp'))
            cr_south_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_South_bounding_final.shp'))
            tec99_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_North_bounding_final.shp'))
            tec99_south_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_South_bounding_final.shp')) 
            cr_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_North_centerline_final.shp'))
            cr_south_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_South_centerline_final.shp'))
            tec99_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_North_centerline_final.shp'))
            tec99_south_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_South_centerline_final.shp'))

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
            te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
            wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
            wma_S_wma_sxsy_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA S & WMA SX-SY')
            model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
            data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
            hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
            candidate_wells_ew_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells UU/MU')
            candidate_wells_mw_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells UU/MU')
            candidate_wells_ew_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='grey', marker='^', markersize=20, label='candidate extraction wells LU/CR')
            candidate_wells_mw_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='lightgreen', marker='o', markersize=15, label='candidate monitoring wells LU/CR')
            cr_north_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.5, label='cr north bounding polygon')
            cr_north_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline')
            cr_south_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.25, label='cr south bounding polygon')
            cr_south_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr south centerline')
            tec99_north_bound.plot(ax=ax, facecolor='fuchsia', edgecolor='black', alpha=0.5, label='tec99 north bounding polygon')
            tec99_north_centerline.plot(ax=ax, color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 north centerline')

            # manually define legend items
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA S & WMA SX-SY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.8, label='cr north bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.6, label='cr south bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--',  alpha=1, label='cr south centerline'),
                Patch(facecolor='fuchsia', edgecolor='black', alpha=0.4, label='tech99 north bounding polygon'),
                Line2D([0], [0], color='red', linewidth=1.5, linestyle='--',  alpha=1, label='tec99 north centerline')
            ]
        else:
            cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
            te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
            wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
            wma_T_wma_txty_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA T & WMA TX-TY')
            model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
            data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
            hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
            candidate_wells_ew_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells UU/MU')
            candidate_wells_mw_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells UU/MU')
            candidate_wells_ew_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='grey', marker='^', markersize=20, label='candidate extraction wells LU/CR')
            candidate_wells_mw_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='lightgreen', marker='o', markersize=15, label='candidate monitoring wells LU/CR')
            cr_north_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.5, label='cr north bounding polygon')
            cr_north_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline')
            cr_south_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.25, label='cr south bounding polygon')
            cr_south_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr south centerline')
            tec99_north_bound.plot(ax=ax, facecolor='fuchsia', edgecolor='black', alpha=0.5, label='tec99 north bounding polygon')
            tec99_north_centerline.plot(ax=ax, color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 north centerline')
            tec99_south_bound.plot(ax=ax, facecolor='fuchsia', edgecolor='black', alpha=0.25, label='tec99 south bounding polygon')
            tec99_south_centerline.plot(ax=ax, color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 south centerline')

            # manually define legend items
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.8, label='cr north bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.6, label='cr south bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--',  alpha=1, label='cr south centerline'),
                Patch(facecolor='fuchsia', edgecolor='black', alpha=0.4, label='tec99 north bounding polygon'),
                Line2D([0], [0], color='red', linewidth=1.5, linestyle='--',  alpha=1, label='tech99 north centerline'),
                Patch(facecolor='fuchsia', edgecolor='black', alpha=0.2, label='tec99 south bounding polygon'),
                Line2D([0], [0], color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 south centerline')
            ]

        ax.legend(handles=legend_elements, loc='upper right')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([133300+y_axis_offset, 135250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        else:
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])

        plt.title('Bounding Polygons & Centerlines from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'all_constituents_mp3du_bounding_centerline_map_rings_ssxsy.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'all_constituents_mp3du_bounding_centerline_map_centroids_ssxsy.png'), dpi=400)
        else:
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'all_constituents_mp3du_bounding_centerline_map_rings.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'all_constituents_mp3du_bounding_centerline_map_centroids.png'), dpi=400)
    else:
        print('generate_bounding_centerline_map_all function selected NOT to run...')

# create points along each source area centerline
def centerline_to_points(flag, gis_d, fig_d, ptrk_calc_d, source_area):
    if flag:
        print(f'turning centerlines into points for {source_area}...')
        polyline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', f'{source_area}_centerline_final.shp'))
        crs_ref = polyline.crs
        spacing = 1 # meters
        points = []
        for idx, row in polyline.iterrows():
            line = row.geometry
            total_length = line.length
            distances = np.arange(0, total_length,spacing)
            if distances[-1] != total_length:
                distances = np.append(distances, total_length)
        for dist in distances:
            point = line.interpolate(dist)
            points.append({
                "geometry": point,
                "line_id": idx,
                "distance_m": dist
            })
        points_gdf = gpd.GeoDataFrame(points, crs=crs_ref)
        points_gdf['x'] = points_gdf.geometry.x
        points_gdf['y'] = points_gdf.geometry.y
        points_gdf.to_file(os.path.join(gis_d, 'shp', 'pathlines', f'{source_area}_centerline_final_points.shp'))
    else:
        print('centerline_to_points function selected NOT to run...')

# this function determines which potential well cells are in the bounding areas for each source zone area
def potential_wells_in_bounding(flag, gis_d, fig_d,ptrk_calc_d, source_area):
    if flag:
        print(f'determining potential wells are in {source_area} bounding...')
        # load in potential well locations
        potential_well_locations = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))
        # load bounding polygon for respective source area
        bounding_polygon = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', f'{source_area}_bounding_final.shp'))
        polygon = bounding_polygon.geometry.iloc[0]  # Assuming you want the first polygon
        locations_within_polygon = potential_well_locations[potential_well_locations.geometry.within(polygon)]
        locations_within_polygon.to_file(os.path.join(gis_d, 'shp', 'pathlines', f'{source_area}_potential_well_locations.shp'))
    else:
        print('potential_wells_in_bounding function selected NOT to run...')

# calculate scores for continuous source areas all constituents combined
def calculate_continuous_source_score_all(flag, gis_d, source_area_list):
    if flag:
        print('calculating continuous source scores for all constituents combined...')
        compiled_scores = []
        for source_area in source_area_list:
            locations_within_polygon = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines',f'{source_area}_potential_well_locations.shp'))
            crs_ref = locations_within_polygon.crs
            centerline_points = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', f'{source_area}_centerline_final_points.shp'))
            coords_a = np.array(list(zip(locations_within_polygon.geometry.x, locations_within_polygon.geometry.y)))
            coords_b = np.array(list(zip(centerline_points.geometry.x, centerline_points.geometry.y)))
            tree = cKDTree(coords_b)
            distances, indices = tree.query(coords_a, k=1)  # Nearest only
            locations_within_polygon["nearest_geom"] = centerline_points.geometry.iloc[indices].values
            locations_within_polygon["nearest_dist_m"] = distances

            locations_within_polygon["dist_cl_m"] = centerline_points["distance_m"].iloc[indices].values
            locations_within_polygon["comb_dist_m"] = locations_within_polygon['nearest_dist_m'] + locations_within_polygon['dist_cl_m']
            locations_within_polygon['row_col_id'] = 10000000+(10000*locations_within_polygon['row'])+locations_within_polygon['col']
            compiled_scores.append(locations_within_polygon)

        compiled_scores_gdf = pd.concat(compiled_scores, ignore_index=True)
        poly_counts = compiled_scores_gdf["row_col_id"].value_counts()
        compiled_scores_gdf["poly_count"] = compiled_scores_gdf["row_col_id"].map(poly_counts)
        min_rows = compiled_scores_gdf.loc[compiled_scores_gdf.groupby("row_col_id")["comb_dist_m"].idxmin()].reset_index(drop=True)
        
        final_compiled_scores_gdf = min_rows.copy()

        # define raw scores
        final_compiled_scores_gdf['raw_score'] = -9999
        for k in range(0, len(final_compiled_scores_gdf)):

            dist_m = final_compiled_scores_gdf.at[k, 'comb_dist_m'] / 2

            # assign scoring
            if dist_m < 50:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 4
            elif 50 <= dist_m < 100:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 3
            elif 100 <= dist_m < 200:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 2
            elif 200 <= dist_m < 300:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 1
            elif dist_m >= 300:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 0
        
        # assign final, scaled score
        final_compiled_scores_gdf['final_score'] = final_compiled_scores_gdf['raw_score'] * final_compiled_scores_gdf['poly_count'] / final_compiled_scores_gdf['poly_count'].max()

        final_compiled_scores_df = final_compiled_scores_gdf[['row_col_id', 'nearest_dist_m', 'dist_cl_m', 'comb_dist_m', 'poly_count', 'raw_score', 'final_score']]
        
        # merge in results to all potential well ids and fill in with ids and zeros where applicable
        all_potential_wells = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))
        all_potential_wells['row_col_id'] = 10000000+(10000*all_potential_wells['row'])+all_potential_wells['col']
        final_compiled_scores_gdf_merged = all_potential_wells.merge(final_compiled_scores_df, on='row_col_id', how='left', suffixes=('','_new'))

        # export to shapefile
        #final_compiled_scores_gdf_merged = final_compiled_scores_gdf_merged.drop(columns='nearest_geom')
        final_compiled_scores_gdf_merged = final_compiled_scores_gdf_merged.drop(columns='geometry')
        final_compiled_scores_gdf_merged = gpd.GeoDataFrame(final_compiled_scores_gdf_merged, geometry=gpd.points_from_xy(final_compiled_scores_gdf_merged.x, final_compiled_scores_gdf_merged.y), crs=crs_ref)
        final_compiled_scores_gdf_merged.to_file(os.path.join(gis_d, 'shp', 'scores', 'all_combined_continuous_source_scores.shp'))

    else:
        print('calculate_continuous_source_score function selected NOT to run for all constituents combined...')

# generate continuous source area map of scores for all constituents combined
def generate_continuous_source_score_map_all(flag, gis_d, fig_d, ptrk_calc_d, part_type):
    if flag: 
        print('generating continuous source scoring map of particle tracking mp3du results for all constituents combined...')
        
        # create output directory for figs
        if not os.path.exists(fig_d):
            os.makedirs(fig_d)
        
        # load in reference gis files for particle starting locations
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX.shp'))
        else:
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))

        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))

        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))

        # check model grid crs
        mdgrd_crs = model_grid_gdf.crs
        crsrc_crs = cr_source_gpf.crs

        if not mdgrd_crs == crsrc_crs:
            print('updating the crs of the model grid for consistency...')
            model_grid_gdf.to_crs(crsrc_crs)
        else:
            print('the crs of the model grid is not being updated for consistency...')

        # load in data gap locations
        data_gap_locs = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))

        # load in HWIS Data Pull 2025
        hwis_data_df = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93_reduced.csv'))
        
        # filter by status of entry
        status_list = ['IN-USE', 'CANDIDATE FOR DECOMMISSIONING', 'AWAITING DRILLING']
        hwis_data_df_red = hwis_data_df[hwis_data_df['STATUS'].isin(status_list)]

        # create geopandas dataframe object and export to shapefile
        hwis_data_df_red_geometry = [Point(xy) for xy in zip(hwis_data_df_red['EASTING'], hwis_data_df_red['NORTHING'])]
        hwis_data_gdf_red = gpd.GeoDataFrame(hwis_data_df_red, geometry=hwis_data_df_red_geometry, crs=mdgrd_crs)
        hwis_data_gdf_red.to_file(os.path.join(gis_d, 'shp', 'misc', 'hwis_data_gdf_reduced.shp'))

        # load in candidate monitoring wells and extraction wells
        candidate_wells_mw_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_uu_mu.shp'))
        candidate_wells_ew_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_uu_mu.shp'))
        candidate_wells_mw_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_lu_cr.shp'))
        candidate_wells_ew_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_lu_cr.shp'))

        # load in bounding polygons and centerlines
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_North_bounding_final.shp'))
            cr_south_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_South_bounding_final.shp'))
            tec99_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_S-SX_North_bounding_final.shp'))
            cr_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_North_centerline_final.shp'))
            cr_south_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_South_centerline_final.shp'))
            tec99_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_S-SX_North_centerline_final.shp'))
        else:
            cr_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_North_bounding_final.shp'))
            cr_south_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_South_bounding_final.shp'))
            tec99_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_North_bounding_final.shp'))
            tec99_south_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_South_bounding_final.shp')) 
            cr_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_North_centerline_final.shp'))
            cr_south_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_South_centerline_final.shp'))
            tec99_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_North_centerline_final.shp'))
            tec99_south_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_South_centerline_final.shp'))

        # load continuous source score points
        continuous_source_score_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'all_combined_continuous_source_scores.shp'))

        # Define the color bands and corresponding colors
        color_bands = [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.3), (1.3, 1.6),
                       (1.6, 2.0), (2.0, 2.3), (2.3, 2.6), (2.6, 2.9), (2.9, 3.0)]
        colors = ['midnightblue', 'royalblue', 'cornflowerblue', 'lightsteelblue', 'lightyellow', 'gold', 'yellow',
                  'darkorange', 'orangered', 'red']

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
            te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
            wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
            wma_S_wma_sxsy_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA S & WMA SX-SY')
            model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
            data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
            hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
            candidate_wells_ew_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells UU/MU')
            candidate_wells_mw_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells UU/MU')
            candidate_wells_ew_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='grey', marker='^', markersize=20, label='candidate extraction wells LU/CR')
            candidate_wells_mw_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='lightgreen', marker='o', markersize=15, label='candidate monitoring wells LU/CR')
            cr_north_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.5, label='cr north bounding polygon')
            cr_north_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline')
            cr_south_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.25, label='cr south bounding polygon')
            cr_south_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr south centerline')
            tec99_north_bound.plot(ax=ax, facecolor='fuchsia', edgecolor='black', alpha=0.5, label='tec99 north bounding polygon')
            tec99_north_centerline.plot(ax=ax, color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 north centerline')
        else:
            cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
            te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
            wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
            wma_T_wma_txty_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA T & WMA TX-TY')
            model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
            data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
            hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
            candidate_wells_ew_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells UU/MU')
            candidate_wells_mw_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells UU/MU')
            candidate_wells_ew_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='grey', marker='^', markersize=20, label='candidate extraction wells LU/CR')
            candidate_wells_mw_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='lightgreen', marker='o', markersize=15, label='candidate monitoring wells LU/CR')
            cr_north_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.5, label='cr north bounding polygon')
            cr_north_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline')
            cr_south_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.25, label='cr south bounding polygon')
            cr_south_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr south centerline')
            tec99_north_bound.plot(ax=ax, facecolor='fuchsia', edgecolor='black', alpha=0.5, label='tec99 north bounding polygon')
            tec99_north_centerline.plot(ax=ax, color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 north centerline')
            tec99_south_bound.plot(ax=ax, facecolor='fuchsia', edgecolor='black', alpha=0.25, label='tec99 south bounding polygon')
            tec99_south_centerline.plot(ax=ax, color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 south centerline')

        colorflood_legend_elements = []
        # plot final continuous source score shapefile here
        for (low, high), color in zip(color_bands, colors):
            subset = continuous_source_score_gdf[(continuous_source_score_gdf['final_scor'] >= low) & (continuous_source_score_gdf['final_scor'] < high)]
            subset.plot(ax=ax, color=color, alpha=1.0, zorder=100, label=f'{low}-{high}')
            # Add corresponding legend patch
            colorflood_legend_elements.append(Patch(facecolor=color, edgecolor='black', alpha=0.75, label=f'{low}-{high}'))

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            # manually define legend items
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA S & WMA SX-SY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.8, label='cr north bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.6, label='cr south bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--',  alpha=1, label='cr south centerline'),
                Patch(facecolor='fuchsia', edgecolor='black', alpha=0.4, label='tec99 north bounding polygon'),
                Line2D([0], [0], color='red', linewidth=1.5, linestyle='--',  alpha=1, label='tec99 north centerline')
            ]
        else:
            # manually define legend items
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.8, label='cr north bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.6, label='cr south bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--',  alpha=1, label='cr south centerline'),
                Patch(facecolor='fuchsia', edgecolor='black', alpha=0.4, label='tec99 north bounding polygon'),
                Line2D([0], [0], color='red', linewidth=1.5, linestyle='--',  alpha=1, label='tec99 north centerline'),
                Patch(facecolor='fuchsia', edgecolor='black', alpha=0.2, label='tec99 south bounding polygon'),
                Line2D([0], [0], color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 south centerline')
            ]

        combined_legend_elements = legend_elements + colorflood_legend_elements

        ax.legend(handles=combined_legend_elements, loc='upper right')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([133300+y_axis_offset, 135250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        else:
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])

        plt.title('Continuous Source Score Map from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'all_constituents_mp3du_continuous_source_score_map_rings_ssxsy.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'all_constituents_mp3du_continuous_source_score_map_centroids_ssxsy.png'), dpi=400)
        else:
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'all_constituents_mp3du_continuous_source_score_map_rings.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'all_constituents_mp3du_continuous_source_score_map_centroids.png'), dpi=400)
    else:
        print('generate_continuous_source_score_map function selected NOT to run for all constituents combined...')

# generate continuous source area map of scores for chromium
def generate_continuous_source_score_map_chromium(flag, gis_d, fig_d, ptrk_calc_d, part_type):
    if flag: 
        print('generating continuous source scoring map of particle tracking mp3du results for chromium combined...')
        
        # create output directory for figs
        if not os.path.exists(fig_d):
            os.makedirs(fig_d)
        
        # load in reference gis files for particle starting locations
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX.shp'))
        else:
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))

        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))

        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))

        # check model grid crs
        mdgrd_crs = model_grid_gdf.crs
        crsrc_crs = cr_source_gpf.crs

        if not mdgrd_crs == crsrc_crs:
            print('updating the crs of the model grid for consistency...')
            model_grid_gdf.to_crs(crsrc_crs)
        else:
            print('the crs of the model grid is not being updated for consistency...')

        # load in data gap locations
        data_gap_locs = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))

        # load in HWIS Data Pull 2025
        hwis_data_df = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93_reduced.csv'))
        
        # filter by status of entry
        status_list = ['IN-USE', 'CANDIDATE FOR DECOMMISSIONING', 'AWAITING DRILLING']
        hwis_data_df_red = hwis_data_df[hwis_data_df['STATUS'].isin(status_list)]

        # create geopandas dataframe object and export to shapefile
        hwis_data_df_red_geometry = [Point(xy) for xy in zip(hwis_data_df_red['EASTING'], hwis_data_df_red['NORTHING'])]
        hwis_data_gdf_red = gpd.GeoDataFrame(hwis_data_df_red, geometry=hwis_data_df_red_geometry, crs=mdgrd_crs)
        hwis_data_gdf_red.to_file(os.path.join(gis_d, 'shp', 'misc', 'hwis_data_gdf_reduced.shp'))

        # load in candidate monitoring wells and extraction wells
        candidate_wells_mw_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_uu_mu.shp'))
        candidate_wells_ew_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_uu_mu.shp'))
        candidate_wells_mw_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_lu_cr.shp'))
        candidate_wells_ew_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_lu_cr.shp'))

        # load in bounding polygons and centerlines
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_North_bounding_final.shp'))
            cr_south_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_South_bounding_final.shp'))
            cr_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_North_centerline_final.shp'))
            cr_south_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_S-SX_South_centerline_final.shp'))
        else:
            cr_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_North_bounding_final.shp'))
            cr_south_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_South_bounding_final.shp'))
            cr_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_North_centerline_final.shp'))
            cr_south_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Chromium_Source_South_centerline_final.shp'))


        # load continuous source score points
        continuous_source_score_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'chromium_continuous_source_scores.shp'))

        # Define the color bands and corresponding colors
        color_bands = [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.3), (1.3, 1.6),
                       (1.6, 2.0), (2.0, 2.3), (2.3, 2.6), (2.6, 2.9), (2.9, 3.0)]
        colors = ['midnightblue', 'royalblue', 'cornflowerblue', 'lightsteelblue', 'lightyellow', 'gold', 'yellow',
                  'darkorange', 'orangered', 'red']

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
            te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
            wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
            wma_S_wma_sxsy_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA S & WMA SX-SY')
            model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
            data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
            hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
            candidate_wells_ew_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells UU/MU')
            candidate_wells_mw_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells UU/MU')
            candidate_wells_ew_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='grey', marker='^', markersize=20, label='candidate extraction wells LU/CR')
            candidate_wells_mw_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='lightgreen', marker='o', markersize=15, label='candidate monitoring wells LU/CR')
            cr_north_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.5, label='cr north bounding polygon')
            cr_north_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline')
            cr_south_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.25, label='cr south bounding polygon')
            cr_south_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr south centerline')
        else:
            cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
            te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
            wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
            wma_T_wma_txty_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA T & WMA TX-TY')
            model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
            data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
            hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
            candidate_wells_ew_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells UU/MU')
            candidate_wells_mw_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells UU/MU')
            candidate_wells_ew_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='grey', marker='^', markersize=20, label='candidate extraction wells LU/CR')
            candidate_wells_mw_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='lightgreen', marker='o', markersize=15, label='candidate monitoring wells LU/CR')
            cr_north_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.5, label='cr north bounding polygon')
            cr_north_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline')
            cr_south_bound.plot(ax=ax, facecolor='chocolate', edgecolor='black', alpha=0.25, label='cr south bounding polygon')
            cr_south_centerline.plot(ax=ax, color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr south centerline')


        colorflood_legend_elements = []
        # plot final continuous source score shapefile here
        for (low, high), color in zip(color_bands, colors):
            subset = continuous_source_score_gdf[(continuous_source_score_gdf['final_scor'] >= low) & (continuous_source_score_gdf['final_scor'] < high)]
            subset.plot(ax=ax, color=color, alpha=1.0, zorder=100, label=f'{low}-{high}')
            # Add corresponding legend patch
            colorflood_legend_elements.append(Patch(facecolor=color, edgecolor='black', alpha=0.75, label=f'{low}-{high}'))

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            # manually define legend items
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA S & WMA SX-SY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.8, label='cr north bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.6, label='cr south bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--',  alpha=1, label='cr south centerline'),
            ]
        else:
            # manually define legend items
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.8, label='cr north bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--', alpha=1, label='cr north centerline'),
                Patch(facecolor='chocolate', edgecolor='black', alpha=0.6, label='cr south bounding polygon'),
                Line2D([0], [0], color='brown', linewidth=1.5, linestyle='--',  alpha=1, label='cr south centerline'),
            ]

        combined_legend_elements = legend_elements + colorflood_legend_elements

        ax.legend(handles=combined_legend_elements, loc='upper right')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([133300+y_axis_offset, 135250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        else:
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])

        plt.title('Chromium Continuous Source Score Map from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'chromium_mp3du_continuous_source_score_map_rings_ssxsy.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'chromium_mp3du_continuous_source_score_map_centroids_ssxsy.png'), dpi=400)
        else:
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'chromium_mp3du_continuous_source_score_map_rings.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'chromium_mp3du_continuous_source_score_map_centroids.png'), dpi=400)
    else:
        print('generate_continuous_source_score_map function selected NOT to run for chromium...')

# generate continuous source area map of scores for tec99
def generate_continuous_source_score_map_tec99(flag, gis_d, fig_d, ptrk_calc_d, part_type):
    if flag: 
        print('generating continuous source scoring map of particle tracking mp3du results for tec99...')
        
        # create output directory for figs
        if not os.path.exists(fig_d):
            os.makedirs(fig_d)
        
        # load in reference gis files for particle starting locations
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source_S-SX.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source_S-SX.shp'))
        else:
            cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
            te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))

        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            wma_S_wma_sxsy_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_S_WMA_SXSY.shp'))
        else:
            wma_T_wma_txty_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WMA_T_WMA_TXTY.shp'))

        model_grid_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'model_grid', 'model_grid.shp'))

        # check model grid crs
        mdgrd_crs = model_grid_gdf.crs
        crsrc_crs = cr_source_gpf.crs

        if not mdgrd_crs == crsrc_crs:
            print('updating the crs of the model grid for consistency...')
            model_grid_gdf.to_crs(crsrc_crs)
        else:
            print('the crs of the model grid is not being updated for consistency...')

        # load in data gap locations
        data_gap_locs = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))

        # load in HWIS Data Pull 2025
        hwis_data_df = pd.read_csv(os.path.join(gis_d, 'xlsx', 'HWIS_Data_Pull', 'qryHWIS_TO93_reduced.csv'))
        
        # filter by status of entry
        status_list = ['IN-USE', 'CANDIDATE FOR DECOMMISSIONING', 'AWAITING DRILLING']
        hwis_data_df_red = hwis_data_df[hwis_data_df['STATUS'].isin(status_list)]

        # create geopandas dataframe object and export to shapefile
        hwis_data_df_red_geometry = [Point(xy) for xy in zip(hwis_data_df_red['EASTING'], hwis_data_df_red['NORTHING'])]
        hwis_data_gdf_red = gpd.GeoDataFrame(hwis_data_df_red, geometry=hwis_data_df_red_geometry, crs=mdgrd_crs)
        hwis_data_gdf_red.to_file(os.path.join(gis_d, 'shp', 'misc', 'hwis_data_gdf_reduced.shp'))

        # load in candidate monitoring wells and extraction wells
        candidate_wells_mw_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_uu_mu.shp'))
        candidate_wells_ew_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_uu_mu.shp'))
        candidate_wells_mw_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_mws_lu_cr.shp'))
        candidate_wells_ew_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'candidate_ews_lu_cr.shp'))

        # load in bounding polygons and centerlines
        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            tec99_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_S-SX_North_bounding_final.shp'))
            tec99_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_S-SX_North_centerline_final.shp'))
        else:    
            tec99_north_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_North_bounding_final.shp'))
            tec99_south_bound = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_South_bounding_final.shp')) 
            tec99_north_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_North_centerline_final.shp'))
            tec99_south_centerline = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', 'Technetium_Source_South_centerline_final.shp'))

        # load continuous source score points
        continuous_source_score_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'tec99_continuous_source_scores.shp'))

        # Define the color bands and corresponding colors
        color_bands = [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.3), (1.3, 1.6),
                       (1.6, 2.0), (2.0, 2.3), (2.3, 2.6), (2.6, 2.9), (2.9, 3.0), (3.0, 4.0), (4.0, 5.0)]
        colors = ['midnightblue', 'royalblue', 'cornflowerblue', 'lightsteelblue', 'lightyellow', 'gold', 'yellow',
                  'darkorange', 'orangered', 'red', 'black', 'white']

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
            te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
            wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
            wma_S_wma_sxsy_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA S & WMA SX-SY')
            model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
            data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
            hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
            candidate_wells_ew_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells UU/MU')
            candidate_wells_mw_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells UU/MU')
            candidate_wells_ew_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='grey', marker='^', markersize=20, label='candidate extraction wells LU/CR')
            candidate_wells_mw_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='lightgreen', marker='o', markersize=15, label='candidate monitoring wells LU/CR')
            tec99_north_bound.plot(ax=ax, facecolor='fuchsia', edgecolor='black', alpha=0.5, label='tec99 north bounding polygon')
            tec99_north_centerline.plot(ax=ax, color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 north centerline')
        else:
            cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
            te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
            wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
            wma_T_wma_txty_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA T & WMA TX-TY')
            model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
            data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor='gainsboro', alpha=0.2, label='data gap locations'),
            hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
            candidate_wells_ew_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='black', marker='^', markersize=20, label='candidate extraction wells UU/MU')
            candidate_wells_mw_uumu_gdf.plot(ax=ax, facecolor='None', edgecolor='green', marker='o', markersize=15, label='candidate monitoring wells UU/MU')
            candidate_wells_ew_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='grey', marker='^', markersize=20, label='candidate extraction wells LU/CR')
            candidate_wells_mw_lucr_gdf.plot(ax=ax, facecolor='None', edgecolor='lightgreen', marker='o', markersize=15, label='candidate monitoring wells LU/CR')
            tec99_north_bound.plot(ax=ax, facecolor='fuchsia', edgecolor='black', alpha=0.5, label='tec99 north bounding polygon')
            tec99_north_centerline.plot(ax=ax, color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 north centerline')
            tec99_south_bound.plot(ax=ax, facecolor='fuchsia', edgecolor='black', alpha=0.25, label='tec99 south bounding polygon')
            tec99_south_centerline.plot(ax=ax, color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 south centerline')

        colorflood_legend_elements = []
        # plot final continuous source score shapefile here
        for (low, high), color in zip(color_bands, colors):
            subset = continuous_source_score_gdf[(continuous_source_score_gdf['final_scor'] >= low) & (continuous_source_score_gdf['final_scor'] < high)]
            subset.plot(ax=ax, color=color, alpha=1.0, zorder=100, label=f'{low}-{high}')
            # Add corresponding legend patch
            colorflood_legend_elements.append(Patch(facecolor=color, edgecolor='black', alpha=0.75, label=f'{low}-{high}'))

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            # manually define legend items
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='fuchsia', edgecolor='black', alpha=0.4, label='tec99 north bounding polygon'),
                Line2D([0], [0], color='red', linewidth=1.5, linestyle='--',  alpha=1, label='tec99 north centerline')
            ]

        else:
            # manually define legend items
            legend_elements = [
                Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
                Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
                Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
                Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
                Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
                Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells UU/MU'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='green', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells UU/MU'),
                Line2D([0], [0], marker='^', markerfacecolor='None', markeredgecolor='grey', linewidth=0.1, alpha=1.0, label = 'candidate extraction wells LU/CR'),
                Line2D([0], [0], marker='o', markerfacecolor='None', markeredgecolor='lightgreen', linewidth=0.1, alpha=1.0, label = 'candidate monitoring wells LU/CR'),
                Patch(facecolor='fuchsia', edgecolor='black', alpha=0.4, label='tech99 north bounding polygon'),
                Line2D([0], [0], color='red', linewidth=1.5, linestyle='--',  alpha=1, label='tech99 north centerline'),
                Patch(facecolor='fuchsia', edgecolor='black', alpha=0.2, label='tec99 south bounding polygon'),
                Line2D([0], [0], color='red', linewidth=1.5, linestyle='--', alpha=1, label='tec99 south centerline')
            ]

        combined_legend_elements = legend_elements + colorflood_legend_elements

        ax.legend(handles=combined_legend_elements, loc='upper right')

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([133300+y_axis_offset, 135250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        else:
            x_axis_offset = 600
            y_axis_offset = 400
            plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
            plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])

        plt.title('Tec99 Continuous Source Score Map from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        if ptrk_calc_d == os.path.join(cwd, 'calcs', 'ptrack-ssx'):
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'tec99_mp3du_continuous_source_score_map_rings_ssxsy.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'tec99_mp3du_continuous_source_score_map_centroids_ssxsy.png'), dpi=400)
        else:
            if part_type == 'ring':
                plt.savefig(os.path.join(fig_d, 'tec99_mp3du_continuous_source_score_map_rings.png'), dpi=400)
            else:
                plt.savefig(os.path.join(fig_d, 'tec99_mp3du_continuous_source_score_map_centroids.png'), dpi=400)

    else:
        print('generate_continuous_source_score_map function selected NOT to run for tec99...')

# calculate scores for continuous source areas chromium only
def calculate_continuous_source_score_chromium(flag, gis_d, source_area_list):
    if flag:
        print('calculating continuous source scores for chromium combined...')
        compiled_scores = []
        for source_area in source_area_list:
            locations_within_polygon = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines',f'{source_area}_potential_well_locations.shp'))
            crs_ref = locations_within_polygon.crs
            centerline_points = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', f'{source_area}_centerline_final_points.shp'))
            coords_a = np.array(list(zip(locations_within_polygon.geometry.x, locations_within_polygon.geometry.y)))
            coords_b = np.array(list(zip(centerline_points.geometry.x, centerline_points.geometry.y)))
            tree = cKDTree(coords_b)
            distances, indices = tree.query(coords_a, k=1)  # Nearest only
            locations_within_polygon["nearest_geom"] = centerline_points.geometry.iloc[indices].values
            locations_within_polygon["nearest_dist_m"] = distances

            locations_within_polygon["dist_cl_m"] = centerline_points["distance_m"].iloc[indices].values
            locations_within_polygon["comb_dist_m"] = locations_within_polygon['nearest_dist_m'] + locations_within_polygon['dist_cl_m']
            locations_within_polygon['row_col_id'] = 10000000+(10000*locations_within_polygon['row'])+locations_within_polygon['col']
            compiled_scores.append(locations_within_polygon)

        compiled_scores_gdf = pd.concat(compiled_scores, ignore_index=True)
        poly_counts = compiled_scores_gdf["row_col_id"].value_counts()
        compiled_scores_gdf["poly_count"] = compiled_scores_gdf["row_col_id"].map(poly_counts)
        min_rows = compiled_scores_gdf.loc[compiled_scores_gdf.groupby("row_col_id")["comb_dist_m"].idxmin()].reset_index(drop=True)
        
        final_compiled_scores_gdf = min_rows.copy()

        # define raw scores
        final_compiled_scores_gdf['raw_score'] = -9999
        for k in range(0, len(final_compiled_scores_gdf)):

            dist_m = final_compiled_scores_gdf.at[k, 'comb_dist_m'] / 2

            # assign scoring
            if dist_m < 50:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 4
            elif 50 <= dist_m < 100:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 3
            elif 100 <= dist_m < 200:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 2
            elif 200 <= dist_m < 300:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 1
            elif dist_m >= 300:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 0
        
        # assign final, scaled score
        final_compiled_scores_gdf['final_score'] = final_compiled_scores_gdf['raw_score'] * final_compiled_scores_gdf['poly_count'] / final_compiled_scores_gdf['poly_count'].max()
        
        final_compiled_scores_df = final_compiled_scores_gdf[['row_col_id', 'nearest_dist_m', 'dist_cl_m', 'comb_dist_m', 'poly_count', 'raw_score', 'final_score']]
        
        # merge in results to all potential well ids and fill in with ids and zeros where applicable
        all_potential_wells = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))
        all_potential_wells['row_col_id'] = 10000000+(10000*all_potential_wells['row'])+all_potential_wells['col']
        final_compiled_scores_gdf_merged = all_potential_wells.merge(final_compiled_scores_df, on='row_col_id', how='left', suffixes=('','_new'))

        # export to shapefile
        final_compiled_scores_gdf_merged = gpd.GeoDataFrame(final_compiled_scores_gdf_merged, geometry=gpd.points_from_xy(final_compiled_scores_gdf_merged.x, final_compiled_scores_gdf_merged.y), crs=crs_ref)
        final_compiled_scores_gdf_merged.to_file(os.path.join(gis_d, 'shp', 'scores', 'chromium_continuous_source_scores.shp'))
        
    else:
        print('calculate_continuous_source_score function selected NOT to run for chromium combined...')

# calculate scores for continuous source areas tec99 only
def calculate_continuous_source_score_tec99(flag, gis_d, source_area_list):
    if flag:
        print('calculating continuous source scores for tec99 combined...')
        compiled_scores = []
        for source_area in source_area_list:
            locations_within_polygon = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines',f'{source_area}_potential_well_locations.shp'))
            crs_ref = locations_within_polygon.crs
            centerline_points = gpd.read_file(os.path.join(gis_d, 'shp', 'pathlines', f'{source_area}_centerline_final_points.shp'))
            coords_a = np.array(list(zip(locations_within_polygon.geometry.x, locations_within_polygon.geometry.y)))
            coords_b = np.array(list(zip(centerline_points.geometry.x, centerline_points.geometry.y)))
            tree = cKDTree(coords_b)
            distances, indices = tree.query(coords_a, k=1)  # Nearest only
            locations_within_polygon["nearest_geom"] = centerline_points.geometry.iloc[indices].values
            locations_within_polygon["nearest_dist_m"] = distances

            locations_within_polygon["dist_cl_m"] = centerline_points["distance_m"].iloc[indices].values
            locations_within_polygon["comb_dist_m"] = locations_within_polygon['nearest_dist_m'] + locations_within_polygon['dist_cl_m']
            locations_within_polygon['row_col_id'] = 10000000+(10000*locations_within_polygon['row'])+locations_within_polygon['col']
            compiled_scores.append(locations_within_polygon)

        compiled_scores_gdf = pd.concat(compiled_scores, ignore_index=True)
        poly_counts = compiled_scores_gdf["row_col_id"].value_counts()
        compiled_scores_gdf["poly_count"] = compiled_scores_gdf["row_col_id"].map(poly_counts)
        min_rows = compiled_scores_gdf.loc[compiled_scores_gdf.groupby("row_col_id")["comb_dist_m"].idxmin()].reset_index(drop=True)
        
        final_compiled_scores_gdf = min_rows.copy()

        # define raw scores
        final_compiled_scores_gdf['raw_score'] = -9999
        for k in range(0, len(final_compiled_scores_gdf)):

            dist_m = final_compiled_scores_gdf.at[k, 'comb_dist_m'] / 2

            # assign scoring
            if dist_m < 50:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 4
            elif 50 <= dist_m < 100:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 3
            elif 100 <= dist_m < 200:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 2
            elif 200 <= dist_m < 300:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 1
            elif dist_m >= 300:
                final_compiled_scores_gdf.at[k, 'raw_score'] = 0
        
        # assign final, scaled score
        final_compiled_scores_gdf['final_score'] = final_compiled_scores_gdf['raw_score'] * final_compiled_scores_gdf['poly_count'] / final_compiled_scores_gdf['poly_count'].max()
        
        final_compiled_scores_df = final_compiled_scores_gdf[['row_col_id', 'nearest_dist_m', 'dist_cl_m', 'comb_dist_m', 'poly_count', 'raw_score', 'final_score']]
        
        # merge in results to all potential well ids and fill in with ids and zeros where applicable
        all_potential_wells = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))
        all_potential_wells['row_col_id'] = 10000000+(10000*all_potential_wells['row'])+all_potential_wells['col']
        final_compiled_scores_gdf_merged = all_potential_wells.merge(final_compiled_scores_df, on='row_col_id', how='left', suffixes=('','_new'))

        # export to shapefile
        final_compiled_scores_gdf_merged = gpd.GeoDataFrame(final_compiled_scores_gdf_merged, geometry=gpd.points_from_xy(final_compiled_scores_gdf_merged.x, final_compiled_scores_gdf_merged.y), crs=crs_ref)
        final_compiled_scores_gdf_merged.to_file(os.path.join(gis_d, 'shp', 'scores', 'tec99_continuous_source_scores.shp'))
        
    else:
        print('calculate_continuous_source_score function selected NOT to run for tec99 combined...')

# calculate scores for continuous source areas ctet only
def calculate_continuous_source_score_ctet(flag, gis_d):
    if flag:
        print('calculating continuous source scores for ctet combined...')

        # assign all values to zero for ctet as there are not continuing sources identified
        all_potential_wells = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))
        all_potential_wells['row_col_id'] = 10000000+(10000*all_potential_wells['row'])+all_potential_wells['col']
        all_potential_wells['nearest_dist_m'] = 0
        all_potential_wells['dist_cl_m'] = 0
        all_potential_wells['comb_dist_m'] = 0
        all_potential_wells['poly_count'] = 0
        all_potential_wells['raw_score'] = 0
        all_potential_wells['final_score'] = 0

        # export to shapefile
        all_potential_wells.to_file(os.path.join(gis_d, 'shp', 'scores', 'ctet_continuous_source_scores.shp'))
        
    else:
        print('calculate_continuous_source_score function selected NOT to run for ctet...')

# combine scores only for Smw, Sew, Scs
def combine_all_scores_only(flag, gis_d):
    if flag:
        print('combine_all_scores_only function selected to run...')

        # load in potential wells shapefile as geopandas gdf & store crs for reference
        potential_wells_gdf = gpd.read_file(os.path.join('gis','shp','data_gap_wells', 'potential_wells.shp'))
        crs_ref = potential_wells_gdf.crs

        potential_wells_df = potential_wells_gdf.drop(columns='geometry')
        potential_wells_df['row_col_id'] = 10000000+(10000*potential_wells_df['row'])+potential_wells_df['col']

        # load in individual scoring shapefiles as gdfs
        mw_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'score_monitoring_well_proximity_uu_mu.shp'))
        mw_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'score_monitoring_well_proximity_lu_cr.shp'))
        ew_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'score_extraction_well_proximity_uu_mu.shp'))
        ew_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'score_extraction_well_proximity_lu_cr.shp'))
        cr_cs_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'chromium_continuous_source_scores.shp'))
        tec99_cs_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'tec99_continuous_source_scores.shp'))
        ctet_cs_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'ctet_continuous_source_scores.shp'))

        # store only necessary information
        mw_uumu_df = mw_uumu_gdf[['row_col_id', 'score_mw']]
        ew_uumu_df = ew_uumu_gdf[['row_col_id', 'score_ext']]
        mw_lucr_df = mw_lucr_gdf[['row_col_id', 'score_mw']]
        ew_lucr_df = ew_lucr_gdf[['row_col_id', 'score_ext']]
        cr_cs_df = cr_cs_gdf[['row_col_id', 'final_scor']]
        tec99_cs_df = tec99_cs_gdf[['row_col_id', 'final_scor']]
        ctet_cs_df = ctet_cs_gdf[['row_col_id', 'final_scor']]

        # fill nan vals with zeros
        cr_cs_df['final_scor'] = cr_cs_df['final_scor'].fillna(0).astype(int)
        tec99_cs_df['final_scor'] = tec99_cs_df['final_scor'].fillna(0).astype(int)
        ctet_cs_df['final_scor'] = ctet_cs_df['final_scor'].fillna(0).astype(int)

        # add in score detail in column name
        mw_uumu_df.rename(columns={'score_mw': 's_mw_uumu'}, inplace=True)
        ew_uumu_df.rename(columns={'score_ext': 's_ext_uumu'}, inplace=True)
        mw_lucr_df.rename(columns={'score_mw': 's_mw_lucr'}, inplace=True)
        ew_lucr_df.rename(columns={'score_ext': 's_ext_lucr'}, inplace=True)
        cr_cs_df.rename(columns={'final_scor': 's_cr_cs'}, inplace=True)
        tec99_cs_df.rename(columns={'final_scor': 's_tec_cs'}, inplace=True)
        ctet_cs_df.rename(columns={'final_scor': 's_ctet_cs'}, inplace=True)

        # merge into one df
        dfs_to_merge = [potential_wells_df, mw_uumu_df, ew_uumu_df, mw_lucr_df, ew_lucr_df, cr_cs_df, tec99_cs_df, ctet_cs_df]

        combined_scores_df = reduce(lambda left, right: pd.merge(left, right, on='row_col_id', how='outer'), dfs_to_merge)
        
        outpth = os.path.join('scores_combined')
        if not os.path.exists(outpth):
            os.makedirs(outpth)

        combined_scores_df['NORTHING'] = combined_scores_df['y'] # meters
        combined_scores_df['EASTING'] = combined_scores_df['x'] # meters

        combined_scores_df.to_csv(os.path.join(outpth, 'scores_only_Smw_Sew_Scs.csv'))

        # create gdf from merged df and export to shapefile
        combined_scores_gdf = gpd.GeoDataFrame(combined_scores_df, geometry = gpd.points_from_xy(combined_scores_df.EASTING, combined_scores_df.NORTHING), crs=crs_ref)
        combined_scores_gdf.to_file(os.path.join(outpth, 'scores_only_Smw_Sew_Scs.shp'))

    else:
        print('combine_all_scores_only function NOT selected to run...')

# combine scores with details for Smw, Sew, Scs
def combine_all_scores_detailed(flag, gis_d):
    if flag:
        print('combine_all_scores_detailed function selected to run...')

        # load in potential wells shapefile as geopandas gdf & store crs for reference
        potential_wells_gdf = gpd.read_file(os.path.join('gis','shp','data_gap_wells', 'potential_wells.shp'))
        crs_ref = potential_wells_gdf.crs

        potential_wells_df = potential_wells_gdf.drop(columns='geometry')
        potential_wells_df['row_col_id'] = 10000000+(10000*potential_wells_df['row'])+potential_wells_df['col']

        # load in individual scoring shapefiles as gdfs
        mw_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'score_monitoring_well_proximity_uu_mu.shp'))
        mw_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'score_monitoring_well_proximity_lu_cr.shp'))
        ew_uumu_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'score_extraction_well_proximity_uu_mu.shp'))
        ew_lucr_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'score_extraction_well_proximity_lu_cr.shp'))
        cr_cs_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'chromium_continuous_source_scores.shp'))
        tec99_cs_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'tec99_continuous_source_scores.shp'))
        ctet_cs_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'scores', 'ctet_continuous_source_scores.shp'))

        # store only necessary information
        mw_uumu_df = mw_uumu_gdf[['row_col_id', 'score_mw', 'Distance_m', 'Mw_Well_Na']]
        ew_uumu_df = ew_uumu_gdf[['row_col_id', 'score_ext', 'Distance_m', 'Ext_Well_N']]
        mw_lucr_df = mw_lucr_gdf[['row_col_id', 'score_mw', 'Distance_m', 'Mw_Well_Na']]
        ew_lucr_df = ew_lucr_gdf[['row_col_id', 'score_ext', 'Distance_m', 'Ext_Well_N']]
        cr_cs_df = cr_cs_gdf[['row_col_id', 'final_scor', 'nearest_di', 'dist_cl_m', 'comb_dist_', 'poly_count', 'raw_score']]
        tec99_cs_df = tec99_cs_gdf[['row_col_id', 'final_scor', 'nearest_di', 'dist_cl_m', 'comb_dist_', 'poly_count', 'raw_score']]
        ctet_cs_df = ctet_cs_gdf[['row_col_id', 'final_scor', 'nearest_di', 'dist_cl_m', 'comb_dist_', 'poly_count', 'raw_score']]

        # fill nan vals with zeros
        cr_cs_df['final_scor'] = cr_cs_df['final_scor'].fillna(0).astype(int)
        tec99_cs_df['final_scor'] = tec99_cs_df['final_scor'].fillna(0).astype(int)
        ctet_cs_df['final_scor'] = ctet_cs_df['final_scor'].fillna(0).astype(int)

        # add in score detail in column name
        mw_uumu_df.rename(columns={'score_mw': 's_mw_uumu'}, inplace=True)
        mw_uumu_df.rename(columns={'Distance_m': 'dist_m_mw_uumu'}, inplace=True)
        mw_uumu_df.rename(columns={'Mw_Well_Na': 'mw_near_mw_uumu'}, inplace=True)

        ew_uumu_df.rename(columns={'score_ext': 's_ext_uumu'}, inplace=True)
        ew_uumu_df.rename(columns={'Distance_m': 'dist_m_ew_uumu'}, inplace=True)
        ew_uumu_df.rename(columns={'Ext_Well_N': 'ew_near_ew_uumu'}, inplace=True)
        
        mw_lucr_df.rename(columns={'score_mw': 's_mw_lucr'}, inplace=True)
        mw_lucr_df.rename(columns={'Distance_m': 'dist_m_mw_lucr'}, inplace=True)
        mw_lucr_df.rename(columns={'Mw_Well_Na': 'mw_near_mw_lucr'}, inplace=True)
        
        ew_lucr_df.rename(columns={'score_ext': 's_ext_lucr'}, inplace=True)
        ew_lucr_df.rename(columns={'Distance_m': 'dist_m_ew_lucr'}, inplace=True)
        ew_lucr_df.rename(columns={'Ext_Well_N': 'ew_near_ew_lucr'}, inplace=True)
        
        cr_cs_df.rename(columns={'final_scor': 's_cr_cs'}, inplace=True)
        cr_cs_df.rename(columns={'nearest_di': 'neardist_m_cr_cs'}, inplace=True)
        cr_cs_df.rename(columns={'dist_cl_m': 'cl_dist_m_cr_cs'}, inplace=True)
        cr_cs_df.rename(columns={'comb_dist_': 'combdist_m_cr_cs'}, inplace=True)
        cr_cs_df.rename(columns={'poly_count': 'plycnt_cr_cs'}, inplace=True)
        cr_cs_df.rename(columns={'raw_score': 'raw_s_cr_cs'}, inplace=True)

        tec99_cs_df.rename(columns={'final_scor': 's_tec_cs'}, inplace=True)
        tec99_cs_df.rename(columns={'nearest_di': 'neardist_m_tec_cs'}, inplace=True)
        tec99_cs_df.rename(columns={'dist_cl_m': 'cl_dist_m_tec_cs'}, inplace=True)
        tec99_cs_df.rename(columns={'comb_dist_': 'combdist_m_tec_cs'}, inplace=True)
        tec99_cs_df.rename(columns={'poly_count': 'plycnt_tec_cs'}, inplace=True)
        tec99_cs_df.rename(columns={'raw_score': 'raw_s_tec_cs'}, inplace=True)
        
        ctet_cs_df.rename(columns={'final_scor': 's_ctet_cs'}, inplace=True)
        ctet_cs_df.rename(columns={'nearest_di': 'neardist_m_ctet_cs'}, inplace=True)
        ctet_cs_df.rename(columns={'dist_cl_m': 'cl_dist_m_ctet_cs'}, inplace=True)
        ctet_cs_df.rename(columns={'comb_dist_': 'combdist_m_ctet_cs'}, inplace=True)
        ctet_cs_df.rename(columns={'poly_count': 'plycnt_ctet_cs'}, inplace=True)
        ctet_cs_df.rename(columns={'raw_score': 'raw_s_ctet_cs'}, inplace=True)

        # fillna with zeros where appropriate
        cr_cs_df['neardist_m_cr_cs'] = cr_cs_df['neardist_m_cr_cs'].fillna(0)
        cr_cs_df['cl_dist_m_cr_cs'] = cr_cs_df['cl_dist_m_cr_cs'].fillna(0)
        cr_cs_df['combdist_m_cr_cs'] = cr_cs_df['combdist_m_cr_cs'].fillna(0)
        cr_cs_df['plycnt_cr_cs'] = cr_cs_df['plycnt_cr_cs'].fillna(0)
        cr_cs_df['raw_s_cr_cs'] = cr_cs_df['raw_s_cr_cs'].fillna(0)

        tec99_cs_df['neardist_m_tec_cs'] = tec99_cs_df['neardist_m_tec_cs'].fillna(0)
        tec99_cs_df['cl_dist_m_tec_cs'] = tec99_cs_df['cl_dist_m_tec_cs'].fillna(0)
        tec99_cs_df['combdist_m_tec_cs'] = tec99_cs_df['combdist_m_tec_cs'].fillna(0)
        tec99_cs_df['plycnt_tec_cs'] = tec99_cs_df['plycnt_tec_cs'].fillna(0)
        tec99_cs_df['raw_s_tec_cs'] = tec99_cs_df['raw_s_tec_cs'].fillna(0)

        # merge into one df
        dfs_to_merge = [potential_wells_df, mw_uumu_df, ew_uumu_df, mw_lucr_df, ew_lucr_df, cr_cs_df, tec99_cs_df, ctet_cs_df]

        combined_scores_df = reduce(lambda left, right: pd.merge(left, right, on='row_col_id', how='outer'), dfs_to_merge)
        
        outpth = os.path.join('scores_combined')
        if not os.path.exists(outpth):
            os.makedirs(outpth)

        combined_scores_df['NORTHING'] = combined_scores_df['y'] # meters
        combined_scores_df['EASTING'] = combined_scores_df['x'] # meters

        combined_scores_df.to_csv(os.path.join(outpth, 'scores_detailed_Smw_Sew_Scs.csv'))

        # create gdf from merged df and export to shapefile
        combined_scores_gdf = gpd.GeoDataFrame(combined_scores_df, geometry = gpd.points_from_xy(combined_scores_df.EASTING, combined_scores_df.NORTHING), crs=crs_ref)
        combined_scores_gdf.to_file(os.path.join(outpth, 'scores_detailed_Smw_Sew_Scs.shp'))

    else:
        print('combine_all_scores_detailed function NOT selected to run...')

# this main function contains all of the calculations, processing, plotting, scoring outputs
def main():
    if flag_new_ecf == True:

        ################################################################################################################
        ################################################################################################################
        ################################## ECF-200ZP1-25-0092 Calculations Starting Here ###############################
        ################################################################################################################
        ################################################################################################################
        print('starting ecf calculations...')
        
        # calculate scores for proximity/distance to extraction wells
        print('starting proximal distance to extraction well scoring calculations...')
        calc_proximal_distance_to_ews(flag_proximal_distance_to_ews, gis_d, fig_d, data_gap_wells_flag='updated')

        # calculate scores for proximity/distance to monitoring wells
        print('starting proximal distance to monitoring well scoring calculations...')
        calc_proximal_distance_to_mws(flag_proximal_distance_to_mws, gis_d, fig_d, data_gap_wells_flag='updated')

        # start particle tracking calculations
        print('starting particle tracking calculations...')
        for constituent in constituent_list:
            create_ptrk_folder(flag_create_ptrk_folder, constituent, flow_source_d, ptrk_calc_d, exe_d, exe_list)
        
        # ssx tank farms continuing source
        for constituent in constituent_list:
            create_ptrk_folder(flag_create_ptrk_folder, constituent, flow_source_d, ptrk_ssx_calc_d, exe_d, exe_list)

        copy_transport_props(flag_copy_transport_props, transport_source_d, ptrk_calc_d)
        
        # ssx tank farms continuing source
        copy_transport_props(flag_copy_transport_props, transport_source_d, ptrk_ssx_calc_d)
        
        mf_inp_fnm = 'P2Rv8.3_start2015_sp2024.nam'
        for constituent in constituent_list:
            run_modflow(flag_run_modflow, constituent, ptrk_calc_d, mf_executable, mf_inp_fnm)

        # ssx tank farms continuing source
        for constituent in constituent_list:
            run_modflow(flag_run_modflow, constituent, ptrk_ssx_calc_d, mf_executable, mf_inp_fnm)

        xoff = 557800.00
        yoff = 142800.00 - 26600.00 # 142800 is the top of the model grid in the y direction
        rot = 0.000000
        flow_fnm = mf_inp_fnm
        dis_type = 'STRUCTURED_DIS'
        gsf_nm = 'mp3du.gsf'
        gsf_json_nm = 'gsf_json'

        for constituent in constituent_list:
            write_gsf_json_input(flag_write_gsf_json_input, constituent, ptrk_calc_d, 
                                xoff, yoff, rot, flow_fnm, dis_type, 
                                gsf_nm, gsf_json_nm
                                )
        
        # ssx tank farms continuing source
        for constituent in constituent_list:
            write_gsf_json_input(flag_write_gsf_json_input, constituent, ptrk_ssx_calc_d, 
                                xoff, yoff, rot, flow_fnm, dis_type, 
                                gsf_nm, gsf_json_nm
                                )
            
        for constituent in constituent_list:
            run_gsfwriter(flag_run_gsfwriter, constituent, ptrk_calc_d, gsf_executable, gsf_json_nm)

        # ssx tank farms continuing source
        for constituent in constituent_list:
            run_gsfwriter(flag_run_gsfwriter, constituent, ptrk_ssx_calc_d, gsf_executable, gsf_json_nm)

        binary_print_ID = 88 # check to make sure that you are not overwriting any other packages with this binary ID
        package_name = 'mp3du.p3d'
        package_type = 'PATH'
        for constituent in constituent_list:
            modify_nam_file_with_new_package_mp3du(flag_modify_nam_file_with_new_package_mp3du, 
                                                   constituent, ptrk_calc_d, mf_inp_fnm, 
                                                   binary_print_ID, package_name, package_type
                                                   )
        
        # ssx tank farms continuing source
        for constituent in constituent_list:
            modify_nam_file_with_new_package_mp3du(flag_modify_nam_file_with_new_package_mp3du, 
                                                   constituent, ptrk_ssx_calc_d, mf_inp_fnm, 
                                                   binary_print_ID, package_name, package_type
                                                   )

        p3d_fnm = package_name
        for constituent in constituent_list:
            write_p3d_mp3du(flag_write_p3d_mp3du, constituent, ptrk_calc_d, p3d_fnm)

        # ssx tank farms continuing source
        for constituent in constituent_list:
            write_p3d_mp3du(flag_write_p3d_mp3du, constituent, ptrk_ssx_calc_d, p3d_fnm)

        pstrt_fnm = 'particle_starting_locations'
        for constituent in constituent_list:
            generate_part_start_locs(flag_generate_part_start_locs, constituent,
                                     ptrk_calc_d, pstrt_fnm, gis_d
                                     )
            
        # ssx tank farms continuing source
        for constituent in constituent_list:
            generate_part_start_locs(flag_generate_part_start_locs, constituent,
                                     ptrk_ssx_calc_d, pstrt_fnm, gis_d
                                     )

        mp3du_json_nm = mf_inp_fnm.replace('.nam', '')
        part_type = 'ring'
        for constituent in constituent_list:
            pthlin_nm = f'{constituent}.shp'
            write_mp3du_json_input(flag_mp3du_json_input, constituent, 
                                   ptrk_calc_d, gsf_nm, flow_fnm,
                                   mp3du_json_nm, pstrt_fnm, pthlin_nm,
                                   part_type
                                   )
            
        # ssx tank farms continuing source
        for constituent in constituent_list:
            pthlin_nm = f'{constituent}.shp'
            write_mp3du_json_input(flag_mp3du_json_input, constituent, 
                                   ptrk_ssx_calc_d, gsf_nm, flow_fnm,
                                   mp3du_json_nm, pstrt_fnm, pthlin_nm,
                                   part_type
                                   )

        input_file = mp3du_json_nm+'.json'
        for constituent in constituent_list:
            run_mp3du(flag_run_mp3du, constituent, ptrk_calc_d, mp3du_executable, input_file)

        # ssx tank farms continuing source
        for constituent in constituent_list:
            run_mp3du(flag_run_mp3du, constituent, ptrk_ssx_calc_d, mp3du_executable, input_file)

        for constituent in constituent_list:
            p3doutput_json_nm = f'{constituent}_path_p3doutput'
            write_p3doutput_json_input_path(flag_write_p3doutput_json_input_path, constituent,
                                       ptrk_calc_d, p3doutput_json_nm
                                       )

        # ssx tank farms continuing source
        for constituent in constituent_list:
            p3doutput_json_nm = f'{constituent}_path_p3doutput'
            write_p3doutput_json_input_path(flag_write_p3doutput_json_input_path, constituent,
                                       ptrk_ssx_calc_d, p3doutput_json_nm
                                       )
            
        for constituent in constituent_list:
            p3doutput_json_nm = f'{constituent}_path_p3doutput'+'.json'
            run_writep3doutput(flag_run_writep3doutput, constituent, 
                               ptrk_calc_d, writep3doutput_executable,
                               p3doutput_json_nm
                               )

        # ssx tank farms continuing source
        for constituent in constituent_list:
            p3doutput_json_nm = f'{constituent}_path_p3doutput'+'.json'
            run_writep3doutput(flag_run_writep3doutput, constituent, 
                               ptrk_ssx_calc_d, writep3doutput_executable,
                               p3doutput_json_nm
                               )
            
        for constituent in constituent_list:
            p3doutput_json_nm = f'{constituent}_endpts_p3doutput'
            write_p3doutput_json_input_endpts(flag_write_p3doutput_json_input_endpts, constituent,
                                       ptrk_calc_d, p3doutput_json_nm
                                       )

        # ssx tank farms continuing source
        for constituent in constituent_list:
            p3doutput_json_nm = f'{constituent}_endpts_p3doutput'
            write_p3doutput_json_input_endpts(flag_write_p3doutput_json_input_endpts, constituent,
                                       ptrk_ssx_calc_d, p3doutput_json_nm
                                       )

        for constituent in constituent_list:
            p3doutput_json_nm = f'{constituent}_endpts_p3doutput'+'.json'
            run_writep3doutput(flag_run_writep3doutput, constituent, 
                               ptrk_calc_d, writep3doutput_executable,
                               p3doutput_json_nm
                               )
            
        # ssx tank farms continuing source
        for constituent in constituent_list:
            p3doutput_json_nm = f'{constituent}_endpts_p3doutput'+'.json'
            run_writep3doutput(flag_run_writep3doutput, constituent, 
                               ptrk_ssx_calc_d, writep3doutput_executable,
                               p3doutput_json_nm
                               )

        # plot results of particle tracking
        generate_pathlines_map(flag_generate_pathlines_map, gis_d, fig_d, ptrk_calc_d, part_type)
        # ssx tank farms continuing source
        generate_pathlines_map(flag_generate_pathlines_map, gis_d, fig_d, ptrk_ssx_calc_d, part_type)

        generate_endpoints_map(flag_generate_endpoints_map, gis_d, fig_d, ptrk_calc_d, part_type)
        # ssx tank farms continuing source
        generate_endpoints_map(flag_generate_endpoints_map, gis_d, fig_d, ptrk_ssx_calc_d, part_type)
        
        generate_pathlines_endpoints_map(flag_generate_pathlines_endpoints_map, gis_d, fig_d, ptrk_calc_d, part_type)
        # ssx tank farms continuing source
        generate_pathlines_endpoints_map(flag_generate_pathlines_endpoints_map, gis_d, fig_d, ptrk_ssx_calc_d, part_type)

        calc_relative_path_count(flag_calc_relative_path_count, gis_d, ptrk_calc_d)
        # ssx tank farms continuing source
        calc_relative_path_count(flag_calc_relative_path_count, gis_d, ptrk_ssx_calc_d)

        generate_relcount_pathlines_map(flag_relcount_pathlines_map, gis_d, fig_d, ptrk_calc_d, part_type)
        # ssx tank farms continuing source
        generate_relcount_pathlines_map(flag_relcount_pathlines_map, gis_d, fig_d, ptrk_ssx_calc_d, part_type)

        # generate bounding polygons for each continuing source
        parse_source_zones(flag_parse_source_zones, gis_d, fig_d, ptrk_calc_d, part_type)

        # ssx tank farms continuing source
        parse_source_zones(flag_parse_source_zones, gis_d, fig_d, ptrk_ssx_calc_d, part_type)
        
        # run these functions if bounding and centerline shapefiles do not exists
        source_area_list = ['Chromium_Source_North', 'Chromium_Source_South', 'Technetium_Source_North', 'Technetium_Source_South']
        for source_area in source_area_list:
            if flag_generate_bounding_and_centerline:
                
                print(f'creating final bounding polygon from scratch for {source_area}...')
                generate_bounding_polygon(flag_generate_bounding_polygon, gis_d, fig_d, ptrk_calc_d, part_type, source_area)
                
                print(f'creating final centerline from scratch for {source_area}...')
                generate_centerline(flag_generate_centerline, gis_d, fig_d, ptrk_calc_d, part_type, source_area)
            
            else:
                print(f'NOT creating final bounding polygon from scratch for {source_area}...')
                print(f'NOT creating final centerline from scratch for {source_area}...')
            
        # now discretize each center line into points & determine which potential well cells are in bounding areas
        for source_area in source_area_list:
            centerline_to_points(flag_centerline_to_points, gis_d, fig_d, ptrk_calc_d, source_area)
            potential_wells_in_bounding(flag_potential_wells_in_bounding, gis_d, fig_d,ptrk_calc_d, source_area)

        # plot bounding polygons with centerlines for all continuing source areas combined
        generate_bounding_centerline_map_all(flag_generate_bounding_centerline_map_all, gis_d, fig_d, ptrk_calc_d, part_type)
        
        # ssx tank farms continuing source
        # run these functions if bounding and centerline shapefiles do not exists
        source_area_list = ['Chromium_Source_S-SX_North', 'Chromium_Source_S-SX_South', 'Technetium_Source_S-SX_North']
        for source_area in source_area_list:
            if flag_generate_bounding_and_centerline:
                
                print(f'creating final bounding polygon from scratch for {source_area}...')
                generate_bounding_polygon(flag_generate_bounding_polygon, gis_d, fig_d, ptrk_ssx_calc_d, part_type, source_area)
                
                print(f'creating final centerline from scratch for {source_area}...')
                generate_centerline(flag_generate_centerline, gis_d, fig_d, ptrk_ssx_calc_d, part_type, source_area)
            
            else:
                print(f'NOT creating final bounding polygon from scratch for {source_area}...')
                print(f'NOT creating final centerline from scratch for {source_area}...')
            
        # now discretize each center line into points & determine which potential well cells are in bounding areas
        for source_area in source_area_list:
            centerline_to_points(flag_centerline_to_points, gis_d, fig_d, ptrk_ssx_calc_d, source_area)
            potential_wells_in_bounding(flag_potential_wells_in_bounding, gis_d, fig_d,ptrk_ssx_calc_d, source_area)

        # plot bounding polygons with centerlines for all continuing source areas combined
        generate_bounding_centerline_map_all(flag_generate_bounding_centerline_map_all, gis_d, fig_d, ptrk_ssx_calc_d, part_type)

        # define all source area sites, including WMA S and WMA SX & SY
        source_area_list = ['Chromium_Source_North', 'Chromium_Source_South', 'Technetium_Source_North', 'Technetium_Source_South', 'Chromium_Source_S-SX_North', 'Chromium_Source_S-SX_South', 'Technetium_Source_S-SX_North']
        # calculate continuous source scores for all continuing source areas combined
        calculate_continuous_source_score_all(flag_calculate_continuous_source_score_all, gis_d, source_area_list)
        
        # generate map of continuous source scores for all continuing source areas combined
        generate_continuous_source_score_map_all(flag_generate_continuous_source_score_map_all, gis_d, fig_d, ptrk_calc_d, part_type)
        # ssx tank farms continuing source
        generate_continuous_source_score_map_all(flag_generate_continuous_source_score_map_all, gis_d, fig_d, ptrk_ssx_calc_d, part_type)

        # calculate scores for chromium only
        source_area_list_cr = ['Chromium_Source_North', 'Chromium_Source_South', 'Chromium_Source_S-SX_North', 'Chromium_Source_S-SX_South']
        calculate_continuous_source_score_chromium(flag_calculate_continuous_source_score_chromium, gis_d, source_area_list_cr)
        # generate map of continuous source scores for chromium
        generate_continuous_source_score_map_chromium(flag_generate_continuous_source_score_map_chromium, gis_d, fig_d, ptrk_calc_d, part_type)
        # ssx tank farms continuing source
        generate_continuous_source_score_map_chromium(flag_generate_continuous_source_score_map_chromium, gis_d, fig_d, ptrk_ssx_calc_d, part_type)
        
        # calculate scores for tec99 only
        source_area_list_tec99 = ['Technetium_Source_North', 'Technetium_Source_South', 'Technetium_Source_S-SX_North']
        calculate_continuous_source_score_tec99(flag_calculate_continuous_source_score_tec99, gis_d, source_area_list_tec99)
        # generate map of continuous source scores for tec99
        generate_continuous_source_score_map_tec99(flag_generate_continuous_source_score_map_tec99, gis_d, fig_d, ptrk_calc_d, part_type)
        # ssx tank farms continuing source
        generate_continuous_source_score_map_tec99(flag_generate_continuous_source_score_map_tec99, gis_d, fig_d, ptrk_ssx_calc_d, part_type)

        # calculate scores for ctet only (all zeros)
        calculate_continuous_source_score_ctet(flag_calculate_continuous_source_score_ctet, gis_d)

        # combine calculated scores into csvs and shapefiles for Smw, Sew, Scs
        combine_all_scores_only(flag_combine_all_scores_only, gis_d)             
        combine_all_scores_detailed(flag_combine_all_scores_detailed, gis_d)
        
    else:
        print('ecf calculations workflows NOT selected to run, \n check booleans...')

# this runs the main function
if __name__ == "__main__":
    main()