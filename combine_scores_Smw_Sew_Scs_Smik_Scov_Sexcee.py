# -*- coding: utf-8 -*-
"""
Created September 2025 

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
#### Combines all scores together  Smw, Sew, Scs, Smik, Scov, Sexcee     ####
#### Also extracts and exports sample dry well scoring to results table  ####


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
from scipy.stats import percentileofscore

# define some global variables
ecf_name = 'ECF-200ZP1-25-0092'

# collect the local working directory
cwd = os.getcwd()

exe_d = os.path.join(cwd, 'bin', 'win')
flow_source_d = os.path.join(cwd, 'source_files', 'flow', 'source')
transport_source_d = os.path.join(cwd, 'source_files', 'transport')
ptrk_calc_d = os.path.join(cwd, 'calcs', 'ptrack')
gis_d = os.path.join(cwd, 'gis')
fig_d = os.path.join(cwd, 'figs')

# global boolen to turn the ECF workflow on, must be turned to call helper functions in the main()
flag_combine_scores = True

# booleans to turn on each calc function incrementally that are called in main()
# perform sequentially by turning each boolen on, running function, then turning off before running the next function. 
# Please note, figures will need to be closed as they open up for the workflow to continue forward

################################################################################################
# FIRST CODE BLOCK TO RUN. Can be run in series, turn off after successfuly completion.
################################################################################################

# combine scores Smw, Sew, Scs, Smik, Scov, Sexcee with details and scores only
flag_combine_all_scores = True
tag_list = ['only', 'detailed']
################################################################################################

# define helper functions for the workflow calculations

# combine scores for Smw, Sew, Scs, Smik, Scov, Sexcee 
# exports sample dry well scoring results to table
def combine_all_scores(flag, tag):
    if flag:
        print('combine_all_scores function selected to run...')

        # load in potential wells shapefile as geopandas gdf & store crs for reference
        potential_wells_gdf = gpd.read_file(os.path.join('gis','shp','data_gap_wells', 'potential_wells.shp'))
        crs_ref = potential_wells_gdf.crs

        potential_wells_df = potential_wells_gdf.drop(columns='geometry')
        potential_wells_df['row_col_id'] = 10000000+(10000*potential_wells_df['row'])+potential_wells_df['col']

        # load in grid for the score results to be spatially mapped to
        grid_gdf = gpd.read_file(os.path.join('gis', 'shp', 'model_grid', 'model_grid.shp'))
        grid_gdf = grid_gdf.to_crs(crs_ref)

        # load in pandas dataframes
        Smw_Sew_Scs_df = pd.read_csv(os.path.join('scores_combined', f'scores_{tag}_Smw_Sew_Scs.csv'))
        Smik_Scov_Sexcee_df = pd.read_csv(os.path.join('scores_combined', f'scores_{tag}_Smik_Scov_Sexcee.csv'))
        Smik_Scov_Sexcee_df['row_col_id'] = 10000000+(10000*Smik_Scov_Sexcee_df['row'])+Smik_Scov_Sexcee_df['col']
        Smik_Scov_Sexcee_df.drop(columns=['row', 'col', 'x', 'y', 'Shape'], inplace=True)

        # combine all scores
        combined_scores_df = pd.merge(Smw_Sew_Scs_df, Smik_Scov_Sexcee_df, on='row_col_id', how='outer')

        # calculate final total datagap scores
        combined_scores_df['ctet_uumu_tdg_s'] = Smw_Sew_Scs_df['s_mw_uumu'] + Smw_Sew_Scs_df['s_ext_uumu'] + Smw_Sew_Scs_df['s_ctet_cs'] + Smik_Scov_Sexcee_df['ctet_uu_mu_mik_cv_score'] + Smik_Scov_Sexcee_df['ctet_uu_mu_mik_cov_score'] + Smik_Scov_Sexcee_df['ctet_uumu_clean_level_exceedance_score']
        combined_scores_df['ctet_lucr_tdg_s'] = Smw_Sew_Scs_df['s_mw_lucr'] + Smw_Sew_Scs_df['s_ext_lucr'] + Smw_Sew_Scs_df['s_ctet_cs'] + Smik_Scov_Sexcee_df['ctet_lu_cr_mik_cv_score'] + Smik_Scov_Sexcee_df['ctet_lu_cr_mik_cov_score'] + Smik_Scov_Sexcee_df['ctet_lucr_clean_level_exceedance_score'] 

        combined_scores_df['hcr_uumu_tdg_s'] = Smw_Sew_Scs_df['s_mw_uumu'] + Smw_Sew_Scs_df['s_ext_uumu'] + Smw_Sew_Scs_df['s_cr_cs'] + Smik_Scov_Sexcee_df['hexcr_uu_mu_mik_cv_score'] + Smik_Scov_Sexcee_df['hexcr_uu_mu_mik_cov_score'] + Smik_Scov_Sexcee_df['hexcr_uumu_clean_level_exceedance_score']
        combined_scores_df['hcr_lucr_tdg_s'] = Smw_Sew_Scs_df['s_mw_lucr'] + Smw_Sew_Scs_df['s_ext_lucr'] + Smw_Sew_Scs_df['s_cr_cs'] + Smik_Scov_Sexcee_df['hexcr_lu_cr_mik_cv_score'] + Smik_Scov_Sexcee_df['hexcr_lu_cr_mik_cov_score'] + Smik_Scov_Sexcee_df['hexcr_lucr_clean_level_exceedance_score']

        combined_scores_df['tec_uumu_tdg_s'] = Smw_Sew_Scs_df['s_mw_uumu'] + Smw_Sew_Scs_df['s_ext_uumu'] + Smw_Sew_Scs_df['s_tec_cs'] + Smik_Scov_Sexcee_df['tc99_uu_mu_mik_cv_score'] + Smik_Scov_Sexcee_df['tc99_uu_mu_mik_cov_score'] + Smik_Scov_Sexcee_df['tc99_uumu_clean_level_exceedance_score']
        combined_scores_df['tec_lucr_tdg_s'] = Smw_Sew_Scs_df['s_mw_lucr'] + Smw_Sew_Scs_df['s_ext_lucr'] + Smw_Sew_Scs_df['s_tec_cs'] + Smik_Scov_Sexcee_df['tc99_lu_cr_mik_cv_score'] + Smik_Scov_Sexcee_df['tc99_lu_cr_mik_cov_score'] + Smik_Scov_Sexcee_df['tc99_lucr_clean_level_exceedance_score'] 

        combined_scores_df['uumu_s_tot_cs'] = Smw_Sew_Scs_df['s_ctet_cs'] + Smw_Sew_Scs_df['s_cr_cs'] + Smw_Sew_Scs_df['s_tec_cs']
        combined_scores_df['uumu_s_tot_mik'] = Smik_Scov_Sexcee_df['ctet_uu_mu_mik_cv_score'] + Smik_Scov_Sexcee_df['hexcr_uu_mu_mik_cv_score'] + Smik_Scov_Sexcee_df['tc99_uu_mu_mik_cv_score']
        combined_scores_df['uumu_s_tot_cov'] = Smik_Scov_Sexcee_df['ctet_uu_mu_mik_cov_score'] + Smik_Scov_Sexcee_df['hexcr_uu_mu_mik_cov_score'] + Smik_Scov_Sexcee_df['tc99_uu_mu_mik_cov_score']
        combined_scores_df['uumu_s_tot_excee'] = Smik_Scov_Sexcee_df['ctet_uumu_clean_level_exceedance_score'] + Smik_Scov_Sexcee_df['hexcr_uumu_clean_level_exceedance_score'] + Smik_Scov_Sexcee_df['tc99_uumu_clean_level_exceedance_score']
        
        combined_scores_df['lucr_s_tot_cs'] = Smw_Sew_Scs_df['s_ctet_cs'] + Smw_Sew_Scs_df['s_cr_cs'] + Smw_Sew_Scs_df['s_tec_cs'] 
        combined_scores_df['lucr_s_tot_mik'] = Smik_Scov_Sexcee_df['ctet_lu_cr_mik_cv_score'] + Smik_Scov_Sexcee_df['hexcr_lu_cr_mik_cv_score'] + Smik_Scov_Sexcee_df['tc99_lu_cr_mik_cv_score']
        combined_scores_df['lucr_s_tot_cov'] = Smik_Scov_Sexcee_df['ctet_lu_cr_mik_cov_score'] + Smik_Scov_Sexcee_df['hexcr_lu_cr_mik_cov_score'] + Smik_Scov_Sexcee_df['tc99_lu_cr_mik_cov_score']
        combined_scores_df['lucr_s_tot_excee'] = Smik_Scov_Sexcee_df['ctet_lucr_clean_level_exceedance_score'] + Smik_Scov_Sexcee_df['hexcr_lucr_clean_level_exceedance_score'] + Smik_Scov_Sexcee_df['tc99_lucr_clean_level_exceedance_score']

        combined_scores_df['uumu_s_grandtotal'] = Smw_Sew_Scs_df['s_mw_uumu'] + Smw_Sew_Scs_df['s_ext_uumu'] + combined_scores_df['uumu_s_tot_cs'] + combined_scores_df['uumu_s_tot_mik'] + combined_scores_df['uumu_s_tot_cov'] + combined_scores_df['uumu_s_tot_excee']
        combined_scores_df['lucr_s_grandtotal'] = Smw_Sew_Scs_df['s_mw_lucr'] + Smw_Sew_Scs_df['s_ext_lucr'] + combined_scores_df['lucr_s_tot_cs'] + combined_scores_df['lucr_s_tot_mik'] + combined_scores_df['lucr_s_tot_cov'] + combined_scores_df['lucr_s_tot_excee']

        # contaminant-specific contaminant persistence scores
        combined_scores_df['hcr_uumu_persis_s'] = Smw_Sew_Scs_df['s_cr_cs'] + Smik_Scov_Sexcee_df['hexcr_uumu_clean_level_exceedance_score']
        combined_scores_df['hcr_lucr_persis_s'] = Smw_Sew_Scs_df['s_cr_cs'] + Smik_Scov_Sexcee_df['hexcr_lucr_clean_level_exceedance_score']

        combined_scores_df['tec_uumu_persis_s'] = Smw_Sew_Scs_df['s_tec_cs'] + Smik_Scov_Sexcee_df['tc99_uumu_clean_level_exceedance_score']
        combined_scores_df['tec_lucr_persis_s'] = Smw_Sew_Scs_df['s_tec_cs'] + Smik_Scov_Sexcee_df['tc99_lucr_clean_level_exceedance_score']

        combined_scores_df['ctet_uumu_persis_s'] = Smw_Sew_Scs_df['s_ctet_cs'] + Smik_Scov_Sexcee_df['ctet_uumu_clean_level_exceedance_score']
        combined_scores_df['ctet_lucr_persis_s'] = Smw_Sew_Scs_df['s_ctet_cs'] + Smik_Scov_Sexcee_df['ctet_lucr_clean_level_exceedance_score']

        # calculate percentiles for grand total scores in the UU/MU and LU/CR
        # the percentiles here to be contextualized with all 5330 data gap score locations
        combined_scores_df['uumu_s_grandtotal_percentile'] = combined_scores_df['uumu_s_grandtotal'].apply(lambda x: percentileofscore(combined_scores_df['uumu_s_grandtotal'], x, kind='rank'))
        combined_scores_df['lucr_s_grandtotal_percentile'] = combined_scores_df['lucr_s_grandtotal'].apply(lambda x: percentileofscore(combined_scores_df['lucr_s_grandtotal'], x, kind='rank'))

        # export to csv
        combined_scores_df.to_csv(os.path.join('scores_combined', f'scores_{tag}_Smw_Sew_Scs_Smik_Scov_Sexcee.csv'))

        # create gdf from merged df and export to shapefile
        combined_scores_gdf = gpd.GeoDataFrame(combined_scores_df, geometry = gpd.points_from_xy(combined_scores_df.EASTING, combined_scores_df.NORTHING), crs=crs_ref)
        combined_scores_gdf.to_file(os.path.join('scores_combined', f'scores_{tag}_Smw_Sew_Scs_Smik_Scov_Sexcee.shp'))
        combined_scores_gdf_grid = gpd.sjoin(grid_gdf, combined_scores_gdf, how='inner', predicate='contains')
        combined_scores_gdf_grid.to_file(os.path.join('scores_combined', f'scores_{tag}_Smw_Sew_Scs_Smik_Scov_Sexcee_grid.shp'))

        # drop conflict columns for secondary merge
        combined_scores_gdf_grid = combined_scores_gdf_grid.drop(columns=['index_right'])
        # now pull in sample dry wells to extract calculated scores for the respective locations
        sample_dry_wells_df = pd.read_csv(os.path.join('gis','xlsx', 'Sample_Dry_Wells', 'sampledrywells.csv'))
        sample_dry_wells_gdf = gpd.GeoDataFrame(sample_dry_wells_df, geometry = gpd.points_from_xy(sample_dry_wells_df.EASTING, sample_dry_wells_df.NORTHING), crs=crs_ref)

        # now extract results from combined_scores_gdf_grd and save to shapefile and csv
        sample_dry_wells_scores_grid = gpd.sjoin(combined_scores_gdf_grid, sample_dry_wells_gdf, how='inner', predicate='contains')
        sample_dry_wells_scores_grid.to_file(os.path.join('scores_combined', f'sample_dry_wells_scores_{tag}_Smw_Sew_Scs_Smik_Scov_Sexcee_grid.shp'))
        sample_dry_wells_scores_grid = sample_dry_wells_scores_grid.drop(columns='geometry')

        # calculate percentiles for grand total scores in the UU/MU and LU/CR
        # update the percentiles here to be contextualized with all 5330 data gap score locations, calculate percentiles before merge datagap wells
        #sample_dry_wells_scores_grid['uumu_s_grandtotal_percentile'] = sample_dry_wells_scores_grid['uumu_s_grandtotal'].apply(lambda x: percentileofscore(sample_dry_wells_scores_grid['uumu_s_grandtotal'], x, kind='rank'))
        #sample_dry_wells_scores_grid['lucr_s_grandtotal_percentile'] = sample_dry_wells_scores_grid['lucr_s_grandtotal'].apply(lambda x: percentileofscore(sample_dry_wells_scores_grid['lucr_s_grandtotal'], x, kind='rank'))
        
        sample_dry_wells_scores_grid.to_csv(os.path.join('scores_combined', f'sample_dry_wells_scores_{tag}_Smw_Sew_Scs_Smik_Scov_Sexcee_grid.csv'))

        if tag == 'only':

            uumu_grandtotal_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'uumu_s_grandtotal']]
            uumu_grandtotal_gdf = gpd.GeoDataFrame(uumu_grandtotal_df, geometry = gpd.points_from_xy(uumu_grandtotal_df.EASTING, uumu_grandtotal_df.NORTHING), crs=crs_ref)
            uumu_grandtotal_gdf.to_file(os.path.join('scores_combined', f'uumu_grandtotaldatagap_score_{tag}.shp'))
            uumu_grandtotal_gdf_grid = gpd.sjoin(grid_gdf, uumu_grandtotal_gdf, how='inner', predicate='contains')
            uumu_grandtotal_gdf_grid.to_file(os.path.join('scores_combined', f'uumu_grandtotaldatagap_score_{tag}_grid.shp'))

            lucr_grandtotal_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'lucr_s_grandtotal']]
            lucr_grandtotal_gdf = gpd.GeoDataFrame(lucr_grandtotal_df, geometry = gpd.points_from_xy(lucr_grandtotal_df.EASTING, lucr_grandtotal_df.NORTHING), crs=crs_ref)
            lucr_grandtotal_gdf.to_file(os.path.join('scores_combined', f'lucr_grandtotaldatagap_score_{tag}.shp'))
            lucr_grandtotal_gdf_grid = gpd.sjoin(grid_gdf, lucr_grandtotal_gdf, how='inner', predicate='contains')
            lucr_grandtotal_gdf_grid.to_file(os.path.join('scores_combined', f'lucr_grandtotaldatagap_score_{tag}_grid.shp'))

            ctet_uumu_tdg_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'ctet_uumu_tdg_s']]
            ctet_uumu_tdg_s_gdf = gpd.GeoDataFrame(ctet_uumu_tdg_s_df, geometry = gpd.points_from_xy(ctet_uumu_tdg_s_df.EASTING, ctet_uumu_tdg_s_df.NORTHING), crs=crs_ref)
            ctet_uumu_tdg_s_gdf.to_file(os.path.join('scores_combined', f'ctet_uumu_totaldatagap_score_{tag}.shp'))
            ctet_uumu_tdg_s_gdf_grid = gpd.sjoin(grid_gdf, ctet_uumu_tdg_s_gdf, how='inner', predicate='contains')
            ctet_uumu_tdg_s_gdf_grid.to_file(os.path.join('scores_combined', f'ctet_uumu_totaldatagap_score_{tag}_grid.shp'))

            ctet_lucr_tdg_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'ctet_lucr_tdg_s']]
            ctet_lucr_tdg_s_gdf = gpd.GeoDataFrame(ctet_lucr_tdg_s_df, geometry = gpd.points_from_xy(ctet_lucr_tdg_s_df.EASTING, ctet_uumu_tdg_s_df.NORTHING), crs=crs_ref)
            ctet_lucr_tdg_s_gdf.to_file(os.path.join('scores_combined', f'ctet_lucr_totaldatagap_score_{tag}.shp'))
            ctet_lucr_tdg_s_gdf_grid = gpd.sjoin(grid_gdf, ctet_lucr_tdg_s_gdf, how='inner', predicate='contains')
            ctet_lucr_tdg_s_gdf_grid.to_file(os.path.join('scores_combined', f'ctet_lucr_totaldatagap_score_{tag}_grid.shp'))

            hcr_uumu_tdg_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'hcr_uumu_tdg_s']]
            hcr_uumu_tdg_s_gdf = gpd.GeoDataFrame(hcr_uumu_tdg_s_df, geometry = gpd.points_from_xy(hcr_uumu_tdg_s_df.EASTING, hcr_uumu_tdg_s_df.NORTHING), crs=crs_ref)
            hcr_uumu_tdg_s_gdf.to_file(os.path.join('scores_combined', f'hcr_uumu_totaldatagap_score_{tag}.shp'))
            hcr_uumu_tdg_s_gdf_grid = gpd.sjoin(grid_gdf, hcr_uumu_tdg_s_gdf, how='inner', predicate='contains')
            hcr_uumu_tdg_s_gdf_grid.to_file(os.path.join('scores_combined', f'hcr_uumu_totaldatagap_score_{tag}_grid.shp'))

            hcr_lucr_tdg_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'hcr_lucr_tdg_s']]
            hcr_lucr_tdg_s_gdf = gpd.GeoDataFrame(hcr_lucr_tdg_s_df, geometry = gpd.points_from_xy(hcr_lucr_tdg_s_df.EASTING, hcr_uumu_tdg_s_df.NORTHING), crs=crs_ref)
            hcr_lucr_tdg_s_gdf.to_file(os.path.join('scores_combined', f'hcr_lucr_totaldatagap_score_{tag}.shp'))
            hcr_lucr_tdg_s_gdf_grid = gpd.sjoin(grid_gdf, hcr_lucr_tdg_s_gdf, how='inner', predicate='contains')
            hcr_lucr_tdg_s_gdf_grid.to_file(os.path.join('scores_combined', f'hcr_lucr_totaldatagap_score_{tag}_grid.shp'))

            tec_uumu_tdg_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'tec_uumu_tdg_s']]
            tec_uumu_tdg_s_gdf = gpd.GeoDataFrame(tec_uumu_tdg_s_df, geometry = gpd.points_from_xy(tec_uumu_tdg_s_df.EASTING, tec_uumu_tdg_s_df.NORTHING), crs=crs_ref)
            tec_uumu_tdg_s_gdf.to_file(os.path.join('scores_combined', f'tec_uumu_totaldatagap_score_{tag}.shp'))
            tec_uumu_tdg_s_gdf_grid = gpd.sjoin(grid_gdf, tec_uumu_tdg_s_gdf, how='inner', predicate='contains')
            tec_uumu_tdg_s_gdf_grid.to_file(os.path.join('scores_combined', f'tec_uumu_totaldatagap_score_{tag}_grid.shp'))

            tec_lucr_tdg_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'tec_lucr_tdg_s']]
            tec_lucr_tdg_s_gdf = gpd.GeoDataFrame(tec_lucr_tdg_s_df, geometry = gpd.points_from_xy(tec_lucr_tdg_s_df.EASTING, tec_uumu_tdg_s_df.NORTHING), crs=crs_ref)
            tec_lucr_tdg_s_gdf.to_file(os.path.join('scores_combined', f'tec_lucr_totaldatagap_score_{tag}.shp'))
            tec_lucr_tdg_s_gdf_grid = gpd.sjoin(grid_gdf, tec_lucr_tdg_s_gdf, how='inner', predicate='contains')
            tec_lucr_tdg_s_gdf_grid.to_file(os.path.join('scores_combined', f'tec_lucr_totaldatagap_score_{tag}_grid.shp'))

            # contaminant-specific contaminant persistence scores
            hcr_uumu_persis_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'hcr_uumu_persis_s']]
            hcr_uumu_persis_s_gdf = gpd.GeoDataFrame(hcr_uumu_persis_s_df, geometry = gpd.points_from_xy(hcr_uumu_persis_s_df.EASTING, hcr_uumu_persis_s_df.NORTHING), crs=crs_ref)
            hcr_uumu_persis_s_gdf.to_file(os.path.join('scores_combined', f'hcr_uumu_persistence_score_{tag}.shp'))
            hcr_uumu_persis_s_gdf_grid = gpd.sjoin(grid_gdf, hcr_uumu_persis_s_gdf, how='inner', predicate='contains')
            hcr_uumu_persis_s_gdf_grid.to_file(os.path.join('scores_combined', f'hcr_uumu_persistence_score_{tag}_grid.shp'))

            hcr_lucr_persis_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'hcr_lucr_persis_s']]
            hcr_lucr_persis_s_gdf = gpd.GeoDataFrame(hcr_lucr_persis_s_df, geometry = gpd.points_from_xy(hcr_lucr_persis_s_df.EASTING, hcr_lucr_persis_s_df.NORTHING), crs=crs_ref)
            hcr_lucr_persis_s_gdf.to_file(os.path.join('scores_combined', f'hcr_lucr_persistence_score_{tag}.shp'))
            hcr_lucr_persis_s_gdf_grid = gpd.sjoin(grid_gdf, hcr_lucr_persis_s_gdf, how='inner', predicate='contains')
            hcr_lucr_persis_s_gdf_grid.to_file(os.path.join('scores_combined', f'hcr_lucr_persistence_score_{tag}_grid.shp'))

            tec_uumu_persis_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'tec_uumu_persis_s']]
            tec_uumu_persis_s_gdf = gpd.GeoDataFrame(tec_uumu_persis_s_df, geometry = gpd.points_from_xy(tec_uumu_persis_s_df.EASTING, tec_uumu_persis_s_df.NORTHING), crs=crs_ref)
            tec_uumu_persis_s_gdf.to_file(os.path.join('scores_combined', f'tec_uumu_persistence_score_{tag}.shp'))
            tec_uumu_persis_s_gdf_grid = gpd.sjoin(grid_gdf, tec_uumu_persis_s_gdf, how='inner', predicate='contains')
            tec_uumu_persis_s_gdf_grid.to_file(os.path.join('scores_combined', f'tec_uumu_persistence_score_{tag}_grid.shp'))

            tec_lucr_persis_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'tec_lucr_persis_s']]
            tec_lucr_persis_s_gdf = gpd.GeoDataFrame(tec_lucr_persis_s_df, geometry = gpd.points_from_xy(tec_lucr_persis_s_df.EASTING, tec_lucr_persis_s_df.NORTHING), crs=crs_ref)
            tec_lucr_persis_s_gdf.to_file(os.path.join('scores_combined', f'tec_lucr_persistence_score_{tag}.shp'))
            tec_lucr_persis_s_gdf_grid = gpd.sjoin(grid_gdf, tec_lucr_persis_s_gdf, how='inner', predicate='contains')
            tec_lucr_persis_s_gdf_grid.to_file(os.path.join('scores_combined', f'tec_lucr_persistence_score_{tag}_grid.shp'))

            ctet_uumu_persis_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'ctet_uumu_persis_s']]
            ctet_uumu_persis_s_gdf = gpd.GeoDataFrame(ctet_uumu_persis_s_df, geometry = gpd.points_from_xy(ctet_uumu_persis_s_df.EASTING, ctet_uumu_persis_s_df.NORTHING), crs=crs_ref)
            ctet_uumu_persis_s_gdf.to_file(os.path.join('scores_combined', f'ctet_uumu_persistence_score_{tag}.shp'))
            ctet_uumu_persis_s_gdf_grid = gpd.sjoin(grid_gdf, ctet_uumu_persis_s_gdf, how='inner', predicate='contains')
            ctet_uumu_persis_s_gdf_grid.to_file(os.path.join('scores_combined', f'ctet_uumu_persistence_score_{tag}_grid.shp'))

            ctet_lucr_persis_s_df = combined_scores_df[['row', 'col', 'row_col_id', 'x', 'y', 'EASTING', 'NORTHING', 'ctet_lucr_persis_s']]
            ctet_lucr_persis_s_gdf = gpd.GeoDataFrame(ctet_lucr_persis_s_df, geometry = gpd.points_from_xy(ctet_lucr_persis_s_df.EASTING, ctet_lucr_persis_s_df.NORTHING), crs=crs_ref)
            ctet_lucr_persis_s_gdf.to_file(os.path.join('scores_combined', f'ctet_lucr_persistence_score_{tag}.shp'))
            ctet_lucr_persis_s_gdf_grid = gpd.sjoin(grid_gdf, ctet_lucr_persis_s_gdf, how='inner', predicate='contains')
            ctet_lucr_persis_s_gdf_grid.to_file(os.path.join('scores_combined', f'ctet_lucr_persistence_score_{tag}_grid.shp'))

    else:
        print('combine_all_scores function NOT selected to run...')


# this main function contains all of the calculations, processing, plotting, scoring outputs
def main():
    if flag_combine_scores == True:

        ################################################################################################################
        ################################################################################################################
        ################################## ECF-200ZP1-25-0092 Calculations Starting Here ###############################
        ################################################################################################################
        ################################################################################################################
        print('combining scores for Smw, Sew, Scs, Smik, Scov, Sexcee...')
        
        # combine calculated scores into csvs and shapefiles for Smw, Sew, Scs, Smik, Scov, Sexcee
        for tag in tag_list:
            combine_all_scores(flag_combine_all_scores, tag)             
        
    else:
        print('ecf calculations workflows NOT selected to run, \n check booleans...')

# this runs the main function
if __name__ == "__main__":
    main()