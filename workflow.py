# -*- coding: utf-8 -*-
"""
Created July 2025

@author: afoster
"""

##############################################################################
##############################################################################

########################## ECF-200E-25-0027 ################################
############################ python workflow #################################

#### This workflow is developed to support particle tracking simulations ####
#### at sites in the 200E area w/modpath-3du particle tracking program   ####
#### with P2R Flow Model Version 8.3 with temporal extension to 2137     ####

##############################################################################
##############################################################################

# note make sure to create the virtual environments from the .yml files
# note you will need to clone and use an arcpy virtual environment for ARCGIS PRO
# the arcpy virtual environment processing will need the advanced 3D spatial analyst licensing activated for ARCGIS PRO

# import necessary python packages and libraries
import os
import glob
import subprocess
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import shapefile
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from shapely.geometry import Point
from osgeo import gdal
from datetime import datetime
start_time = datetime.now()
import shapefile
from math import sin, cos, pi
import numpy as np
import time
import json

# define some global variables
# define ecf
ecf_name = 'ECF-200E-25-0027_R0'
# define output folder name
ecf_folder_name = 'RCRA.Quarterly.2025'

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
gis_d = os.path.join(cwd, 'gis')
fig_d = os.path.join(cwd, 'figs')

# global boolen to turn the workflow on
flag_new_ecf = False

# booleans to turn on each calc function incrementally, perform sequentially. 
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
flag_write_p3doutput_json_input = False
flag_run_writep3doutput = False
flag_generate_pathlines_map = False
flag_calc_relative_path_count = False
flag_relcount_pathlines_map = False

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
            source_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
        
        if constituent == 'tec-99':
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
        
        # only keep particle starting locations if in layers 3 or 4
        lays_of_interest = [3, 4]
        prt_strt_locs_df = prt_strt_locs_df[prt_strt_locs_df['layer'].isin(lays_of_interest)]

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
                        "CAPTURE_RADIUS": 10.0,
                        "OPTIONS": [
                            "DISPERSION",
                            "RETARDATION",
                            "TERMINATION"
                        ],
                        "PARTICLE_START_LOCATIONS": {
                            "REPEAT": 1,
                            "REPEAT_DT": 365,
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
                        "CAPTURE_RADIUS": 10.0,
                        "OPTIONS": [
                            "DISPERSION",
                            "RETARDATION",
                            "TERMINATION"
                        ],
                        "PARTICLE_START_LOCATIONS": {
                            "REPEAT": 1,
                            "REPEAT_DT": 365,
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

# this function generates a json file that goes into writep3d output files from bin mp3du outputs
def write_p3doutput_json_input(flag, constituent, ptrk_calc_d, p3doutput_json_nm):
    if flag:
        print(f'creating p3doutput json for {constituent}...')

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
        print('write_gsf_json_input Function selected NOT to run')

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
        cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
        te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))
        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))
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

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)

        cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
        te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
        wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
        wma_T_wma_txty_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA T & WMA TX-TY')
        model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
        cr_pathlines.plot(ax=ax, linewidth=0.2, color='chocolate', zorder=1, alpha=0.5, label='cr mp3du pathlines')
        tec99_pathlines.plot(ax=ax, linewidth=0.2, color='fuchsia', zorder=1, alpha=0.5, label='tec-99 mp3du pathlines')
        cr_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='chocolate', zorder=1, alpha=1, label='cr particle starting locations')
        tec99_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='fuchsia', zorder=1, alpha=1, label='tec-99 particle starting locations')
        data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor=None, alpha=0.2, label='data gap locations'),
        hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
        
        # manually define legend items
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
            Line2D([0], [0], marker='o', color='w', markerfacecolor=None, markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
            Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs')
        ]

        ax.legend(handles=legend_elements, loc='upper right')

        x_axis_offset = 600
        y_axis_offset = 400
        plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
        plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        plt.title('Pathlines from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        if part_type == 'ring':
            plt.savefig(os.path.join(fig_d, 'mp3du_pathlines_map_rings.png'), dpi=400)
        else:
            plt.savefig(os.path.join(fig_d, 'mp3du_pathlines_map_centroids.png'), dpi=400)
        plt.show()

    else:
        print('generate_pathlines_map function selected NOT to run...')

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
        pathlines_gdf.to_file(os.path.join(pathlines_fpth, 'pathlines_both_source_areas.shp')) 

        # spatial join combined source area pathlines with data gap model cell locations
        mdgrd_sjoin = gpd.sjoin(model_grid_gdf, pathlines_gdf, how='left', predicate='intersects')
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
        mdgrd_sjoin.to_file(os.path.join(out_fpth, "mdgrd_sjoin_with_counts.shp"))

        # calculate the relative path count per data gap model cell location & normalize to the max
        # Keep only the first occurrence of each unique ID
        unique_mdgrd_gdf_sjoin = mdgrd_sjoin.drop_duplicates(subset=["ID"]).reset_index(drop=True)

        print('calculating relative detectability...')
        # calculate relative detectability (rd) by ECF-200PO1-21-0021 Section 6.2 -> rd (decimal %) = N/MNP
        unique_mdgrd_gdf_sjoin['rel_det'] = unique_mdgrd_gdf_sjoin['path_count'] / unique_mdgrd_gdf_sjoin['path_count'].max()

        # export the relative normalization to a shapefile
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
        cr_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Chromium_Source.shp'))
        te_source_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'source_areas', 'Technetium_Source.shp'))
        wids_poly_gpf = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'WIDS_polygons_published.shp'))
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
        relcount_path_gdf = gpd.read_file(os.path.join(gis_d, 'shp', 'pathline_count', 'mp3du_relative_detectability.shp'))
                                          
        # load in data gap locations
        data_gap_locs = gpd.read_file(os.path.join(gis_d, 'shp', 'data_gap_wells', 'potential_wells.shp'))

        # load in HWIS Data Pull 2025
        hwis_data_gdf_red = gpd.read_file(os.path.join(gis_d, 'shp', 'misc', 'hwis_data_gdf_reduced.shp'))

        # start plotting shapefiles
        fig, ax = plt.subplots(figsize=(10,10), dpi=400)

        cr_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='brown', alpha=1, label='chromium source zones')
        te_source_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='violet', alpha=1, label='tec-99 source zones')
        wids_poly_gpf.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=1, facecolor='lightgrey', alpha=0.3, label='WIDS')
        wma_T_wma_txty_gpf.plot(ax=ax, edgecolor='black', linewidth=0.25, zorder=1, facecolor='yellow', alpha=0.2, label='WMA T & WMA TX-TY')
        model_grid_gdf.plot(ax=ax, edgecolor='black', linewidth=0.20, zorder=0, facecolor='white', alpha=0.2, label='flow model grid')
        cr_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='chocolate', zorder=1, alpha=1, label='cr particle starting locations')
        tec99_part_starts.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.2, facecolor='fuchsia', zorder=1, alpha=1, label='tec-99 particle starting locations')
        data_gap_locs.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1, facecolor=None, alpha=0.2, label='data gap locations'),
        hwis_data_gdf_red.plot(ax=ax, edgecolor='black', markersize=4, linewidth=0.1, zorder=1, facecolor='black', alpha=1.0, label='hwis pull locs')
        
        colorflood_legend_elements = []
        # plot relative pathline count shapefile here
        for (low, high), color in zip(color_bands, colors):
            subset = relcount_path_gdf[(relcount_path_gdf['rel_det'] >= low) & (relcount_path_gdf['rel_det'] < high)]
            subset.plot(ax=ax, color=color, alpha=0.75, label=f'{low}-{high}')
            # Add corresponding legend patch
            colorflood_legend_elements.append(Patch(facecolor=color, edgecolor='black', alpha=0.75, label=f'{low}-{high}'))


        # manually define legend items
        legend_elements = [
            Patch(facecolor='brown', edgecolor='black', alpha=1, label='Chromium source zones'),
            Patch(facecolor='violet', edgecolor='black', alpha=1, label='Tec-99 source zones'),
            Patch(facecolor='lightgrey', edgecolor='black', alpha=0.3, label='WIDS'),
            Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='WMA T & WMA TX-TY'),
            Patch(facecolor='white', edgecolor='black', alpha=0.2, label='flow model grid'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='chocolate', markeredgecolor='black', alpha=1, markersize=8, label='cr particle starting locations'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='fuchsia', markeredgecolor='black', alpha=1, markersize=8, label='tec-99 particle starting locations'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=None, markeredgecolor='black', linewidth=0.7, alpha=0.2, label='data gap locations'),
            Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', linewidth=0.1, alpha=1.0, label = 'hwis pull locs')
        ]

        combined_legend_elements = legend_elements + colorflood_legend_elements

        ax.legend(handles=combined_legend_elements, loc='upper right')

        x_axis_offset = 600
        y_axis_offset = 400
        plt.ylim([135300+y_axis_offset, 137250+y_axis_offset])
        plt.xlim([566000+x_axis_offset, 567950+x_axis_offset])
        plt.title('Color Flood of Relative Pathline Counts from mp3du tracking')
        plt.ylabel('Northing (meters)')
        plt.xlabel('Easting (meters)')
        plt.tight_layout()

        if part_type == 'ring':
            plt.savefig(os.path.join(fig_d, 'mp3du_relative_count_pathlines_map_rings.png'), dpi=400)
        else:
            plt.savefig(os.path.join(fig_d, 'mp3du_relative_count_pathlines_map_centroids.png'), dpi=400)
        plt.show()

    else:
        print('generate_relcount_pathlines_map function selected NOT to run...')

# this main function contains all of the calculations and processing
def main():
    if flag_new_ecf == True:

        ################################################################################################################
        ################################################################################################################
        ################################## ECF-200E-25-0027 Calculations Starting Here ###############################
        ################################################################################################################
        ################################################################################################################
        print('starting ecf calculations...')
        for constituent in constituent_list:
            create_ptrk_folder(flag_create_ptrk_folder, constituent, flow_source_d, ptrk_calc_d, exe_d, exe_list)

        copy_transport_props(flag_copy_transport_props, transport_source_d, ptrk_calc_d)

        mf_inp_fnm = 'P2Rv8.3_start2015_sp2024.nam'
        for constituent in constituent_list:
            run_modflow(flag_run_modflow, constituent, ptrk_calc_d, mf_executable, mf_inp_fnm)

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

        for constituent in constituent_list:
            run_gsfwriter(flag_run_gsfwriter, constituent, ptrk_calc_d, gsf_executable, gsf_json_nm)

        binary_print_ID = 88 # check to make sure that you are not overwriting any other packages with this binary ID
        package_name = 'mp3du.p3d'
        package_type = 'PATH'
        for constituent in constituent_list:
            modify_nam_file_with_new_package_mp3du(flag_modify_nam_file_with_new_package_mp3du, 
                                                   constituent, ptrk_calc_d, mf_inp_fnm, 
                                                   binary_print_ID, package_name, package_type
                                                   )

        p3d_fnm = package_name
        for constituent in constituent_list:
            write_p3d_mp3du(flag_write_p3d_mp3du, constituent, ptrk_calc_d, p3d_fnm)

        pstrt_fnm = 'particle_starting_locations'
        for constituent in constituent_list:
            generate_part_start_locs(flag_generate_part_start_locs, constituent,
                                     ptrk_calc_d, pstrt_fnm, gis_d
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
        
        input_file = mp3du_json_nm+'.json'
        for constituent in constituent_list:
            run_mp3du(flag_run_mp3du, constituent, ptrk_calc_d, mp3du_executable, input_file)
        
        for constituent in constituent_list:
            p3doutput_json_nm = f'{constituent}_p3doutput'
            write_p3doutput_json_input(flag_write_p3doutput_json_input, constituent,
                                       ptrk_calc_d, p3doutput_json_nm
                                       )

        for constituent in constituent_list:
            p3doutput_json_nm = f'{constituent}_p3doutput'+'.json'
            run_writep3doutput(flag_run_writep3doutput, constituent, 
                               ptrk_calc_d, writep3doutput_executable,
                               p3doutput_json_nm
                               )

        generate_pathlines_map(flag_generate_pathlines_map, gis_d, fig_d, ptrk_calc_d, part_type)

        calc_relative_path_count(flag_calc_relative_path_count, gis_d, ptrk_calc_d)

        generate_relcount_pathlines_map(flag_relcount_pathlines_map, gis_d, fig_d, ptrk_calc_d, part_type)

    else:
        print('ecf calculations workflows NOT selected to run, \n check booleans...')

# this runs the main function
if __name__ == "__main__":
    main()
