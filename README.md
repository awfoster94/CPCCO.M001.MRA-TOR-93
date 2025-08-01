# CPCCO.M001.MRA-TOR-93
RCRA.Quarterly.2025\
Repository for ECF-200E-25-0027 

last updated: 05-16-2025\
by: awfoster94

IMPORTANT NOTE:

ArcGIS Pro Advanced 3D Analyst License will be needed to run some of the arcpy functions

Items 1. - 6. indicate where the ECF-200E-25-0027 calculations, output files, arcgis pro files and figures, are located

Item 7. copies all of the contents of this repository to a local folder

Items 8. - 10. are super general instructions to re-run the workflow 

Item 11. arcgis pro figures for the ECF-200E-25-0027 document

1. IMPORTANT\
   file locations for ECF-200E-25-0027
   
      a. all ECF-200E-25-0027 calculations are located in:\
         &emsp;i. \RCRA.Quarterly.2025\ECF-200E-25-0027_R0
    
      b. water table information used as inputs into the mp3du calculations are located in:\
         &emsp;i. \RCRA.Quarterly.2025\ECF-200E-25-0027_R0\ECF-HANFORD-24-0054_R0\
         &emsp;ii. \RCRA.Quarterly.2025\ECF-200E-25-0027_R0\QuarterlyReport_CY2024_Q1
    
      c. from inputs above, a mosaic is calculated using arcpy used as input into the mp3du calculation located in:\
         &emsp;i. \RCRA.Quarterly.2025\ECF-200E-25-0027_R0\CY2024Grid_SitewideWaterTable_March_Q1CY2024
    
      d. all mp3du calculations and output and interim gis processing is located in :\
         &emsp;i. \RCRA.Quarterly.2025\ECF-200E-25-0027_R0\Calcs
    
      e. all relevant gis output files for figures are located in:\
         &emsp;i. RCRA.Quarterly.2025\ECF-200E-25-0027_R0\Gis\
         &emsp;ii. "_projected" tag on filenames are the final projected data based on a wkt from SSPA
    
      f. some water table information for the 2020 mp3du results using trim is located in:\
         &emsp;i. \RCRA.Quarterly.2025\ECF-200E-25-0027_R0\CY2020_TRIM_WT_InterpFromGRDInput

      g. some waste site, facilities information is located in:\
         &emsp;i. \RCRA.Quarterly.2025\ECF-200E-25-0027_R0\200E_Sites

      h. some well network information from client in excel (standarized format) is located in:\
         &emsp;i. \RCRA.Quarterly.2025\ECF-200E-25-0027_R0\Well_Networks

      i. some geology information (dated 2020) for basalt and mud above 2020 water table is located in:\
         &emsp;i. \RCRA.Quarterly.2025\ECF-200E-25-0027_R0\Geology

      j. zip folders, if present are backups preserving random-seed within dispersion & respective calculation results

2. locations of previous 2020 SSPA calcs (prev_calcs) and INTERA rerun of 2020 SSPA calcs located in:\
      a. \prev_calcs\Calcs\
      b. \rerun_calcs\Calcs

3. locations of mp3du and writep3doutput fortran executables are located in:\
      a. bin\win

4. fresh set of ECF-200E-25-0027 input files is located in: \
       a. \source_files

5. comparison figures of the 2020 SSPA previous and rerun calcs with ECF-200E-25-0027 calcs & draft ECF figures is located in:\
       a. \figures

6. arcigs pro geographic information system (gis) data and figure files located in:\
       a. \gis_figures

7. clone this github repository to local directory of choice
   
   a. open git bash\
   b. run command, git clone https://github.com/INTERA-Inc/CPCCO.M001.MRA-TOR-49-6.git

8. install primary workflow virtual python environment
   
   a. open anaconda prompt\
   b. cd into the cloned directory with the .yml files & workflow.py\
   c. run command: conda env create -f mp3du-hanford-env.yml -n mp3du-hanford-env

9. install a clone of the arcgis pro python 3 virtual environment from your local installation
   
   a. https://pro.arcgis.com/en/pro-app/latest/arcpy/get-started/clone-an-environment.htm

10. run the workflow.py
   
   a. use an IDE (pycharm, spyder, vscode, notepad++, etc.) to open the cloned project folder and workflow.py to run each piece of the workflow as desired\
      &emsp;i. this will require flipping between two virtual environments: arcgis pro py3 arcpy env clone and the mp3du-hanford-env clone
   
   b. the workflow is controlled by programming booleans at the beginning of the script, denoted with "flag_xxx" in the script\
   c. to start all booleans are FALSE (off)
   
   d. there are two main forks:\
      &emsp;i. rerun SSPA 2020 calcs for confidence building\
      &emsp;i. INTERA 2024 calcs
   
   e. to operate within a given fork, turn the global boolean from FALSE (off) to TRUE (on)\
      &emsp;i. Use one fork at a time. When not using a fork, turn it to FALSE (off)
   
   f. within the fork of choice, incrementally turn local booleans from FALSE (off) to TRUE (on) to walk through functions ONE-BY-ONE\
      &emsp;i. this will require flipping between the mp3du-hanford-env virtual environment & the clone of arcgis pro py3 virtual environment to run arcpy functions\
      &emsp;ii. there are notes in each local boolean section to inform which environment to use when working through the script\
      &emsp;iii. after successful run of each boolean, turn from TRUE (on) to FALSE (off), and progress to the next one!

11. ECF document figures

   open the arcgis pro project document in the gis_figures folder, "200E_Pathlines_CY24_V5.aprx". This project document is linked to the projected shapefile results created from running all steps in the workflow.py and contains the figures for the ECF-200E-25-0027 document.
      
