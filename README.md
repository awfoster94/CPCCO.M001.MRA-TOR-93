# CPCCO.M001.MRA-TOR-93
Repository for ECF-200ZP1-25-0092
Data Gap & Redundancy Analysis

last updated: 09-04-2025\
by: awfoster94

ReadMe Section for workflow_Smw_Sew_Scs.py\

1. clone this github repository to local directory of choice
   
   a. open git bash\
   b. run command, git clone https://github.com/awfoster94/CPCCO.M001.MRA-TOR-93.git


2. To calculate Smw, Sew, and Sp-cs scores, install the following virtual python environment
   
   a. open miniforge or mamba prompt\
   b. cd into the cloned directory with the .yml file & workflow.py\
   c. run command: conda env create -f mp3du-hanford-env.yml -n mp3du-hanford-env

3. To calculate Smw, Sew, and Sp-cs, run the workflow_Smw_Sew_Scs.py\
   
   a. use an IDE (pycharm, spyder, vscode, notepad++, etc.) to open the cloned project folder and workflow_Smw_Sew_Scs.py to run each piece of the workflow as desired\
      &emsp;i. this will require using (1) virtual environment: mp3du-hanford-env\
   b. copy in flow and transport folders into the source_files folder (too large for github)\
   c. copy in qryAWLN_SSPA_TO93.txt & qryAWLN2021_Present_TO93.txt into the HEIS_Data_Pull folder in gis -> xlsx -> HEIS_Data_Pull (too large for github)\
   d. the workflow is controlled by programming booleans at the beginning of the script, denoted with "flag_xxx" at the beginning of the script\
   e. to start all booleans are FALSE (off)\
   f. to run, sequentially turn each boolean to TRUE (on) to run functions in the main()\
   g. once successfully run, turn boolean flags to FALSE (off) before incrementing to the next function\
   h. additional instructions are included as comments in workflow_Smw_Sew_Scs.py on how to run certain functions\

4. After completion of workflow_Smw_Sew_Scs.py\
   a. review scores calculated wrapped up as shapefiles for each component for each potential well locations in gis -> shp -> scores\
   b. calc scores are zipped up in scores_08262025.zip in the same folder for comparison\
   c. review exported figures in figs folder\
   d. figs summarizing calculations and supporting shapefiles are zipped up in figs_08262025.zip\

ReadMe Section for combine_scors_Smww_Sew_Scs_Smik_Scov_Sexcee.py\

5. Calculate Combined Total Data Gap Scores for the UU/MU & LU/CR for hex-chrome, tec99, and carbon tetrachloride\
   a. check to make sure mp3du-hanford-env has been installed as a virtual environment to run script\
   b. check to make sure the following csv's have been generated from other workflows:\
      i.   scores_detailed_Smw_Sew_Scs.csv\
      ii.  scores_only_Smw_Sew_Scs.csv\
      iii. scores_detailed_Smik_Scov_Sexcee.csv\
      iv. scores_only_Smike_Scov_Sexcee.csv\
   c. run combine_scores_SMw_Sew_Scs_Smik_Scov_Sexcee.py\
   d. outputs generated include csv's and shapefiles with detailed supporting information and also for scores only\
      
