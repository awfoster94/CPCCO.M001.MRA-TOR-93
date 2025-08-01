# CPCCO.M001.MRA-TOR-93
Repository for ECF-200ZP1-22-0098
Data Gap & Redundancy Analysis

last updated: 07-31-2025\
by: awfoster94

1. clone this github repository to local directory of choice
   
   a. open git bash\
   b. run command, git clone https://github.com/awfoster94/CPCCO.M001.MRA-TOR-93.git

2. install primary workflow virtual python environment
   
   a. open anaconda prompt\
   b. cd into the cloned directory with the .yml file & workflow.py\
   c. run command: conda env create -f mp3du-hanford-env.yml -n mp3du-hanford-env

3. run the workflow.py
   
   a. use an IDE (pycharm, spyder, vscode, notepad++, etc.) to open the cloned project folder and workflow.py to run each piece of the workflow as desired\
      &emsp;i. this will require using (1) virtual environment: mp3du-hanford-env
   
   b. the workflow is controlled by programming booleans at the beginning of the script, denoted with "flag_xxx" in the script\
   c. to start all booleans are FALSE (off)
   d. to run, sequentially turn each boolean to TRUE (on) to run functions in the main()
   e. once successfully run, turn boolean flags to FALSE (off) before incrementing to the next function
   
      
