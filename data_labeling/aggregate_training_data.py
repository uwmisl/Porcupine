#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import h5py
import numpy as np
import pandas as pd
import logging
import re

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale=1.8, style="whitegrid")


# In[2]:


logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s - %(name)s] %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG,
                        handlers=[logging.StreamHandler()])


# # Import run settings

# In[6]:


def import_gdrive_sheet(gdrive_key, sheet_id):
    run_spreadsheet = pd.read_csv("https://docs.google.com/spreadsheet/ccc?key=" +                                   gdrive_key + "&output=csv&gid=" + sheet_id)
    if "date" in run_spreadsheet.columns:
        run_spreadsheet["date"] = run_spreadsheet["date"].astype(str)
    return run_spreadsheet

gdrive_key = "gsheet_id_here"
sheet_id = "0"
set_sheet_id = "512509543"

run_spreadsheet = import_gdrive_sheet(gdrive_key, sheet_id)
set_spreadsheet = import_gdrive_sheet(gdrive_key, set_sheet_id)


# In[7]:


molbit_file = "../porcupine_sequences.fa"


# In[10]:


training_run_names = ["08_09_2019_run_01",
                      "08_13_2019_run_02",
                      "08_13_2019_run_03",
                      "08_15_2019_run_02",
                      "08_15_2019_run_03",
                      "08_16_2019_run_01",]


# In[13]:


training_runs = run_spreadsheet[run_spreadsheet["run_name"].apply(lambda x: x in training_run_names)]


# In[14]:


training_runs


# ## Define which molbits are in each set

# In[15]:


# Create set_N variables based on spreadsheet
molbit_sets = {}
for ix, row in set_spreadsheet.iterrows():
    set_no = re.findall(r"set ([\d]+)", row["set"])[0]
    molbits = row["molbits_in_set"]
    molbit_sets[set_no] = molbits.split(", ")


# ## Specify which molbits are present in each run

# In[20]:


molbits_by_run = {}
for i, run_data in training_runs.iterrows():
    run_name = run_data["run_name"]
    molbits_by_run[run_name] = {}
    print(run_name)
    
    molbit_set_in_run = str(run_data.get("molbit_set"))
    molbit_sets_on_flowcell = run_data.get("prev_on_flowcell")

    molbits_in_run = molbit_sets[molbit_set_in_run]
    molbits_on_flowcell = molbits_in_run[:]
    if molbit_sets_on_flowcell != "none":
        molbit_sets_on_flowcell = molbit_sets_on_flowcell.split(", ")
        for m in molbit_sets_on_flowcell:
            molbits_on_flowcell.extend(molbit_sets[m])
    molbits_by_run[run_name]["molbits_in_run"] = molbits_in_run
    molbits_by_run[run_name]["molbits_on_flowcell"] = molbits_on_flowcell


# ## Create new file with only the reads/molbits we want to use for training

# In[40]:


for i, run_data in training_runs.iterrows():
    run_name = run_data["run_name"]
    molbits_in_run = molbits_by_run[run_name]["molbits_in_run"]
    sw_calls_file = run_data["sw_calls_file"]
    sw = pd.read_csv(sw_calls_file, sep="\t", index_col=0)
    
    sw_scores = sw.filter(regex="score_molbit_.*")
    best_molbits = sw_scores.apply(np.argmax, axis=1)
    accept_molbit = sw.lookup(sw.index, best_molbits) >= 15
    sw["best_molbit"] = best_molbits.str.extract(r"score_molbit_([\d]+)")
    sw["accept_molbit"] = accept_molbit
    sw["best_molbit_is_in_run"] = sw["best_molbit"].apply(lambda x: x in molbits_in_run)
    
    use_for_training = sw[np.logical_and(sw["accept_molbit"], sw["best_molbit_is_in_run"])]
    
    training_calls_file = sw_calls_file.replace("all", "filtered_molbits_in_run")
    print(training_calls_file)
    
    best_score = np.max(use_for_training[[f"sw_score_molbit_{molbit}" for molbit in molbits_in_run]], axis=1)
    use_for_training["best_score"] = best_score

    use_for_training = use_for_training.loc[:, ["best_molbit", "best_score"]]
    use_for_training.columns = ["molbit", "sw_score"]
    
    use_for_training.to_csv(training_calls_file, sep="\t", index=True)


# In[42]:


use_for_training

