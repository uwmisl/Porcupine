#!/usr/bin/env python
# coding: utf-8

# Data used to select which molbits to reorder from IDT

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
from scipy.stats import pearsonr

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

# In[3]:


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


# # Import training data

# In[4]:


training_run_names = ["08_09_2019_run_01",
                      "08_13_2019_run_02",
                      "08_13_2019_run_03",
                      "08_15_2019_run_02",
                      "08_15_2019_run_03",
                      "08_16_2019_run_01",]


# In[5]:


training_runs = run_spreadsheet[run_spreadsheet["run_name"].apply(lambda x: x in training_run_names)]


# In[9]:


train_sw = []
for i, run_data in training_runs.iterrows():
    run_name = run_data["run_name"]
    label_file = run_data["filtered_sw_labels"]
    
    sw = pd.read_csv(label_file, sep="\t", index_col=0)
    sw["run_name"] = run_name
    train_sw.append(sw)
train_sw = pd.concat(train_sw)


# # Import test sets

# In[10]:


test_run_names = ["08_28_2019_run_01",
                  "08_30_2019_run_01",]


# In[11]:


test_runs = run_spreadsheet[run_spreadsheet["run_name"].apply(lambda x: x in test_run_names)]


# In[12]:


test_sw = []
for i, run_data in test_runs.iterrows():
    run_name = run_data["run_name"]
    label_file = run_data["filtered_sw_labels"]
    
    sw = pd.read_csv(label_file, sep="\t", index_col=0)
    sw["run_name"] = run_name
    test_sw.append(sw)
test_sw = pd.concat(test_sw) 


# # Import sequences & compute features

# In[13]:


seqs_sheet_id = "1268399855"
seq_spreadsheet = import_gdrive_sheet(gdrive_key, seqs_sheet_id)
seq_spreadsheet.index = seq_spreadsheet["seq_no"]


# In[14]:


seq_df = seq_spreadsheet.copy()


# # Get train and test read counts

# In[16]:


train_molbit_counts = []
train_norms = []
norm_train_molbit_counts = []
train_bits = []
for set_i, start_molbit in enumerate(range(0, 96, 16)):
    run_data = dict(training_runs[training_runs["molbit_set"] == set_i].iloc[0, :])
    run_name = run_data["run_name"]
        
    for half in [0, 1]:
        if half == 0:
            molbits_in_range = range(start_molbit, start_molbit+8)
        else:
            molbits_in_range = range(start_molbit+8, start_molbit+16)

        labels_from_run = train_sw[train_sw["run_name"] == run_name]
        filtered_labels_in_run = labels_from_run[labels_from_run["molbit"].apply(lambda x: x in molbits_in_range)]
        l, c = np.unique(filtered_labels_in_run["molbit"], return_counts=True)
        for l_, c_ in zip(l, c):
            train_molbit_counts.append(c_)
            norm_train_molbit_counts.append(c_ / sum(c))
        train_norms.append(sum(c))
        train_bits.append(molbits_in_range)


# In[17]:


test_molbit_counts = []
test_norms = []
norm_test_molbit_counts = []
test_bits = []
for set_i, start_molbit in enumerate(range(0, 96, 16)):
    for half in [0, 1]:
        if half == 0:
            run_name = test_run_names[0]
            molbits_in_range = range(start_molbit, start_molbit+8)
        else:
            run_name = test_run_names[1]
            molbits_in_range = range(start_molbit+8, start_molbit+16)
        test_sw_run = test_sw[test_sw["run_name"] == run_name]
        filtered_labels_in_run = test_sw_run[test_sw_run["molbit"].apply(lambda x: x in molbits_in_range)]
        l, c = np.unique(filtered_labels_in_run["molbit"], return_counts=True)
        for l_, c_ in zip(l, c):
            test_molbit_counts.append(c_)
            norm_test_molbit_counts.append(c_ / sum(c))
        test_norms.append(sum(c))
        test_bits.append(molbits_in_range)
    


# ## Add counts to df

# In[18]:


seq_df["norm_counts_train"] = norm_train_molbit_counts
seq_df["norm_counts_test"] = norm_test_molbit_counts
seq_df["counts_train"] = train_molbit_counts
seq_df["counts_test"] = test_molbit_counts


# In[23]:


seq_df.sort_values(by="norm_counts_test")


# In[ ]:




