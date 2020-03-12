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
from scipy.stats import pearsonr

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale=1.7, style="white")


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


# # Import cnn & basecalling data

# In[4]:


run_to_include = ["08_09_2019_run_01",
                  "08_13_2019_run_02",
                  "08_13_2019_run_03",
                  "08_15_2019_run_02",
                  "08_15_2019_run_03",
                  "08_16_2019_run_01",
                  "08_28_2019_run_01",
                  "08_30_2019_run_01",
                  "09_05_2019_run_02"]


# In[5]:


runs = run_spreadsheet[run_spreadsheet["run_name"].apply(lambda x: x in run_to_include)]


# In[6]:


label_df = []
for i, run_data in runs.iterrows():
    run_name = run_data["run_name"]
    label_file = run_data["model_v4_0_1_labeled_preds"]
    
    label_df_ = pd.read_csv(label_file, sep="\t", index_col=0)
    label_df_["run_name"] = run_name
    label_df.append(label_df_)
label_df = pd.concat(label_df)


# In[10]:


all_df = []
for i, run_data in runs.iterrows():
    run_name = run_data["run_name"]
    label_file = run_data["model_v4_0_1_all_preds"]
    
    all_df_ = pd.read_csv(label_file, sep="\t", index_col=0)
    all_df_["run_name"] = run_name
    all_df.append(all_df_)
all_df = pd.concat(all_df)


# In[11]:


all_df


# In[12]:


unlabeled_df = all_df.drop(label_df.index)


# In[13]:


unlabeled_df


# # Get cnn & basecalling read counts

# In[7]:


label_df


# In[9]:


np.unique(label_df["sw_label"])


# In[8]:


sum(label_df["sw_label"] == -1)


# In[15]:


# counts_cnn = []
counts_sw = []
for run_name, df in label_df.groupby("run_name"):
    print(run_name)
#     l_cnn, c_cnn = np.unique(df["cnn_label"], return_counts=True)
    l_sw, c_sw = np.unique(df["sw_label"], return_counts=True)
#     l_cnn = list(l_cnn)
    l_sw = list(l_sw)
    
    for molbit in range(96):
#         try:
#             i_cnn = l_cnn.index(molbit)
#             c = c_cnn[i_cnn]
#         except ValueError:
#             c = 0 
#         counts_cnn.append(c)
        
        try:
            i_sw = l_sw.index(molbit)
            c = c_sw[i_sw]
        except ValueError:
            c = 0
        counts_sw.append(c)
    


# In[16]:


counts_cnn = []
# counts_sw = []
for run_name, df in unlabeled_df.groupby("run_name"):
    print(run_name)
    l_cnn, c_cnn = np.unique(df["cnn_label"], return_counts=True)
    l_cnn = list(l_cnn)
    
    for molbit in range(96):
        try:
            i_cnn = l_cnn.index(molbit)
            c = c_cnn[i_cnn]
        except ValueError:
            c = 0 
        counts_cnn.append(c)
    
    


# In[17]:


max(counts_sw)


# In[29]:


fig, ax = plt.subplots(figsize=(6, 5))
edge = 78000
plt.plot([0, edge], [0, edge], linewidth=0.5, color="k", alpha=0.5)
ax.scatter(counts_sw, counts_cnn, c="k", marker=".")
ax.set_xlabel("Molbit counts (basecalling+alignment)")
ax.set_ylabel("Molbit counts (CNN)")
r, p = pearsonr(counts_sw, counts_cnn)
ax.text(1000, 60000, f"Pearson r={r:.4f}, p<{p:.4f}1", fontsize=14)
ax.set_xlim([0, edge])
ax.set_ylim([0, edge])
ax.set_title("Comparison of molbit counts from\nalignment (sequence data)\nvs. CNN (signal data)")
plt.tight_layout()
plt.savefig("../v4/plots/read_counts_seq_vs_squig.svg")


# In[19]:


r


# In[21]:


p


# In[20]:


ax.get_ylim()[1]


# In[ ]:




