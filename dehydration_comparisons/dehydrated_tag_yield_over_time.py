#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import h5py
import numpy as np
import pandas as pd
import csv
import re
from matplotlib import pyplot as plt
import seaborn as sns
from io import StringIO
sns.set(font_scale=1.9, style="white")


# # Import data
# 
# This is all for the MISL tag.
# 
# ## Dehydrated & sequenced immediately
f5_dir_short = "/path/to/data/MinION_sequencing_data_20191216/12_16_19_run_01/12_16_19_run_01/20191216_1931_MN25769_FAL22488_f420bcc0/fast5"
# In[2]:


seq_summary_short = "/path/to/data/MinION_sequencing_data_20191216/guppy_3.2.2_12_16_19_run_01_exec_20200107/sequencing_summary.txt"


# In[3]:


f5_meta_short = pd.read_csv(seq_summary_short, header=0, sep="\t")

d = f5_dir_short
metadata = []
for f5_fname in os.listdir(d):
    with h5py.File(os.path.join(d, f5_fname), "r") as f5:
        for grp_name, grp in f5.items():
            a = dict(grp.get("Raw").attrs)
            read_id = a.get("read_id")
            start_time = a.get("start_time")
            start_time_s = start_time / 4000.
            metadata.append((read_id, start_time, start_time_s))
            
dhd_short = pd.DataFrame(metadata, columns=["read_id", "start_time", "start_time_s"])
# ## Fresh
f5_dir_fresh = "/path/to/data/MinION_sequencing_data_20191011/10_11_19_run_02/10_11_19_run_02/20191011_2256_MN25769_FAL01586_63c1565a"
# In[4]:


seq_summary_fresh = "/path/to/data/MinION_sequencing_data_20191011/guppy_3.2.2_10_11_19_run_02_exec_20191014/sequencing_summary.txt"


# In[5]:


f5_meta_fresh = pd.read_csv(seq_summary_fresh, header=0, sep="\t")

d = f5_dir_fresh
metadata = []
for dirname in os.listdir(d):
    if "fast5" in dirname:  # in ["pass", "fail"]: #and "fail" not in dirname:
        for f5_fname in os.listdir(os.path.join(d, dirname)):
            print(os.path.join(d, dirname, f5_fname))
            try:
                with h5py.File(os.path.join(d, dirname, f5_fname), "r") as f5:
                    for grp_name, grp in f5.items():
                        a = dict(grp.get("Raw").attrs)
                        read_id = a.get("read_id")
                        start_time = a.get("start_time")
                        start_time_s = start_time / 4000.
                        metadata.append((read_id, start_time, start_time_s))
            except OSError:
                print("OSError")
            
dhd_fresh = pd.DataFrame(metadata, columns=["read_id", "start_time", "start_time_s"])
# ## Dehydrated & sequenced after 4 weeks

# In[13]:


seq_summary_long = "/path/to/data/MinION_sequencing_data_20200121/guppy_3.2.2_01_21_20_run_01_exec_20200121/sequencing_summary.txt"


# In[15]:


f5_meta_long = pd.read_csv(seq_summary_long, header=0, sep="\t")

d = f5_dir_long
metadata = []
for f5_fname in os.listdir(d):
    with h5py.File(os.path.join(d, f5_fname), "r") as f5:
        for grp_name, grp in f5.items():
            a = dict(grp.get("Raw").attrs)
            read_id = a.get("read_id")
            start_time = a.get("start_time")
            start_time_s = start_time / 4000.
            metadata.append((read_id, start_time, start_time_s))
            
dhd_long = pd.DataFrame(metadata, columns=["read_id", "start_time", "start_time_s"])
# # Accumulate reads over time

# ## Dehydrated & sequenced immediately

# In[16]:


dhd = f5_meta_short.copy()
reads_over_time_bins = []
offset = 200  # Delay in sample loading
bins = np.arange(0, max(dhd["start_time"]), 250)

for bin_end in bins:
    n_reads_so_far = sum(dhd["start_time"] < bin_end + offset)
    reads_over_time_bins.append(n_reads_so_far)
plt.plot(bins, reads_over_time_bins)
# plt.axvline(50)
# plt.axvline(100)
# plt.axvline(150)
# plt.axvline(200)
# plt.axvline(250)
plt.xlim([0, 500])
plt.ylim([0, 5000])
reads_over_time_short = zip(bins, reads_over_time_bins)


# ## Fresh

# In[17]:


dhd = f5_meta_fresh.copy()
reads_over_time_bins = []
bins = np.arange(0, max(dhd["start_time"]), 250)
for bin_end in bins:
    n_reads_so_far = sum(dhd["start_time"] < bin_end)
    reads_over_time_bins.append(n_reads_so_far)
plt.plot(bins, reads_over_time_bins)
reads_over_time_fresh = zip(bins, reads_over_time_bins)


# ## 4 weeks

# In[18]:


dhd = f5_meta_long.copy()
reads_over_time_bins = []
bins = np.arange(0, max(dhd["start_time"]), 250)
for bin_end in bins:
    n_reads_so_far = sum(dhd["start_time"] < bin_end)
    reads_over_time_bins.append(n_reads_so_far)
plt.plot(bins, reads_over_time_bins)
reads_over_time_long = zip(bins, reads_over_time_bins)


# # Plots

# In[19]:


bins_mins = [x / 60. for x in bins]


# In[20]:


data = {
    "short": {"binned_data": reads_over_time_short,
              "plot_label": "Dehydrated tag (0 days)",
              "color": "tab:green",
              "offset": 250},
    "long": {"binned_data": reads_over_time_long,
              "plot_label": "Dehydrated tag (4 weeks)",
              "color": "tab:blue",
              "offset": 0},
    "fresh": {"binned_data": reads_over_time_fresh,
              "plot_label": "Fresh tag",
              "color": "tab:orange",
              "offset": 0},
}


# In[21]:


fig, ax = plt.subplots(figsize=(8, 6))

for d in data.values():
    bins, reads = zip(*d.get("binned_data"))
    bins_mins = [x / 60. for x in bins]
    
    ax.plot(bins_mins, reads, color=d.get("color"), label=d.get("plot_label"))
    
ax.set_xlabel("Time (mins)")
ax.set_ylabel("Cumulative reads")
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()
fig.savefig("../plots/dehydrated_comparisons/yield.svg")
fig.savefig("../plots/dehydrated_comparisons/yield.png", dpi=300)


# In[ ]:




