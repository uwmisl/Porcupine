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
sns.set(font_scale=1.5, style="whitegrid")


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


# In[6]:


train_sw = []
for i, run_data in training_runs.iterrows():
    run_name = run_data["run_name"]
    label_file = run_data["filtered_sw_labels"]
    
    sw = pd.read_csv(label_file, sep="\t", index_col=0)
    sw["run_name"] = run_name
    train_sw.append(sw)
train_sw = pd.concat(train_sw)


# # Import test sets

# In[7]:


test_run_names = ["08_28_2019_run_01",
                  "08_30_2019_run_01",]


# In[8]:


test_runs = run_spreadsheet[run_spreadsheet["run_name"].apply(lambda x: x in test_run_names)]


# In[9]:


test_sw = []
test_cnn = []
for i, run_data in test_runs.iterrows():
    run_name = run_data["run_name"]
    sw_label_file = run_data["filtered_sw_labels"]
    
    sw = pd.read_csv(sw_label_file, sep="\t", index_col=0)
    sw["run_name"] = run_name
    test_sw.append(sw)
    
    cnn_label_file = run_data["model_v4_0_1_labeled_preds"]
test_sw = pd.concat(test_sw)


# # Get train and test read counts

# In[10]:


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


# In[11]:


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
    


# In[12]:


def plot_read_counts_mod(labels, counts,
                     possible_labels=None, ax=None, vmax=None, title_note=None):
    from matplotlib.patches import Patch
    if vmax is None:
        vmax = max(counts) + max(counts)
    if ax == None:
        fig, ax = plt.subplots(figsize=(34, 8))
    if possible_labels is None:
        possible_labels = labels[:]
    g = sns.barplot(x=labels, y=counts, order=possible_labels, ax=ax)
    title = "#/reads identified per barcode"
    if title_note is not None:
        title += "\n(%s)" % title_note
    ax.set_title(title)
    ax.set_xlabel("Barcode ID")
    ax.set_ylabel("Read counts")
    ax.set_ylim([0, vmax])
    
    prev_text_height = 0
    for j, label in enumerate(possible_labels):
        if label in labels:
            count = counts[labels.index(label)]
        else:
            count = 0
        if label == "-1":
            continue

#         if label in labels_in_run:
#             g.containers[0].get_children()[j].set_facecolor("tab:red")
#             font_kwargs = {"color": "tab:red", "weight": "bold"} 
#         elif label in labels_on_flowcell:
#             g.containers[0].get_children()[j].set_facecolor("tab:blue")
#             font_kwargs = {"color": "k", "weight": "bold"}
#         else:
        g.containers[0].get_children()[j].set_facecolor("k")
        font_kwargs = {"color": "k", "weight": "normal"}
            
        diff = prev_text_height - (count + 0.01 * vmax)
        if int(count) == count:
            count_fmt = f"{count}"
        else:
            count_fmt = f"{count:.3f}"
        if count < 100:
            text_height = count + .01 * vmax
            ax.text(j, text_height, count_fmt, ha="center", **font_kwargs)
        elif diff < 0 and np.abs(diff) < vmax * .06:
            text_height = np.max([prev_text_height + .035 * vmax, count + .01 * vmax])
            ax.text(j, text_height, count_fmt, ha="center", **font_kwargs)
        elif np.abs(diff) < vmax * .05:
            text_height = np.min([prev_text_height - .01 * vmax, count + .01 * vmax])
            ax.text(j, text_height, count_fmt, ha="center", **font_kwargs)
        else:
            text_height = count + .01 * vmax
            ax.text(j, text_height, count_fmt, ha="center", **font_kwargs)
        prev_text_height = text_height
        
#     legend_elements = [Patch(facecolor='k', edgecolor='k',
#                          label='never been run on this flowcell'),
#                        Patch(facecolor='tab:blue', edgecolor='tab:blue',
#                          label='prev run on flowcell'),
#                        Patch(facecolor='tab:red', edgecolor='tab:red',
#                          label='current run on flowcell')]
#     leg = ax.legend(handles=legend_elements)
#     t1, t2, t3 = leg.get_texts()
#     t2._fontproperties = t1._fontproperties.copy()
#     t3._fontproperties = t1._fontproperties.copy()
#     t2.set_weight('bold')
#     t3.set_weight('bold')
#     t3.set_color("tab:red")
    return ax


# In[14]:


ax = plot_read_counts_mod([str(x) for x in range(96)], test_molbit_counts, possible_labels=[str(x) for x in range(96)])
ax.set_ylim([0, 40000])


# In[15]:


ax = plot_read_counts_mod([str(x) for x in range(96)], norm_test_molbit_counts, possible_labels=[str(x) for x in range(96)])
ax.set_ylim([0, 0.6])


# # Combine read counts from the two test runs and re-normalize

# In[16]:


norm_counts_test = np.array(norm_test_molbit_counts) / sum(norm_test_molbit_counts)


# In[17]:


ax = plot_read_counts_mod([str(x) for x in range(96)], norm_counts_test, possible_labels=[str(x) for x in range(96)])
ax.set_ylim([0, 0.05])


# # Do sampling of the test dataset to develop the scaling factor

# In[84]:


n_samples = 1000
sample_depth = 10000


# In[85]:


def get_read_counts(labels, possible_labels=[]):
    labels = list(labels)

    for i, label in enumerate(labels):
        if label not in possible_labels:
            labels[i] = "-1"
    labels = np.array(labels, dtype=np.array(possible_labels).dtype)
    labels, counts = np.unique(labels, return_counts=True)
    ordered_counts = np.zeros(len(possible_labels), dtype=int)
    for i, possible_label in enumerate(possible_labels):
        ix = np.argwhere(labels == possible_label)
        if len(ix) > 0:
            ix = ix[0][0]
            ordered_counts[i] = counts[ix]
            assert labels[ix] == possible_label
            ordered_counts[i] = counts[ix]
    return ordered_counts


# In[86]:


sample_record = np.zeros((n_samples, 96))
for sample_i in range(n_samples):
    sampled_reads = np.random.choice(list(range(96)), p=norm_counts_test/sum(norm_counts_test), size=sample_depth)
    sampled_read_counts = get_read_counts(sampled_reads, possible_labels=list(range(96)))
    sampled_frac_read_counts = sampled_read_counts / sum(sampled_read_counts)
    sample_record[sample_i, :] = sampled_frac_read_counts


# In[87]:


sample_record.shape


# In[88]:


scaling_factor = 1 / np.sum(sample_record, axis=0) * sample_depth


# In[89]:


scaling_factor

scaling_factor = 1 / norm_counts_test
# In[90]:


scaled_test_counts = [int(scale * count) for scale, count in zip(scaling_factor, test_molbit_counts)] / (sum(test_molbit_counts))


# In[91]:


ax = plot_read_counts_mod([str(x) for x in range(96)], scaled_test_counts, possible_labels=[str(x) for x in range(96)])
# ax.set_ylim([0, 1])


# In[92]:


scaled_train_counts = [int(scale * count) for scale, count in zip(scaling_factor, train_molbit_counts)] / sum(train_molbit_counts)


# In[93]:


ax = plot_read_counts_mod([str(x) for x in range(96)], scaled_train_counts, possible_labels=[str(x) for x in range(96)])
# ax.set_ylim([0, 4])


# In[94]:


u = np.mean(scaled_train_counts)
s = np.std(scaled_train_counts)
print(f"{u} +/- {s}")
print(f"variability: {s / u * 100:0.3f}%")


# In[95]:


ax = plot_read_counts_mod([str(x) for x in range(96)], train_molbit_counts, possible_labels=[str(x) for x in range(96)])
ax.set_ylim([0, 76000])


# In[96]:


u = np.mean(train_molbit_counts)
s = np.std(train_molbit_counts)
print(f"{u} +/- {s}")
print(f"variability: {s / u * 100:0.3f}%")


# In[ ]:




