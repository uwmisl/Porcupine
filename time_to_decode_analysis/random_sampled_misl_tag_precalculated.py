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
from subprocess import check_output

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2, style="white")

import dask.bag as db
from dask.diagnostics import ProgressBar
ProgressBar().register()


# In[2]:


logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s - %(name)s] %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG,
                        handlers=[logging.StreamHandler()])


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


# In[4]:


run_spreadsheet.columns


# In[5]:


set_spreadsheet.columns


# In[6]:


set_spreadsheet


# # Import test sets
# 
# * `label_df`: Dataframe containing read_ids, labels, scores, and run_names
# * `molbit_sets`: Dict: set id (str, usually a number) -> list of molbit ids
# * `molbits_by_run`: Dict: run_name -> {molbits_in_run and molbits_on_flowcell}

# In[7]:


spreadsheet_col_name = "model_v4_0_1_all_preds"
test_run_names = ["10_11_19_run_02"]
test_runs = run_spreadsheet[run_spreadsheet["run_name"].apply(lambda x: x in test_run_names)]

label_df = []
for i, run_data in test_runs.iterrows():
    run_name = run_data["run_name"]
    label_file = run_data[spreadsheet_col_name]
    
    cnn_df = pd.read_csv(label_file, sep="\t", index_col=0)
    cnn_df["run_name"] = run_name
    label_df.append(cnn_df)
label_df = pd.concat(label_df)


# In[8]:


label_df


# ## Define which molbits are in each set

# In[9]:


# Create set_N variables based on spreadsheet
molbit_sets = {}
for ix, row in set_spreadsheet.iterrows():
    set_no = re.findall(r"set ([\d]+)", row["set"])[0]
    molbits = row["molbits_in_set"]
    molbit_sets[set_no] = molbits.split(", ")


# In[10]:


molbit_sets


# ## Specify which molbits are present in each run

# In[12]:


molbits_by_run = {}
for i, run_data in test_runs.iterrows():
    run_name = run_data["run_name"]
    molbits_by_run[run_name] = {}
    print(run_name)
    
    molbit_set_in_run = str(int(run_data.get("molbit_set")))
    molbit_sets_on_flowcell = run_data.get("prev_on_flowcell")

    molbits_in_run = molbit_sets[molbit_set_in_run]
    molbits_on_flowcell = molbits_in_run[:]
    if molbit_sets_on_flowcell != "none":
        molbit_sets_on_flowcell = molbit_sets_on_flowcell.split(", ")
        for m in molbit_sets_on_flowcell:
            molbits_on_flowcell.extend(molbit_sets[m])
    molbits_by_run[run_name]["molbits_in_run"] = molbits_in_run
    molbits_by_run[run_name]["molbits_on_flowcell"] = molbits_on_flowcell


# In[13]:


molbits_by_run


# # Define helper functions for sampling & counting errors
# 
# * `count_bit_errors(actual, predicted)`: Get a bitwise diff of the two tags and the # of bit errors
# * `get_read_counts(labels, possible_labels=[])`: Get the # of each label, only for the labels in possible_labels.
# * `get_tag(read_counts, t=0)`: Given an ordered list of read counts, apply the threshold t to create a bit string.
# * `mask_tag(tag, mask)`: Use this to ignore bits that were previously on the flowcell.
# * `find_optimal_threshold(read_counts, actual_tag, step=1)`: Find the threshold that produces the closest tag to the actual tag.
# * `get_sample_depths(n_reads)`: Get a list of #/reads we should sample down to, spaced nicely for the number of reads overall.

# In[14]:


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

def get_tag(read_counts, t=0):
    read_counts = np.array(read_counts)
    tag = list(np.where(read_counts > t, 1, 0))
    return tag

def get_sample_depths_logspace(n_reads):
    n_points = np.log(n_reads) * 3
    sample_depths = list(np.geomspace(30, n_reads, num=n_points, dtype=int))
    return sample_depths

def remap_molbits_random(old_codeword, seed=0):
    old_ones = np.where(np.array(old_codeword) == 1)[0]
    np.random.seed(seed)
    new_ones = np.random.choice(old_ones, size=len(old_ones), replace=False)
    old_zeros = np.where(np.array(old_codeword) == 0)[0]
    new_zeros = np.random.choice(old_zeros, size=len(old_zeros), replace=False)
    
    molbit_map = {}
    for old, new in zip(old_ones, new_ones):
        molbit_map[int(old)] = int(new)
    for old, new in zip(old_zeros, new_zeros):
        molbit_map[int(old)] = int(new)
    return molbit_map

def remap_molbits(old_codeword, new_codeword, seed=0):
    assert sum(old_codeword) == sum(new_codeword)
    np.random.seed(seed)
    old_ones = np.where(np.array(old_codeword) == 1)[0]
    new_order_ones = list(np.random.choice(old_ones, size=len(old_ones), replace=False))
    old_zeros = np.where(np.array(old_codeword) == 0)[0]
    new_order_zeros = list(np.random.choice(old_zeros, size=len(old_zeros), replace=False))
    
    molbit_map = {}
    for new_bit_i, bit in enumerate(new_codeword):
        if bit == 1:
            one_bit = new_order_ones.pop()
            molbit_map[one_bit] = new_bit_i
        else:
            zero_bit = new_order_zeros.pop()
            molbit_map[zero_bit] = new_bit_i
    assert len(new_order_zeros) == 0
    assert len(new_order_ones) == 0
        
    return molbit_map


# # Decoding functions

# In[15]:


def decode_c(received_codeword):
    result = check_output(["../ecc/decoder", received_codeword]).decode("utf-8").split("\n")
    codeword_distance = int(re.findall(r"\"distance\": ([\d]+)", result[2])[0])
    corrected_message = re.findall(r"\"message\": \"([\d]+)\"", result[1])[0]
    return corrected_message, codeword_distance

def compute_decoding_helper(in_data, generator_matrix_file=""):
    read_counts, depth, sample_i, correct_message = in_data
    results = []
    thresholds = list(np.sort(np.unique(read_counts)))
    if len(thresholds) > 2:
        thresholds = thresholds[:-2]
    for t in thresholds:  # 1-bits set by counts > t (not >= t)
        codeword_at_t = get_tag(read_counts, t=t)
        codeword_str = "".join([str(x) for x in codeword_at_t])
        closest_msg, closest_d = decode_c(codeword_str)
        correct_decoding = False
        if closest_msg is not None:
            closest_msg_ints = np.array([int(x) for x in closest_msg])
            if sum(correct_message - closest_msg_ints) == 0:
                correct_decoding = True
        codeword_at_t = "".join([str(x) for x in codeword_at_t])
        results.append((depth, sample_i, correct_decoding, codeword_at_t, closest_msg, closest_d, t))
    return results


# # Sample

# In[16]:


def rescale_counts(read_counts, scaling_factors):
    rescaled_counts = np.multiply(np.array(read_counts), scaling_factors)
    rescaled_counts = np.ceil(rescaled_counts / sum(rescaled_counts) * sum(read_counts))
    return rescaled_counts


# In[17]:


scaling_factors = [117.44079692,  296.08219178,   79.3902663 ,   63.8680128 ,
        301.24041812,  106.23305345,   50.35782934,   94.36710933,
        261.39458779,   23.42805573,  236.19903327,  215.71332122,
         72.68457433, 1674.82258065,  359.61558442,   92.43485034,
         55.15762106,  147.49710313,  161.68942521,   41.8235584 ,
         72.38555587,  124.39775226,  207.99019608,  599.71731449,
        410.15625   ,  146.23955432,   81.21546961,  151.60891089,
        265.91895803,   93.01442673,   59.58171206,   41.92334018,
         75.73638033,  100.18461538,  178.88385542,  176.9227836 ,
         35.15      ,   99.06164932,  435.15123095,  124.01737387,
        100.70515917,  113.01108647,  127.24327323,   34.53376496,
        113.68327138,   86.11075652,  317.00898411,  239.53629243,
         83.78780013,  276.0384821 ,   89.75808133,   32.18069662,
        250.71262136,  310.93798916,   76.84392204,  187.19391084,
        211.31315136,  165.0372093 ,   71.34651475,  403.21590909,
         35.59571978,  201.41721854,  126.01242971,   66.43719769,
       1425.49333333,  102.0477251 ,   39.45092251,   84.89571202,
         68.85702018,  148.00922935,  204.68155712,  104.81568627,
         66.45394046,  150.09968354,   32.68883529,   74.21318208,
        797.16806723,   93.0257416 ,  348.76102941,  372.37684004,
         95.12844828,   56.96902426,  143.82404692,  231.58237146,
        171.5491644 ,   65.69370442,   68.64634526,  119.36073553,
        128.91764706,   32.27093687,  114.79353994,  433.62242374,
         92.13242249,  293.19063545,  129.10751105,   86.49629995]
scaling_factors = np.array(scaling_factors)


# In[18]:


generator_matrix_file = "generator_matrix_20190924.txt"


# In[19]:


possible_labels_str = [str(x) for x in range(96)]
possible_labels = list(range(96))


# In[20]:


sample_depths = get_sample_depths_logspace(100000)


# In[ ]:


n_codewords_per_test_run = 3
n_samples = 10
for run_name in test_run_names:
    print(f"Starting run: {run_name}")
    # Get original reads & labels
    original_label_df = label_df[label_df["run_name"] == run_name]
    original_label_df = original_label_df[original_label_df["cnn_label"] != -1]
    
    # Get original codeword & message
    molbits_in_run = molbits_by_run.get(run_name).get("molbits_in_run")
    original_codeword = np.array(get_tag(get_read_counts(molbits_in_run, possible_labels=possible_labels_str)))
    original_message = original_codeword[:32]
         
    original_label_df["target_label"] = original_label_df["cnn_label"]

    original_labels = original_label_df["target_label"]
    original_read_counts = get_read_counts(original_labels,
                                           possible_labels=possible_labels)
    original_read_probs = original_read_counts / sum(original_read_counts)

    # Sample the read counts
    sampled_counts = []
    print(f"Sampling at each depth...")
    for depth in sample_depths:
        print(f"Depth: {depth} ({n_samples} samples)")
        for sample_i in range(n_samples):
            # 1. Sample
            #    Using labels + probabilities is significantly faster than sampling from a list of reads
            sample_labels = np.random.choice(range(96), p=original_read_probs, size=depth)
            sample_read_counts = get_read_counts(sample_labels, possible_labels=possible_labels)

            # 2. Rescale
            sample_read_counts = rescale_counts(sample_read_counts, scaling_factors)

            # 3. Save for parallel execution
            sampled_counts.append((sample_read_counts, depth, sample_i, original_message))

    # Decode
    bag = db.from_sequence(sampled_counts)
    dask_map = bag.map(compute_decoding_helper, generator_matrix_file=generator_matrix_file)
    logger.debug(f"Running .")
    results = dask_map.compute(num_workers=10)

    # Save results
    save_fname = f"synthetic_tags/{run_name}_misl_100k_scaled.tsv"
    print(f"Saving to: {save_fname}")

    with open(save_fname, "w+") as f:
        c = ''.join([str(x) for x in original_codeword])
        m = "".join([str(x) for x in original_message])
        f.write(f"Original codeword: {c}\n")
        f.write(f"Original message: {m}\n")
        f.write(f"Bit mapping (original -> original): {original_to_target_map}\n")

    flattened_results = []
    for item in results:
        flattened_results.extend(item)

    subsample_df = pd.DataFrame(flattened_results, columns=["sample_depth", "sample_i", "correct_decoding", "codeword_at_t", "closest_msg", "closest_d", "t"])
    subsample_df.to_csv(save_fname, sep="\t", mode='a', index=False)


# In[ ]:


# Save results
save_fname = f"synthetic_tags/{run_name}_misl_100k_scaled.tsv"
print(f"Saving to: {save_fname}")

with open(save_fname, "w+") as f:
    c = ''.join([str(x) for x in original_codeword])
    m = "".join([str(x) for x in original_message])
    f.write(f"Original codeword: {c}\n")
    f.write(f"Original message: {m}\n")

flattened_results = []
for item in results:
    flattened_results.extend(item)

subsample_df = pd.DataFrame(flattened_results, columns=["sample_depth", "sample_i", "correct_decoding", "codeword_at_t", "closest_msg", "closest_d", "t"])
subsample_df.to_csv(save_fname, sep="\t", mode='a', index=False)

