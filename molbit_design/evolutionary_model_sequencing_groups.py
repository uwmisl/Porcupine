#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[56]:


import os
import subprocess
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from _ucrdtw import ucrdtw
from pore_utils.smith_waterman import s_w
sns.set(font_scale=1.5, style="white")
import scipy as sp


# In[81]:


def revcomp(seq):
    seq = seq.upper()
    seq = seq.replace("A", "X")
    seq = seq.replace("T", "A")
    seq = seq.replace("X", "T")
    seq = seq.replace("C", "X")
    seq = seq.replace("G", "C")
    seq = seq.replace("X", "G")
    return seq[::-1]


def calculate_dtw(sequences):
    n_seqs = len(sequences)
    D = np.zeros((n_seqs, n_seqs))
    scrappie_dfs = [None for _ in range(n_seqs)]
    squiggles = [None for _ in range(n_seqs)]
    for j in range(n_seqs):
        if scrappie_dfs[j] is None:
            scrappie_dfs[j] = simulate_squiggle(sequences[j])
            squiggles[j] = list(scrappie_dfs[j]["current"])
        for i in range(n_seqs):
            if scrappie_dfs[i] is None:
                scrappie_dfs[i] = simulate_squiggle(sequences[i])
                squiggles[i] = list(scrappie_dfs[i]["current"])
            if i >= j:
                continue
            else:
                dtw_dist = calc_dtw(squiggles[i], squiggles[j])
                D[i, j] = dtw_dist
                D[j, i] = dtw_dist
    return D


def calc_dtw(scrappie_df_1, scrappie_df_2, warp_width=0.1):
    _, dtw_dist = ucrdtw(scrappie_df_1, scrappie_df_2, warp_width, False)
    dtw_dist = np.float32(dtw_dist)
    return dtw_dist


def simulate_squiggle(sequence):
    rand = np.random.randint(0, 500000)
    temp_fasta_fname = sequence + "_" + str(rand) + "_temp.fa"
    temp_scrappie_fname = sequence + "_" + str(rand) + "_temp.scrappie"
    with open(temp_fasta_fname, "w") as f:
        f.write(">temp\n%s\n" % (sequence))    
    scrappie_str = "scrappie squiggle -o %s %s" % (temp_scrappie_fname, temp_fasta_fname)
    os.system(scrappie_str)
    os.remove(temp_fasta_fname)
    
    with open(temp_scrappie_fname, "r") as f:
        scrappie_lines = f.readlines()
    os.remove(temp_scrappie_fname)
    
    scrappie_sim = []
    seq_name = None
    df = None
    for i, line in enumerate(scrappie_lines):
        line = line.strip()
        if line.startswith("#"):
            seq_name = line
        elif line.startswith("pos"):
            continue
        else:
            scrappie_sim.append(line.split("\t"))
    df = pd.DataFrame(scrappie_sim, columns=["pos", "base", "current", "sd", "dwell"])
    df = df.astype({"pos": int, "base": str, "current": float, "sd": float, "dwell": float})
    return df


# In[7]:


def calculate_sw(sequences):
    D_sw = np.zeros((96, 96))
    for j in range(96):
        for i in range(96):
            if i >= j:
                continue
            else:
                _, sw, a, _,  _ = s_w(sequences[j], sequences[i], cost_fn={"match": 1, "mismatch": -1, "gap": -8})
                D_sw[i, j] = sw
                D_sw[j, i] = sw
    return D_sw


# In[5]:


with open("evolve_from_v3_sw_2adj_seqs.json", "r") as f:
    seqs = json.load(f)


# In[10]:


D_sw = calculate_sw(seqs)


# In[57]:


D_dtw = calculate_dtw(seqs)


# In[11]:


sw_condensed = sp.spatial.distance.squareform(D_sw)
Z = sp.cluster.hierarchy.linkage(sw_condensed, method="weighted")


# In[12]:


fig = plt.figure(figsize=(25, 10))
dn = sp.cluster.hierarchy.dendrogram(Z)
plt.show()


# In[47]:


cluster_init = np.random.choice(range(96), size=6, replace=False)
print (cluster_init)

dist_copy = D_sw.copy()
for i in range(len(dist_copy)):
    for j in range(len(dist_copy)):
        dist_copy[i, j] = np.max(D_sw) - dist_copy[i, j]

for cluster_i in cluster_init:
    dist_copy[:, cluster_i] = 0



nodes_left = set(range(96)) - set(cluster_init)
clusters = [[x] for x in cluster_init]

while(len(nodes_left) > 0):
    print ("\n\n")
    for cluster_i, cluster in enumerate(clusters):
        print ("Cluster:", cluster_i)
        print (np.max(dist_copy[cluster, :]))
#         print dist_copy[cluster, :]
        min_c_dist = np.empty(96)
        min_c_dist.fill(9999)
        for c in cluster:
            min_c_dist = np.minimum(dist_copy[c, :], min_c_dist)

        cluster_to_add = np.argmax(min_c_dist)
        nodes_left.remove(cluster_to_add)
        clusters[cluster_i].append(cluster_to_add)

        dist_copy[:, cluster_to_add] = 0
        

    


# In[48]:


clusters


# In[61]:


for cluster_i, cluster in enumerate(clusters):
    print("Cluster:", cluster_i)
    
    mini_D_sw = np.zeros((16, 16))
    mini_D_dtw = np.zeros((16, 16))
    mini_dtw_flat = []
    for local_i, i in enumerate(cluster):
        for local_j, j in enumerate(cluster):
            if i == j:
                continue
            sw = D_sw[i, j]
            mini_D_sw[local_i, local_j] = sw
            mini_D_sw[local_j, local_i] = sw
            dtw = D_dtw[i, j]
            mini_D_dtw[local_i, local_j] = dtw
            mini_D_dtw[local_j, local_i] = dtw
            mini_dtw_flat.append(dtw)
    print(np.max(mini_D_sw))
    print(np.mean(mini_D_sw))
    print()
    sns.heatmap(mini_D_sw, vmax=10)
    plt.show()
    print(np.min(mini_dtw_flat))
    print(np.mean(mini_dtw_flat))
    sns.heatmap(mini_D_dtw, vmax=8, vmin=0)
    plt.show()


# In[ ]:




[[75, 0, 17, 22, 3, 59, 86, 29, 48, 93, 64, 65, 81, 80, 16, 30],
 [67, 4, 20, 28, 53, 18, 91, 26, 33, 56, 76, 77, 47, 73, 85, 94],
 [8, 38, 52, 87, 2, 10, 35, 13, 36, 92, 12, 40, 55, 51, 95, 68],
 [62, 23, 24, 58, 5, 70, 9, 43, 57, 90, 11, 46, 79, 72, 66, 88],
 [21, 1, 7, 32, 37, 78, 15, 31, 45, 83, 50, 54, 71, 27, 84, 19],
 [25, 14, 39, 49, 61, 82, 6, 41, 44, 63, 34, 42, 69, 60, 89, 74]]
 
 max 9!![[90, 0, 1, 39, 10, 32, 36, 5, 55, 8, 50, 54, 94, 63, 71, 19],
 [35, 25, 52, 57, 2, 37, 67, 82, 6, 77, 87, 14, 34, 53, 45, 86],
 [41, 9, 20, 95, 22, 88, 13, 15, 21, 91, 69, 17, 27, 51, 85, 89],
 [47, 12, 28, 43, 83, 33, 59, 79, 4, 40, 56, 76, 80, 92, 60, 93],
 [48, 31, 11, 3, 61, 73, 78, 44, 81, 49, 65, 74, 46, 16, 72, 58],
 [62, 38, 18, 7, 23, 30, 70, 24, 29, 64, 75, 26, 84, 42, 68, 66]]
 
 max 9
# In[85]:


idt_lines = []
idt_lines.append("Well Position\tSequence Name\tSequence\n")

fasta_lines = []

new_seq_nos = []
orig_seq_nos = []
seq_counter = 0
new_seq_order = []
for cluster_i, cluster in enumerate(clusters):
    col_i =  cluster_i * 2
    cols = [col_i for _ in range(8)]
    cols.extend([col_i + 1 for _ in range(8)])
    locs = ["%s%d" % (r, c) for r, c in list(zip(rows, cols))]
    print(list(locs))

    for local_seq_no, seq_no in enumerate(cluster):
        seq = seqs[seq_no]
        well = locs[local_seq_no]
        seq_no_f = seq[:-4]
        assert len(seq_no_f) == 36
        seq_no_r = revcomp(seq) + "A"
        assert len(seq_no_r) == 41
        
        idt_lines.append("%s\t%d_f\t/5Phos/%s\n" % (well, seq_counter, seq_no_f))
        idt_lines.append("%s\t%d_r\t/5Phos/%s\n" % (well, seq_counter, seq_no_r))
        fasta_lines.append("> %d_f\n%s\n" % (seq_counter, seq))
        new_seq_nos.append(seq_counter)
        orig_seq_nos.append(seq_no)
        new_seq_order.append(seq)
        seq_counter += 1
        
with open("idt_order.tsv", "w") as f:
    for line in idt_lines:
        f.write(line)
with open("idt_order.fa", "w") as fa:
    for line in fasta_lines:
        fa.write(line)


# In[83]:


idt_lines

[[39, 14, 7, 49, 73, 34, 50, 82, 45, 61, 65, 8, 48, 74, 46, 19],
 [44, 3, 13, 78, 86, 40, 92, 37, 42, 60, 87, 30, 36, 47, 1, 66],
 [9, 25, 15, 59, 93, 31, 91, 6, 63, 71, 57, 77, 94, 51, 58, 5],
 [68, 0, 32, 80, 41, 67, 89, 2, 52, 79, 26, 72, 21, 54, 62, 88],
 [20, 17, 22, 75, 24, 29, 43, 23, 27, 64, 83, 35, 69, 84, 16, 81],
 [10, 4, 11, 53, 18, 70, 90, 28, 38, 55, 76, 12, 33, 56, 85, 95]]
 min 10 (only 1 at 10)[[3, 6, 81, 17, 0, 13, 86, 20, 25, 44, 64, 93, 63, 80, 72, 74],
 [76, 58, 4, 31, 43, 71, 91, 77, 11, 24, 59, 69, 75, 84, 85, 95],
 [38, 8, 52, 87, 2, 32, 35, 29, 41, 48, 16, 21, 26, 51, 94, 73],
 [60, 14, 7, 49, 39, 67, 82, 23, 34, 45, 61, 70, 92, 27, 19, 68],
 [55, 12, 83, 33, 36, 78, 79, 42, 47, 62, 65, 30, 40, 46, 66, 88],
 [18, 1, 28, 37, 10, 53, 90, 9, 15, 50, 54, 56, 57, 89, 5, 22]]
 max 10