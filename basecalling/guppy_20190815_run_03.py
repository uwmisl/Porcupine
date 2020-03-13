#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime
import os


# In[2]:


def import_gdrive_sheet(gdrive_key, sheet_id):
    run_spreadsheet = pd.read_csv("https://docs.google.com/spreadsheet/ccc?key=" +                                   gdrive_key + "&output=csv&gid=" + sheet_id)
    if "date" in run_spreadsheet.columns:
        run_spreadsheet["date"] = run_spreadsheet["date"].astype(str)
    return run_spreadsheet

gdrive_key = "gsheet_id_here"
sheet_id = "0"
# set_sheet_id = "512509543"

run_spreadsheet = import_gdrive_sheet(gdrive_key, sheet_id)
# set_spreadsheet = import_gdrive_sheet(peptide_gdrive_key, set_sheet_id)


# In[3]:


run_date = "20190815"
run_name = "08_15_2019_run_03"


# In[4]:


today = datetime.today().strftime('%Y%m%d')
row = run_spreadsheet[run_spreadsheet["run_name"] == run_name].iloc[0, :]
basecall_dir = os.path.join(row["base_dir"], f"guppy_3.2.2_{run_name}_exec_{today}")
flowcell = row["flowcell_type"]
kit = row["sequencing_kit"]
fast5_dir = row["raw_fast5_dir_multi"]


# In[5]:


basecall_dir


# In[6]:


gpu = "cuda:0"


# In[7]:


guppy_executable = "/path/to/guppy/ont-guppy/bin/guppy_basecaller"


# In[8]:


min_qscore = 9


# # Basecall

# In[9]:


guppy_cmd = f"{guppy_executable} -i {fast5_dir} --recursive -s {basecall_dir} --flowcell {flowcell} " +             f"--kit {kit} --device {gpu} --min_qscore {min_qscore} --qscore_filtering"
guppy_cmd


# In[10]:


get_ipython().system(' {guppy_cmd}')


# # QC

# In[11]:


summary_file = os.path.join(basecall_dir, "sequencing_summary.txt")
html_outfile = os.path.join(basecall_dir, "pycoQC_summary_plots.html")
min_pass_qual = min_qscore


# In[12]:


from pycoQC.pycoQC import pycoQC
from pycoQC.pycoQC_plot import pycoQC_plot

# Import helper functions from pycoQC
from pycoQC.common import jhelp

# Import and setup plotly for offline plotting in Jupyter 
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode (connected=False)


# ## Init pycoQC

# In[13]:


p = pycoQC(summary_file, html_outfile=html_outfile)


# In[ ]:


html_outfile


# In[ ]:


fig = p.summary()
iplot (fig, show_link=False)


# In[ ]:


fig = p.reads_len_1D()
iplot(fig, show_link=False)


# In[ ]:


fig = p.reads_qual_1D()
iplot(fig, show_link=False)


# In[ ]:


fig = p.output_over_time ()
iplot(fig, show_link=False)


# In[ ]:


fig = p.qual_over_time ()
iplot(fig, show_link=False)


# In[ ]:


fig = p.len_over_time ()
iplot(fig, show_link=False)


# In[ ]:


fig = p.channels_activity ()
iplot(fig, show_link=False)


# In[ ]:




