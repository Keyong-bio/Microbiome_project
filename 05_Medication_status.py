
# ============================================================
Source: MAFLD_check.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from pheno_utils import PhenoLoader

pl = PhenoLoader('curated_phenotypes')
pl


# In[3]:


pl.dict[['feature_set', 'relative_location']].drop_duplicates()


# In[2]:


import pandas as pd

df_mafld = pd.read_parquet('../studies/hpp_datasets/curated_phenotypes/mafld.parquet', engine='pyarrow')
df_mafld


# In[3]:


df_mafld_base = df_mafld.query("research_stage == '00_00_visit'")


# In[4]:


df_mafld_base


# In[6]:


df_mafld_base["mafld__curated_phenotype"].value_counts(dropna = False)


# In[7]:


df_mafld_base["mafld__self_reporting"].value_counts(dropna = False)


# In[8]:


pd.crosstab(df_mafld_base["mafld__self_reporting"], df_mafld_base["mafld__curated_phenotype"])


# In[11]:


df_mafld_base["mafld__curated_phenotype_2"] = df_mafld_base["mafld__curated_phenotype"].combine_first(df_mafld_base["mafld__self_reporting"])


# In[12]:


df_mafld_base["mafld__curated_phenotype_2"].value_counts(dropna = False)


# In[14]:


df_mafld_base.index.get_level_values("participant_id").nunique()
df_mafld_base_1 = df_mafld_base["mafld__curated_phenotype_2"].reset_index()
df_mafld_base_1


# In[19]:


df_mafld_base_1["mafld__curated_phenotype_2"].value_counts(dropna = False)


# In[23]:


df_mafld_base_1.loc[:, "mafld_diagnosed"] = df_mafld_base_1["mafld__curated_phenotype_2"].replace({
    "Normal liver": "No",
    "MAFLD/NASH" : "Yes",
    "Suspected MAFLD/NASH / Intermediate risk": "No",
    # Add more
    False : "No"
})

df_mafld_base_1["mafld_diagnosed"].value_counts(dropna = False)


# In[24]:


df_mafld_base_1.to_csv("MAFLD_baseline.csv", index = True)



# ============================================================
Source: Medicical_status.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Consider the Medication

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from pheno_utils import PhenoLoader


# In[2]:


pl_medical = PhenoLoader('medical_conditions')
pl_medical


# In[3]:


pl_medical.dfs['medical_conditions'].head()


# In[4]:


data_medical = pl_medical.dfs['medical_conditions']
data_medical


# ### Baseline prevalent disease

# In[5]:


# filter out baseline visit
data = pl_medical[pl_medical.fields] # get all fields, with age and sex
data_base = data[data.index.get_level_values('research_stage') == '00_00_visit']
data_base


# In[6]:


data_fu = data[data.index.get_level_values('research_stage') == '02_00_visit']
data_fu


# In[7]:


# 10160
data_base.index.get_level_values('participant_id').nunique()


# In[8]:


# 1205 follow-up
data_fu.index.get_level_values('participant_id').nunique()


# In[9]:


data_base2 = data_base[ data_base.data_source == 'initial medical survey']
data_base2.head()


# In[10]:


data_base.data_source.value_counts()


# In[11]:


data_base2.index.get_level_values('participant_id').nunique()


# In[12]:


data_base = data_base.reset_index()
data_base


# In[13]:


Baseline_NAFLD = data_base.loc[data_base["icd11_code"].str.contains(r'DB92'),:]
Baseline_NAFLD


# In[14]:


data_base.loc[data_base["icd11_code"].str.contains(r'5A11'),:]


# In[15]:


data_base.loc[data_base["icd11_code"].str.contains(r'5C'),:]["icd11_code"].value_counts()


# In[16]:


# 1842 individuals with 5C code

Baseline_Dyslipidia = data_base.loc[data_base["icd11_code"].str.contains(r'5C'),:]
Baseline_Dyslipidia["participant_id"].nunique()

Baseline_Dyslipidia_unique = Baseline_Dyslipidia.loc[~Baseline_Dyslipidia.duplicated(subset = "participant_id",
                                                                                                  keep = "last"), :]
Baseline_Dyslipidia_unique


# In[17]:


Baseline_NAFLD_unique = Baseline_NAFLD.loc[~Baseline_NAFLD.duplicated(subset = "participant_id"), :]


# In[18]:


# Baseline_NAFLD.loc[Baseline_NAFLD.duplicated(subset = ["participant_id"]), :]


# In[19]:


Baseline_NAFLD_unique


# In[20]:


data_base["participant_id"].nunique()


# In[21]:


data_base_unique = data_base.loc[~data_base.duplicated(subset = "participant_id", keep = "last")]
data_base_unique


# In[24]:


Baseline_NAFLD_unique.shape


# In[25]:


condition1 = data_base_unique["participant_id"].isin(Baseline_NAFLD_unique["participant_id"])
condition1.value_counts()


# In[26]:


data_base_unique.loc[:, "NAFLD_baseline_diagnosis"] = condition1.map({True:"Yes", False:"No"})
data_base_unique.NAFLD_baseline_diagnosis.value_counts()


# In[27]:


data_base_unique.to_csv("Medication_baseline.csv", index = False)


# In[22]:


# dfs = [data_base_unique, Baseline_NAFLD_unique[["", ""]], df3]
# # Using reduce to merge all DataFrames on 'id'
# res = reduce(lambda left, right: pd.merge(left, right, on='id', how='outer'), dfs)
# print(res)



# ============================================================
Source: Merge.py
# ============================================================

from pathlib import Path

output = Path("Combined.py")
parts = []

for py_file in sorted(Path(".").glob("*.py")):
    if py_file == output:
        continue
    parts.append(f"\n# {'='*60}")
    parts.append(f"Source: {py_file.name}")
    parts.append(f"# {'='*60}\n")
    parts.append(py_file.read_text(encoding = "utf-8"))

output.write_text("\n".join(parts), encoding = "utf-8")
print(f"combined as {output}")