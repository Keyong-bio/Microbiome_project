
# ============================================================
Source: 00_Liver_ultrasoundetc_not_used.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Liver ultrasound data in 10K
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
from pheno_utils import PhenoLoader


# In[3]:


## read in the liver ultrasound data
d2 = PhenoLoader('liver_ultrasound')
liver = d2[d2.fields]
liver.head()


# In[12]:


filenames = '~/studies/hpp_datasets/liver_ultrasound/liver_ultrasound.parquet'
liver_ultrasound = pd.read_parquet(filenames)
liver_ultrasound


# In[13]:


liver_ultrasound.columns


# In[14]:


# liver sound speed, liver viscosity, liver elasticity, liver attnuation

# liver_ultrasound[['elasticity_qbox_median', 'elasticity_qbox_mean', 'velocity_qbox_median', 'velocity_qbox_mean', 
#                   'viscosity_qbox_median', 'viscosity_qbox_mean', 'dispersion_qbox_median', 'dispersion_qbox_mean']]

liver_ultrasound[liver_ultrasound.index.get_level_values('research_stage') == '00_00_visit'].index.get_level_values('participant_id').nunique()


# In[15]:


liver_ultrasound_base = liver_ultrasound[liver_ultrasound.index.get_level_values('research_stage') == '00_00_visit']


# In[16]:


liver_ultrasound_base


# In[17]:


liver_ultrasound_base_select = liver_ultrasound_base[["elasticity_qbox_mean", 
                                                      "velocity_qbox_mean", 
                                                      "viscosity_qbox_mean", 
                                                      "dispersion_qbox_mean", 
                                                      "attenuation_coefficient_qbox", 
                                                      "speed_of_sound_qbox"]]
liver_ultrasound_base_select = liver_ultrasound_base_select.groupby("participant_id").mean()


# In[18]:


liver_ultrasound_base_select = liver_ultrasound_base_select.reset_index()
liver_ultrasound_base_select


# In[29]:


liver_ultrasound_base_select.to_csv("liver_ultrasound_baseline_selected.csv", index = False)


# In[7]:


filenames = '~/studies/hpp_datasets/liver_ultrasound/liver_ultrasound_aggregated.parquet'
liver_ultrasound_aggre = pd.read_parquet(filenames)
liver_ultrasound_aggre


# In[8]:


liver_ultrasound_aggre.index.get_level_values('participant_id').nunique() # 8250 individuals
liver_ultrasound_aggre.columns
liver_ultrasound_aggre[['elasticity_mean_mean_of_qboxes', 'elasticity_mean_median_of_qboxes', 
                        'elasticity_median_mean_of_qboxes', 'elasticity_median_median_of_qboxes']]


# In[9]:


# 6758 individuals
liver_ultrasound_aggre[liver_ultrasound_aggre.index.get_level_values('research_stage') == '00_00_visit'].index.get_level_values('participant_id').nunique()


# In[12]:


liver_ultrasound_aggre.index.get_level_values('research_stage').value_counts()


# In[10]:


liver_ultrasound_aggre_base = liver_ultrasound_aggre[liver_ultrasound_aggre.index.get_level_values('research_stage') == '00_00_visit']
liver_ultrasound_aggre_base


# In[13]:


liver_ultrasound_aggre_base.columns


# In[92]:


liver_ultrasound_aggre_base.index.get_level_values('research_stage').value_counts()


# In[93]:


# col = ["elasticity_median_median_of_qboxes", 'viscosity_median_median_of_qboxes', 'velocity_median_median_of_qboxes', 
#        'attenuation_coefficient_qbox', 'dispersion_median_median_of_qboxes']
age_sex = d2[["age", "sex"]].loc[:,:,"00_00_visit",:,:]
age_sex.loc[:, 'sex'] = pd.to_numeric(age_sex['sex'])
age_sex.info()
# merge with age and sex
age_sex = age_sex.reset_index().groupby(by = 'participant_id')[['age', 'sex']].mean().reset_index()
age_sex


# In[94]:


liver_ultrasound_aggre_base = liver_ultrasound_aggre_base.droplevel(['cohort', "research_stage", "array_index"]).reset_index()
liver_ultrasound_aggre_base = pd.merge(liver_ultrasound_aggre_base, age_sex, on = "participant_id", how = "left")
liver_ultrasound_aggre_base


# In[95]:


# liver_ultrasound_aggre_base.to_csv('liver_ultrasound_aggre_base.csv', index = False)


# In[96]:


## select variables of interests
liver_ultrasound_aggre_base.columns

# 'elasticity_mean_median_of_qboxes', elasticity_mean_mean_of_qboxes, 'elasticity_median_median_of_qboxes', 'elasticity_median_mean_of_qboxes'
# velocity_mean_median_of_qboxes', velocity_mean_mean_of_qboxes, velocity_median_median_of_qboxes, velocity_median_mean_of_qboxes
# viscosity_mean_median_of_qboxes viscosity_mean_mean_of_qboxes, viscosity_median_median_of_qboxes, viscosity_median_mean_of_median
# dispersion_mean_median_of_qboxes, dispersion_mean_mean_of_qboxes, dispersion_median_median_of_qboxes, dispersion_median_mean_of_median


# In[97]:


cols = ["participant_id", 'elasticity_mean_median_of_qboxes', "elasticity_mean_mean_of_qboxes", 'elasticity_median_median_of_qboxes', 'elasticity_median_mean_of_qboxes',
       "velocity_mean_median_of_qboxes", "velocity_mean_mean_of_qboxes", "velocity_median_median_of_qboxes", "velocity_median_mean_of_qboxes",
       "viscosity_mean_median_of_qboxes", "viscosity_mean_mean_of_qboxes", "viscosity_median_median_of_qboxes", "viscosity_median_mean_of_median",
       "dispersion_mean_median_of_qboxes", "dispersion_mean_mean_of_qboxes", "dispersion_median_median_of_qboxes", "dispersion_median_mean_of_median",
       "age", "sex"]
liver_ultrasound_aggre_base2 = liver_ultrasound_aggre_base[cols]
liver_ultrasound_aggre_base2.to_csv('liver_ultrasound_aggre_base.csv', index = False)


# In[98]:


liver_ultrasound_aggre_base2


# ## read in the dietary index 

# In[99]:


import pandas as pd
five_scores = pd.read_csv("DPs_Final_score_outlieradj.csv")
five_scores


# In[120]:


five_scores.set_index('participant_id').corr("spearman")


# In[100]:


liver_ultrasound_aggre_base2_dps= pd.merge(liver_ultrasound_aggre_base2, five_scores, on = "participant_id", how = "left")
liver_ultrasound_aggre_base2_dps


# In[101]:


def scaling(df, col_to_scaling):
    df_scaling = df.copy()
    
    for column in col_to_scaling:
        # calculate the mean and std first
        mean_val = df_scaling[column].mean()
        std_val = df_scaling[column].std()
        
        df_scaling[f"{column}_scaled"] = (df_scaling[column] - mean_val)/std_val
        
    return df_scaling  

col_to_scaling = ['AHEI_2010_score_eadj', 'hPDI_score_eadj', 'rDII_score_eadj', 'AMED_score_eadj', 'rEDIH_score_all_eadj']
liver_ultrasound_aggre_base2_dps_scaled = scaling(liver_ultrasound_aggre_base2_dps, col_to_scaling = col_to_scaling)
liver_ultrasound_aggre_base2_dps_scaled.head()


# ## read in the lifestyle factors

# In[106]:


lifestyle_factor = pd.read_csv("lifestyle_factor_all_disease.csv")
display(lifestyle_factor.head())
lifestyle_factor.shape


# In[108]:


lifestyle_factor = lifestyle_factor.drop(columns = ['sex_y', 'age', 'sex_x'])


# In[109]:


liver_ultrasound_aggre_base2_dps_scaled_life = pd.merge(liver_ultrasound_aggre_base2_dps_scaled,
                                                       lifestyle_factor,
                                                       on = "participant_id",
                                                       how = "left")
liver_ultrasound_aggre_base2_dps_scaled_life


# In[110]:


liver_ultrasound_aggre_base2_dps_scaled_life.columns


# In[111]:


col_to_scaling = ['elasticity_mean_median_of_qboxes',
                  'elasticity_mean_mean_of_qboxes', 
                  'elasticity_median_median_of_qboxes',
                  'elasticity_median_mean_of_qboxes', 
                  'velocity_mean_median_of_qboxes',
                  'velocity_mean_mean_of_qboxes', 
                  'velocity_median_median_of_qboxes',
                  'velocity_median_mean_of_qboxes', 
                  'viscosity_mean_median_of_qboxes',
                  'viscosity_mean_mean_of_qboxes', 
                  'viscosity_median_median_of_qboxes',
                  'viscosity_median_mean_of_median', 
                  'dispersion_mean_median_of_qboxes',
                  'dispersion_mean_mean_of_qboxes', 
                  'dispersion_median_median_of_qboxes',
                  'dispersion_median_mean_of_median']
liver_ultrasound_aggre_base2_dps_scaled_life = scaling(liver_ultrasound_aggre_base2_dps_scaled_life, col_to_scaling = col_to_scaling)
liver_ultrasound_aggre_base2_dps_scaled_life.head()


# In[118]:


liver_ultrasound_aggre_base2_dps_scaled_life.to_csv('liver_ultrasound_aggre_base2_dps_scaled_life_clean.csv', index = False)


# In[113]:


# containing scaled character for the colnames
liver_ultrasound_aggre_base2_dps_scaled_life.columns[liver_ultrasound_aggre_base2_dps_scaled_life.columns.str.contains("_scaled")]


# In[126]:


import seaborn as sns
sns.displot(liver_ultrasound_aggre_base2_dps_scaled_life.loc[liver_ultrasound_aggre_base2_dps_scaled_life.elasticity_mean_median_of_qboxes<10], 
            x = "elasticity_mean_median_of_qboxes")

sns.displot(liver_ultrasound_aggre_base2_dps_scaled_life, 
            x = "velocity_mean_median_of_qboxes")


# In[136]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pheno_utils import PhenoLoader


# ### Carotid ultrasound measure

# In[137]:


pl_CIMT = PhenoLoader('carotid_ultrasound')
pl_CIMT


# In[138]:


pl_CIMT.dict


# In[140]:


# Show all fields
pl_CIMT_full = pl_CIMT[pl_CIMT.fields]
pl_CIMT_full.shape
pl_CIMT_full


# In[142]:


pl_CIMT_full.index.get_level_values('research_stage').value_counts()


# In[143]:


pl_CIMT_base = pl_CIMT_full.loc[:, :, "00_00_visit", :]


# In[147]:


pl_CIMT_base1 = pl_CIMT_base.droplevel(['cohort', 'array_index']).reset_index().drop(columns = ['collection_date',
                                                                                               'collection_timestamp',
                                                                                               'timezone'])
pl_CIMT_base1


# In[154]:


import seaborn as sns
sns.histplot(pl_CIMT_base1, x = "imt_fit_left")


# In[157]:


pl_CIMT_base1['imt_fit_left_log'] =  pl_CIMT_base1.loc[:,'imt_fit_left'].transform(np.log)
pl_CIMT_base1

pl_CIMT_base1['imt_fit_right_log'] =  pl_CIMT_base1.loc[:,'imt_fit_right'].transform(np.log)
pl_CIMT_base1


# In[163]:


sns.histplot(pl_CIMT_base1, x = "imt_fit_right_log")


# In[164]:


p_vascular = PhenoLoader('vascular_health')
p_vascular

p_vascular.dict


# In[165]:


p_vascular.describe_field(["from_l_thigh_to_l_ankle_pwv", "from_r_thigh_to_r_ankle_pwv"])



# ============================================================
Source: 01_Microbiome.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Gut Microbiome data preparation
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
from pheno_utils import PhenoLoader


# In[2]:


dl = PhenoLoader('gut_microbiome')
dl


# In[3]:


dl.fields


# data_path = dl['urs_abundance_parquet'].iloc[0, 0]
# data_urs_abundance = pd.read_parquet(data_path)
# display(data_urs_abundance.head())
# data_urs_abundance.shape # 11295 obs

# In[34]:


### segal lab files
# data_path = dl['metaphlan4_results_txt'].iloc[0, 0]
# # data_urs_metadata = pd.read_parquet(data_path)
# data_path 


# In[4]:


dl_collecttime = dl['collection_date']
display(dl_collecttime.head())
dl['collection_date'].index.get_level_values("research_stage").value_counts()


# data_urs_metadata.shape # 3594 species
# data_urs_metadata['species'].nunique()#2131

# In[5]:


# select the strain 
# data_path = dl['metaphlan_abundance_strain_parquet'].iloc[0, 0]
# data_strain = pd.read_parquet(data_path)
# print(data_strain.shape)
# display(data_strain.head())

# # select the 'metaphlan_abundance_phylum_parquet',
data_path = dl['metaphlan_abundance_phylum_parquet'].iloc[0,0]
data_phylum = pd.read_parquet(data_path)
print(data_phylum.shape) # 22 phylum 
display(data_phylum.head())

# species 
data_path = dl['metaphlan_abundance_species_parquet'].iloc[0, 0]
data_species = pd.read_parquet(data_path)
print(data_species.shape) # 2088
display(data_species.head())

# # select the genus 
data_path = dl['metaphlan_abundance_genus_parquet'].iloc[0, 0]
data_genus = pd.read_parquet(data_path)
print(data_genus.shape) # 1061
display(data_genus.head())

# family parquet
data_path = dl['metaphlan_abundance_family_parquet'].iloc[0,0]
data_family = pd.read_parquet(data_path)
print(data_family.shape) # 267
display(data_family.head())


# In[6]:


# Get the species struction
data_species.columns
data_genus.columns


# In[7]:


# Make a taxonomy table for species and genus
taxonomy_species = pd.DataFrame({"Full_name": data_species.columns})
taxonomy_species


# In[8]:


# taxonomy_species[taxonomy_species["Full_name"] != "unassigned"].shape
taxonomy_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
taxonomy_species[taxonomy_columns] = taxonomy_species["Full_name"].str.split("|", expand = True)
display(taxonomy_species.head())

# Make a taxonomy table for genus
taxonomy_genus = pd.DataFrame({"Full_name": data_genus.columns})

# taxonomy_species[taxonomy_species["Full_name"] != "unassigned"].shape
taxonomy_columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
taxonomy_genus[taxonomy_columns] = taxonomy_genus["Full_name"].str.split("|", expand = True)
taxonomy_genus.head()


# In[9]:


taxonomy_genus.to_csv("./Results/genus_structure.csv", index = False)
taxonomy_species.to_csv("./Results/species_structure.csv", index = False)


# In[10]:


# print(data_strain.columns[1])
print(data_species.columns[1])
print(data_genus.columns[1])
print(data_family.columns[1])
print(data_phylum.columns[1])


# In[11]:


data_genus_2 = data_genus.query("research_stage=='00_00_visit'")
data_family_2 = data_family.query("research_stage=='00_00_visit'")
data_species_2 = data_species.query("research_stage=='00_00_visit'")
# data_strain_2 = data_strain.query("research_stage=='00_00_visit'")

data_phylum_2 = data_phylum.query("research_stage=='00_00_visit'")
data_phylum_2.head()


# In[12]:


data_species_fu = data_species.query("research_stage =='02_00_visit'")
data_species_fu.index.get_level_values("participant_id").nunique()


# In[13]:


overlap_participant = list(set(data_species_2.index.get_level_values("participant_id"))
    .intersection(data_species_fu.index.get_level_values("participant_id")))
len(overlap_participant) # 1975 overlapped individuals having baseline and follow-up


# In[14]:


data_species_fu


# In[46]:


# # strains
# display(data_strain_2.head(5))
# data_strain_2.index.get_level_values('participant_id').nunique() # 9087


# In[15]:


# species
display(data_species_2.head(5))
print(data_species_2.index.get_level_values('participant_id').nunique())
data_genus_2.index.get_level_values('participant_id').nunique() # 9087 individuals had baseline microbiome data


# In[48]:


print(data_phylum_2.index.get_level_values('participant_id').nunique())
data_phylum_2.shape


# In[16]:


# print(data_strain_2.index.get_level_values('array_index').value_counts())
print(data_species_2.index.get_level_values('array_index').value_counts())
data_species.index.get_level_values('research_stage').value_counts()


# In[17]:


# data_phylum_3['missing_count'] = data_phylum_3.isnull().sum(axis=1)
# data_phylum_3_sorted = data_phylum_3.sort_values(by='missing_count')
# data_phylum_3_unique = data_phylum_3_sorted.reset_index().drop_duplicates(subset=['participant_id'], keep='first').set_index('participant_id').drop(columns='missing_count')


# # Plot the composition of the phylum

# In[18]:


data_phylum_2


# In[19]:


data_phylum_3 = data_phylum_2.reset_index().drop(columns = ['cohort', 'research_stage', 'array_index']).groupby("participant_id", as_index=False).mean()
data_phylum_3


# In[20]:


Phylum_names = data_phylum_3.columns.str.split('|').str[-1]

print(Phylum_names)

# Clean names of the Phylum 
cleaned_names = []
for name in Phylum_names:
    if name.startswith('p__'):
        cleaned_names.append(name[3:])
    else:
        cleaned_names.append(name)
cleaned_names
data_phylum_3.columns = cleaned_names


# In[21]:


data_phylum_3_sort = data_phylum_3.sort_values(by = ["Bacteroidetes"], 
                                               ascending=False)
data_phylum_3_sort


# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[23]:


data_phylum_3_sort = data_phylum_3_sort.set_index("participant_id")
data_phylum_3_sort.head()


# In[24]:


# sum by rows
data_phylum_3_sort.sum(axis = 1)


# In[25]:


data_phylum_3_sort = data_phylum_3_sort.fillna(0)
data_phylum_3_sort 


# In[26]:


# data_phylum_3_sort.columns
mb_full_phyla = data_phylum_3_sort.T.reset_index().groupby('index').sum()
mb_full_phyla.head()

# Calculate the percentage of the microbiome
mb_full_phyla = mb_full_phyla.apply(lambda x: x/x.sum(), axis=0)
mb_full_phyla


# In[27]:


# Reorder the data
mb_full_phyla = mb_full_phyla.loc[mb_full_phyla.mean(1)
    .sort_values(ascending=False).index].T.sort_values('Firmicutes', ascending=False)


# In[28]:


mb_full_phyla


# In[29]:


fig, ax = plt.subplots(1, 1, figsize = (15,4))

labels_fontsize = 10
tick_fontsize = 10

# Plot the first 8 phyla gut microbiome
mb_full_phyla.iloc[:, 0:8].plot(kind='bar', 
                                 stacked=True, 
                                 edgecolor='white', 
                                 linewidth=0.1,
                                 ax=ax, 
                                 width=1, 
                                 color=sns.color_palette('Paired', 
                                                      n_colors=mb_full_phyla.shape[1]))
ax.set_xticks([])
ax.set_ylim((0, 1))
ax.tick_params(labelsize=tick_fontsize)
ax.set_xlabel('Participants', fontsize=labels_fontsize)
ax.set_ylabel('Relative abundance', fontsize=labels_fontsize)
leg = ax.legend(ncol=1, 
                loc='center left' , 
                bbox_to_anchor=(1.01, 0.5),
                fontsize=tick_fontsize, frameon=True, labelspacing=0.5, title='Phyla')
leg.get_title().set_fontsize(str(labels_fontsize))

fig.tight_layout()

# save the plot
plt.savefig(
    "Results/Distribution_of_Phylum.pdf",
    format = "pdf",
    bbox_inches = "tight"
)


# In[ ]:


# fig, ax = plt.subplots(figsize = (12,12))

# # Generate colors
# colors = ['#CD5C5C', '#4682B4', '#228B22', '#483D8B', '#DDA0DD', 
#               '#20B2AA', '#F0E68C', '#FF6347', '#40E0D0', '#EE82EE',
#               '#90EE90', '#FFB6C1', '#87CEEB', '#DDA0DD', '#98FB98',
#               '#F0E68C', '#FF69B4', '#87CEFA', '#DDA0DD', '#98FB98',
#               '#F5DEB3', '#FFB6C1']

# # Extend colors if needed
# if len(data_phylum_3_sort.columns) > len(colors):
#     additional_colors = plt.cm.Set3(np.linspace(0, 1, len(data_phylum_3_sort.columns) - len(colors)))
#     colors.extend(additional_colors)

# data_phylum_3_sort_perc = data_phylum_3_sort.divide(data_phylum_3_sort.sum(axis=1), axis=0)

# print(colors)

# ax.stackplot(data_phylum_3_sort_perc.index, *[data_phylum_3_sort_perc[col] for col in data_phylum_3_sort_perc.columns], 
#                 labels = data_phylum_3_sort_perc.columns, colors=colors, alpha=0.8)

# plt.tight_layout()


# In[ ]:


# # data_phylum_3 = data_phylum_3.reset_index()
# data_phylum_3_long = data_phylum_3_unique_sorted.melt(
#     id_vars=['participant_id'],
#     var_name='phylum',
#     value_name='abundance'
# )
# data_phylum_3_long


# In[ ]:


# # Aggregate data
# df1 = data_phylum_3_long.groupby(["participant_id", "phylum"], as_index=False).agg({"abundance": "sum"})
# df1


# In[ ]:


# data_phylum_3_long = data_phylum_3_long.set_index('participant_id')


# # Calculate the average abundance of microbiota at species and genus level

# In[30]:


# First remove the duplicates for the species data
data_species_2_result = data_species_2.reset_index().drop(columns = ['cohort', 'research_stage', 'array_index']).groupby("participant_id", as_index=False).mean()
data_species_2_result
data_species_2_result = data_species_2_result.set_index("participant_id")
data_species_2_result


# In[31]:


data_species_2_result.sum(axis = 1)


# In[32]:


data_species_2_result_long = data_species_2_result.reset_index().melt(
    id_vars=['participant_id'],
    var_name='Species',
    value_name='abundance'
)
data_species_2_result_long


# In[33]:


Average_abundance = data_species_2_result_long.groupby("Species")["abundance"].mean().reset_index().sort_values("abundance", ascending = False)
Average_abundance


# In[34]:


Average_abundance.to_csv("./Results/Average_abundance_species.csv", index = False)


# In[36]:


# Also calculate the average abundance for genus
data_genus_2_result = data_genus_2.reset_index().drop(columns = ['cohort', 'research_stage', 'array_index']).groupby("participant_id", as_index=False).mean()
data_genus_2_result
data_genus_2_result = data_genus_2_result.set_index("participant_id")
data_genus_2_result
data_genus_2_result.sum(axis = 1)

data_genus_2_result_long = data_genus_2_result.reset_index().melt(
    id_vars=['participant_id'],
    var_name='Genus',
    value_name='abundance'
)
data_genus_2_result_long
# Genus abundance
Average_genus_abundance = data_genus_2_result_long.groupby("Genus")["abundance"].mean().reset_index().sort_values("abundance", ascending = False)
# Average_genus_abundance


# In[37]:


# Filter columns
Average_genus_abundance_filter = Average_genus_abundance[~Average_genus_abundance["Genus"].str.contains("GGB", case = False, na = False)].dropna()
Average_genus_abundance_filter
print(Average_genus_abundance_filter.shape)

Average_genus_abundance.to_csv("./Results/Average_abundance_Genus.csv", index = False)
# Prevotella dominate in people with plant-rich diets, and Bacteroides is the most abundant genus in indivdiuals with western diet.


# In[ ]:


pd.set_option("display.max_colwidth", None)
Average_genus_abundance.iloc[0:10,0]


# # Species_level

# data_species_2.columns

# In[38]:


sum_species = data_species_2.iloc[:,:].sum(axis=1)
print(sum_species.shape)
sum_species


# In[39]:


sum_genus = data_genus_2.iloc[:,:].sum(axis=1)
print(sum_genus.shape)
sum_genus


# In[40]:


# data_species_2 = data_species_2[data_species_2['array_index'] == 0] # remove the duplicated participants
## What if select the duplicated one with the less missingness?
data_species_2_copy = data_species_2.copy()
data_species_2_copy.loc[:, 'missing_count'] = data_species_2_copy.isnull().sum(axis=1)
print(data_species_2_copy.shape) 


# In[41]:


data_species_2_sorted = data_species_2_copy.sort_values(by='missing_count').reset_index()


# In[42]:


data_species_2_sorted.head()


# In[ ]:


# dp = data_species_2_sorted[data_species_2_sorted.duplicated(subset = ['participant_id'], keep = False)].sort_values(by = "participant_id")
# dp
# dp.drop(columns = ['cohort', 'research_stage','missing_count', 'array_index'],inplace = True) 
# dp = dp.set_index("participant_id")
# dp.head(10)


# ## There are duplicates 

# In[ ]:


# data_species_2_result = data_species_2_sorted.drop_duplicates(subset = ['participant_id'], keep='first').drop(columns='missing_count')
# data_species_2_result.shape


# In[ ]:


# data_species_2_sorted.head()


# In[43]:


data_species_2_sorted.columns[:5]


# In[44]:


# another way to do is calculating the mean value of the duplicates
data_species_2_result = data_species_2_sorted.drop(columns = ['missing_count', 'cohort', 'research_stage', 'array_index']).groupby("participant_id", as_index=False).mean()
data_species_2_result


# In[45]:


data_species_2_result = data_species_2_result.set_index('participant_id')


# In[46]:


# split string and select the columns
New_names = data_species_2_result.columns.str.split('|').str[-3:] 

# combine two elements by "|"
New_names = New_names.str.join("|")
New_names


# In[47]:


data_species_2_result.columns = New_names
data_species_full = data_species_2_result


# In[48]:


# find duplicate rows
# duplicate_rows = data_species_2.duplicated(subset=['participant_id']) 
# index = np.where(duplicate_rows == True)
# temp = data_species_2.loc[duplicate_rows]

# temp2_duplicated = data_species_2[data_species_2.participant_id.isin(temp.participant_id)]

# pd.set_option('display.max_columns', None)
# temp2_duplicated.head()

display(data_species_full.head())
data_species_full.shape


# In[49]:


data_species_full.to_csv('./Data/data_species_raw.csv', index = True)


# In[50]:


data_species_full.sum(axis = 1)


# In[51]:


data_species_full_2 = data_species_full.fillna(0)
data_species_full_2.shape


# In[52]:


data_species_full_2.columns
data_species_full_2.sum(axis = 1)


# In[53]:


data_species_full_2.to_csv("./Data/Gut_species_abundance_fillna_withZero.csv", index = True)


# ## For the genus levels

# In[54]:


data_genus_2_result

New_names_genus = data_genus_2_result.columns.str.split('|').str[-2:] 

# combine two elements by "|"
New_names_genus = New_names_genus.str.join("|")
New_names_genus

data_genus_2_result.columns = New_names_genus
data_genus_2_result


# In[55]:


data_genus_2_result.to_csv('./Data/data_genus_raw.csv', index = True)


# ## Previous Liver analysis 
# - Not relevant any more

# In[ ]:


# ## turn the outliers into NA
# for column in col:
#     mean = df_elasticity_3[column].mean()
#     std = df_elasticity_3[column].std()
#     df_elasticity_3[column] = df_elasticity_3[column].mask((df_elasticity_3[column] < mean - 10 * std) | (df_elasticity_3[column] > mean + 10*std), pd.NA)


# In[ ]:


# len(df_elasticity_3[df_elasticity_3.isna().any(axis=1)]) # 44 obs contain the missing -> correction the outlierm then there are 53 obs containing missingness
# df_elasticity_3[df_elasticity_3.isna().any(axis=1)]


# In[ ]:


# df_elasticity_3.describe()


# In[ ]:


# ## detect the outlier
# import numpy as np
# def winsorize_column(df, column_name, n_std=10):
    
#     # Calculate median and standard deviation
#     for column in column_name:
#         mean = df[column].mean()
#         std = df[column].std()
#     # Calculate lower and upper bounds
#         lower_bound = mean - n_std * std
#         upper_bound = mean + n_std * std
    
#     # Winsorize the column
#         df[column] = np.clip(df[column], lower_bound, upper_bound)
    
#     return df

# columns_to_winsorize = ["elasticity_median_median_of_qboxes", 
#                         'viscosity_median_median_of_qboxes', 
#                         'velocity_median_median_of_qboxes', 
#                         'attenuation_coefficient_qbox', 
#                         'dispersion_median_median_of_qboxes']

# # df_elasticity_3_full = winsorize_column(df_elasticity_3_full, columns_to_winsorize)


# In[ ]:


# df_elasticity_3.columns.get_loc('dispersion_median_median_of_qboxes')
# columns_to_scale = df_elasticity_3.columns[:5]
# columns_to_keep = df_elasticity_3.columns[5:]


# In[ ]:


# # Calculate the mean and standard deviation for each column
# # Calculate mean and standard deviation for the columns to scale
# mean = df_elasticity_3[columns_to_scale].mean()
# std = df_elasticity_3[columns_to_scale].std()

# # Perform scaling: (value - mean) / std
# scaled_df = (df_elasticity_3[columns_to_scale] - mean) / std

# # Create a new DataFrame for the unchanged columns
# unchanged_df = df_elasticity_3[columns_to_keep]

# # Concatenate the scaled and unscaled data
# df_elasticity_3_scale = pd.concat([scaled_df, unchanged_df], axis=1)

# # Print the final DataFrame
# (df_elasticity_3_scale.head())


# In[ ]:


# df_elasticity_3_scale.to_csv('Liver_outcome_scale.csv', index = True)
# df_elasticity_3.to_csv('Liver_outcome.csv', index = True)


# In[ ]:


# df_elasticity_f = d2[col + ["age", "sex"]].loc[:,:,"02_00_visit",:,:]
# df_elasticity_f.shape
# df_elasticity_f.index.get_level_values('participant_id').nunique()  # 2149 with the second visit


# In[ ]:


# ### merge based on the index
# df_merge = pd.merge(data_species_full_qc_clr, df_elasticity_3, left_index=True, right_index=True)
# df_merge.head()


# In[ ]:


# df_merge_scale = pd.merge(data_species_full_qc_clr_scale, df_elasticity_3_scale, left_index=True, right_index=True)
# df_merge_scale # 5661 obs


# In[ ]:


# df_merge.to_csv('Gut_and_liver_0731_outlier_correction.csv', index = True)


# In[ ]:


# df_merge_scale.to_csv('Gut_and_liver_0731_outlier_correction_scaled.csv', index = True)


# In[ ]:


# ## Also combine with the diverisity information
# div_all = pd.read_csv('diversity_all_0807.csv', index_col= 'participant_id')
# df_merge_2 = pd.merge(df_merge, div_all, left_index = True, right_index = True)


# In[ ]:


# df_merge_2_scaled = pd.merge(df_merge_scale, div_all, left_index = True, right_index = True)


# In[ ]:


# df_merge_2_scaled.to_csv('Gut_and_liver_and_diversity_0807_outlier_correction_scaled.csv', index = True)


# In[ ]:


# print(df_merge_2.shape)
# df_merge_2.to_csv('Gut_and_liver_and_diversity_0807_outlier_correction.csv', index = True)


# In[ ]:


# df_merge_2.columns


# ### Partial correlation analysis and Mulitivariate Linear regression model
# - deprecation

# In[ ]:


# ## 1. adjusted for the age and sex
# ## 1.1 Linear regression model
# import statsmodels.api as sm

# ## d2.get('elasticity', flexible=True)
# df_merge_2.columns


# df_merge_3 = df_merge_2.dropna()

# df_merge_3['sex'] = df_merge_3['sex'].astype('category')

# In[ ]:


# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

# # Assuming df_merge_3 is your DataFrame and 'sex' is the column you want to map
# # Map 'sex' to binary categories (example: 0 and 1)
# df_merge_3['sex_binary'] = df_merge_3['sex'].map({0: 'Male', 1: 'Female'})  # Adjust this mapping based on your data

# # Create a custom color map for the binary categories
# custom_cmap = mcolors.ListedColormap(['blue', 'red'])  # 'Male' -> blue, 'Female' -> red

# # Plotting
# fig, ax = plt.subplots()
# points = ax.scatter(df_merge_3['PC1'], df_merge_3['PC2'], cv=df_merge_3['sex_binary'].map({'Male': 0, 'Female': 1}),
#                     cmap=custom_cmap, alpha=0.5)

# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('PCoA of Bray-Curtis Beta Diversity')

# # Create a color bar with binary categories
# cbar = fig.colorbar(points, ticks=[0, 1])
# cbar.ax.set_yticklabels(['Male', 'Female'])
# cbar.set_label('Sex')

# # Save the plot
# plt.savefig('Beta_Diversity_3.png', dpi=500, bbox_inches='tight')
# plt.show()


# In[ ]:


# df_merge_3[['s__GGB9059_SGB13976', 'age', 'sex', 'elasticity_median_median_of_qboxes']].describe()


# In[ ]:


import seaborn as sns
sns.jointplot(x = 's__Methanobrevibacter_smithii', y = 'elasticity_median_median_of_qboxes', data = df_merge)
plt.show()

sns.jointplot(x = 's__Methanobrevibacter_smithii', y = 'elasticity_median_median_of_qboxes', data = df_merge_scale)
plt.show()


# In[ ]:


sns.lmplot(x = 's__Methanobrevibacter_smithii', y = 'elasticity_median_median_of_qboxes', 
           hue = 'sex', data = df_merge_2, palette= "Set1")
plt.show()


# ## Sklearn modules
# - deprecation

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:


## use the complete cases
X = df_merge_3[['s__Candidatus_Neochristensenella_gallicola', 'age', 'sex']]
y = df_merge_3[['elasticity_median_median_of_qboxes']]


# In[ ]:


## split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train, y_train)


# In[ ]:


X.columns


# In[ ]:


## 2. adjusted for the age and sex, BMI, 


# ## Partial correlation analysis
# ### Pvalue multiple adjustment
# - deprecation

# In[ ]:


## perform the regression model 
get_ipython().system('pip install pingouin')


# In[ ]:


import pingouin as pg

x1 = pg.partial_corr(data = df_merge, 
                     x='s__GGB9059_SGB13976', 
                     y='elasticity_median_median_of_qboxes', 
                     covar = ['age', 'sex']
               ).round(3) # contain missing value
x1


# In[ ]:


df_merge_3['sex'].dtypes


# In[ ]:


df_merge_3_dummies = pd.get_dummies(df_merge_3[['sex']], drop_first=True)  # Drop the first category to avoid multicollinearity
df_merge_4 = pd.concat([df_merge_3, df_merge_3_dummies], axis=1)


# In[ ]:


df_merge_4.columns


# In[ ]:


x1_1 = pg.partial_corr(data = df_merge_4, x='s__GGB9059_SGB13976', y='elasticity_median_median_of_qboxes', covar = ['age', 'sex_1.0']
               ).round(3) # complete cases 
x1_1


# In[ ]:


selected_columns = [col for col in df_merge_3.columns if col.startswith('s__')]


# In[ ]:


len(selected_columns)


# In[ ]:


## extract the species names
species_name = selected_columns + ['shannon_skbio', 'simpson_skbio', 'observed_otus', 'PC1', 'PC2']


# In[ ]:


def partical_cor_func(dt = df_merge, target_column = "elasticity_median_median_of_qboxes"):
    target_column = target_column
    result_cor = []
    for column in species_name:
        # if column != target_column:
        # X = df_merge[[column]]  # Independent variable(s)
        # y = df_merge[target_column]  # Dependent variable

        results = pg.partial_corr(data = dt, 
                        x = column, 
                        y= target_column, 
                        covar = ['age', 'sex_1.0']
               ).round(3).reset_index()
        
        results_dict = results  # Convert DataFrame to dictionary
        results_dict['Microbiome'] = column
        results_dict['Outcome'] = target_column
        
        result_cor.append(results_dict)
        # Create a DataFrame from the results
        result_cor2 =  pd.concat(result_cor)
    return result_cor2


# In[ ]:


x1 = partical_cor_func(dt = df_merge_4, target_column = "elasticity_median_median_of_qboxes")
x1


# In[ ]:


## Multiple_test
from scipy import stats
ps = x1['p-val']
x1['p_adj'] = stats.false_discovery_control(ps)
x1_sig = x1.loc[x1['p_adj'] < 0.05]
x1_sig


# In[ ]:


x2 = partical_cor_func(dt = df_merge_4, target_column = "viscosity_median_median_of_qboxes")
x2
x2['p_adj'] = stats.false_discovery_control(x2['p-val'])
x2_sig = x2.loc[x2['p_adj'] < 0.05]
x2_sig


# In[ ]:


x3 = partical_cor_func(dt = df_merge_4, target_column = "velocity_median_median_of_qboxes")
x3
x3['p_adj'] = stats.false_discovery_control(x3['p-val'])
x3_sig = x3.loc[x3['p_adj'] < 0.05]
x3_sig


# In[ ]:


x4 = partical_cor_func(dt = df_merge_4, target_column = "attenuation_coefficient_qbox")
x4
x4['p_adj'] = stats.false_discovery_control(x4['p-val'])
x4_sig = x4.loc[x4['p_adj'] < 0.05]
x4_sig


# In[ ]:


x5 = partical_cor_func(dt = df_merge_4, target_column = "dispersion_median_median_of_qboxes")
x5
x5['p_adj'] = stats.false_discovery_control(x5['p-val'])
x5_sig = x5.loc[x5['p_adj'] < 0.05]
x5_sig


# In[ ]:


import seaborn as sns
sns.histplot(data=df_merge, x="attenuation_coefficient_qbox")


# In[ ]:


## Write a function to run multiple linear regression model
import statsmodels.api as sm
def run_multiple_regressions(df, predictors, outcomes, control_vars):
    
    results_list = []
    
    for var in predictors:
        # Generate X and Y
        df1 = df[[var] + control_vars + outcomes]
        df1 = df1.dropna()
        
        X = df1[[var] + control_vars]
        X = sm.add_constant(X)
        y = df1[outcomes]
        
        # fit the model
        model = sm.OLS(y, X).fit()
        
        # extract the results for the variable of interest
        coef = model.params[var] 
        ci_lower, ci_upper = model.conf_int().loc[var]
        pvalue = model.pvalues[var]
        
        # Store results
        results_list.append({
            'N': df1.shape[0],
            'Variable': var,
            'Outcome': outcomes,
            'Coefficient': coef,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'P_Value': pvalue
        })
        
        #
    results_df = pd.DataFrame(results_list)
        
    results_df['CI'] = results_df.apply(lambda row: f"({row['CI_Lower']:.4f}, {row['CI_Upper']:.4f})", axis=1)
    return results_df


# In[ ]:


## Combined with bmi information
baseline_info = pd.read_csv('out_with_diet.csv')
baseline_info_1 = baseline_info[['participant_id', 'bmi']].set_index('participant_id')


# baseline_info.columns

# In[ ]:


df_species_complete_meta_scale = pd.merge(df_merge_2_scaled, baseline_info_1, left_index=True, right_index=True, how='inner')

df_species_complete_meta = pd.merge(df_merge_2, baseline_info_1, left_index=True, right_index=True, how='inner')


# In[ ]:


df_species_complete_meta.describe()


# In[ ]:


df_species_complete_meta_scale.describe()


# In[ ]:


df_species_complete_meta.to_csv('df_species_complete_meta.csv', index = True)


# In[ ]:


df_species_complete_meta_scale.to_csv('df_species_complete_meta_scale.csv', index = True)


# In[ ]:


outcomes1 = ['elasticity_median_median_of_qboxes']
control_vars1 = ['age', 'sex', 'bmi']  # Add other control variables as needed
predictors = [col for col in df_species_complete_meta.columns if col.startswith('s__')]
len(predictors) # 379 species


# In[ ]:


lm_fit1 = run_multiple_regressions(df = df_species_complete_meta, predictors = predictors, outcomes = outcomes1, control_vars = control_vars1)
display(lm_fit1)


# In[ ]:


from scipy import stats
lm_fit1['p_adj'] = stats.false_discovery_control(lm_fit1['P_Value'])
lm_fit1_sig = lm_fit1.loc[lm_fit1['p_adj'] < 0.05] # select based on FDR < 0.1
lm_fit1_sig 


# In[ ]:


outcomes2 = ['viscosity_median_median_of_qboxes']

lm_fit2 = run_multiple_regressions(df = df_species_complete_meta, predictors = predictors, outcomes = outcomes2, control_vars = control_vars1)
display(lm_fit2)


# In[ ]:


lm_fit2['p_adj'] = stats.false_discovery_control(lm_fit2['P_Value'])
lm_fit2_sig = lm_fit2.loc[lm_fit2['p_adj'] < 0.1] # select based on FDR < 0.1
lm_fit2_sig 


# In[ ]:


outcomes3 = ['velocity_median_median_of_qboxes']
lm_fit3 = run_multiple_regressions(df = df_species_complete_meta_scale, predictors = predictors, outcomes = outcomes3, control_vars = control_vars1)
# display(lm_fit3)

lm_fit3['p_adj'] = stats.false_discovery_control(lm_fit3['P_Value'])
lm_fit3_sig = lm_fit3.loc[lm_fit3['p_adj'] < 0.05] # select based on FDR < 0.1
lm_fit3_sig 


# In[ ]:


outcomes4 = ['dispersion_median_median_of_qboxes']
lm_fit4 = run_multiple_regressions(df = df_species_complete_meta_scale, predictors = predictors, outcomes = outcomes4, control_vars = control_vars1)
# display(lm_fit3)

lm_fit4['p_adj'] = stats.false_discovery_control(lm_fit4['P_Value'])
lm_fit4_sig = lm_fit4.loc[lm_fit4['p_adj'] < 0.05] # select based on FDR < 0.1
lm_fit4_sig 


# In[ ]:


union = set(lm_fit1_sig['Variable']).union(lm_fit2_sig['Variable'], lm_fit3_sig['Variable'])
union


# ### Convert DataFrame to a list of abundance vectors (one for each sample)
# ### Calculate alpha diversity metrics
# metrics = ['observed_otus', 'shannon', 'simpson']
# alpha_diversities = {}
# for metric in metrics:
#     alpha_diversities[metric] = alpha_diversity(metric, abundance_vectors)
# 
# ### Convert results to a DataFrame for easy viewing
# alpha_div_df = pd.DataFrame(alpha_diversities, index=df.index)

# abundance_vectors = data_strain_full.values.tolist()
# abundance_vectors

# In[ ]:


run_multiple_regressions(df = df_species_complete_meta_scale, predictors = ['shannon_skbio'], outcomes = outcomes1, control_vars = control_vars1)


# In[ ]:


run_multiple_regressions(df = df_species_complete_meta_scale, predictors = ['simpson_skbio'], outcomes = outcomes1, control_vars = control_vars1)


# In[ ]:


run_multiple_regressions(df = df_species_complete_meta, predictors = ['observed_otus'], outcomes = outcomes3, control_vars = control_vars1)


# In[ ]:


df_species_complete_meta.columns


# In[ ]:






# ============================================================
Source: 02_00_DPs_related_Microbiomes-Species-V3.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## DPs related microbiomes
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats


# In[2]:


DPs = pd.read_csv("../01_DPs_Diet_patterns/Adj/DPs_Final_score_outlieradj_v2.csv")
display(DPs.head(5))


# In[3]:


# ## How to flatten the beta diversity (distance matrix)
# distances = []
# for i in range(beta_diversity.shape[0]):
#     for j in range(i+1, beta_diversity.shape[1]):
#         distances.append(beta_diversity[i, j])


# # Quality control
# 
# ## prevalence of species > 10% in all the samples

# In[4]:


data_species_full = pd.read_csv("./Data/data_species_raw.csv", index_col="participant_id")

# prevalence 
percent_missing = (data_species_full.isnull().sum().sort_values(ascending = True) * 100 / len(data_species_full)).round(2)
print(percent_missing)


# In[5]:


percent_notmissing = (data_species_full.notnull().sum().sort_values(ascending = True) * 100 / len(data_species_full)).round(2)
percent_notmissing


# In[6]:


display(percent_notmissing.head())
print(len(percent_notmissing[percent_notmissing >= 10])) # 388 species more than 10%


# In[7]:


species_pre = (data_species_full>0).mean(axis = 0)*100
species_pre.head()


# In[8]:


# sample missingness
percent_missing_row = (data_species_full.isnull().sum(axis=1) / data_species_full.shape[1] * 100).round(2)
percent_missing_row.describe() # for species, the minimum percentage for each individual is 84.58%


# In[9]:


filtered_data = data_species_full.loc[:, species_pre >= 10]

# Calculate mean abundance
species_mean_abundance = filtered_data.mean(axis=0)

abundance_threshold = 0.0001*100 # 0.01%
filtered_data = filtered_data.loc[:, species_mean_abundance >= abundance_threshold]


# In[10]:


filtered_data.shape # 379 species


# In[11]:


# S_names = percent_notmissing[percent_notmissing >= 10].index.to_list()
# len(S_names)

# S_names[0].isin(data_species_full.columns)
# column_name = selects[2]
# if column_name in data_species_full_qc.columns:
#     print(f"The column '{column_name}' is present in the DataFrame.")
# else:
#    print(f"The column '{column_name}' is not present in the DataFrame.")


# In[12]:


# data_species_full_qc = data_species_full[S_names]
data_species_full_qc = filtered_data.copy()
data_species_full_qc.head()


# In[13]:


qc_percent_missing_row = (data_species_full_qc.isnull().sum(axis=1) / data_species_full_qc.shape[1] * 100).round(2)
qc_percent_missing_row.describe() # for species, the minimum percentage for each individual is 35.62%


# In[14]:


# remove the participants with higher missing in sample (more than 90%)
data_species_full_qc.shape


# In[15]:


qc_percent_missing_row[qc_percent_missing_row > 90] # 12 obs

sample_excluded = qc_percent_missing_row[qc_percent_missing_row > 90].index # 12 obs
sample_excluded


# In[16]:


data_species_full_qc = data_species_full_qc[~data_species_full_qc.index.isin(sample_excluded)]


# In[17]:


data_species_full_qc.shape


# In[18]:


pd.set_option("display.width", None)

temp = pd.DataFrame(data_species_full_qc.describe())
temp


# In[19]:


temp = temp.reset_index()


# In[20]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
print(temp.loc[temp["index"] == "min",:].max())


# In[21]:


temp.loc[temp["index"] == "min"].drop(columns = "index").max(axis = 1)


# In[22]:


temp.loc[temp["index"] == "min"].drop(columns = "index").min(axis = 1)


# In[26]:


# data_species_full_qc_2 = data_species_full_qc + min_value/2
# data_species_full_qc_2.head()
data_species_full_qc_2 = data_species_full_qc.fillna(data_species_full_qc.min()/2)
data_species_full_qc_2.shape


# In[23]:


(data_species_full_qc.min()/2).median()


# In[24]:


(data_species_full_qc.min()/2).max()


# In[26]:


(data_species_full_qc.min()/2).min() * 100


# In[38]:


data_species_full_qc.max()


# In[17]:


data_species_full_qc_2.head()
data_species_full_qc_2.to_csv("./Data/data_species_qc.csv", index = True)


# ### Calculated the alpha Diversity index

# In[30]:


# using the QC data to calculate the Diversity index
data_species_full_qc_2 = pd.read_csv('./Data/data_species_qc.csv', index_col="participant_id")

# data_species_full_qc_2 = data_species_full_qc_2.fillna(0)
relative_abundance_matrix = data_species_full_qc_2.to_numpy()


# In[34]:


data_species_full_qc_2


# In[36]:


data_species_full_qc_2.sum(axis = 1).sort_values(ascending = False)


# In[19]:


get_ipython().system('pip install --quiet scikit-bio')


# In[20]:


get_ipython().system('pip install "numpy < 2.0"')


# In[21]:


import skbio
skbio.diversity.alpha.shannon(relative_abundance_matrix[1,:])


# In[22]:


# Calculate alpha diversity metrics
shannon_skbio = skbio.diversity.alpha_diversity('shannon', relative_abundance_matrix, ids = data_species_full_qc_2.index)
simpson_skbio = skbio.diversity.alpha_diversity('simpson', relative_abundance_matrix, ids = data_species_full_qc_2.index)
simpson_evenness = skbio.diversity.alpha_diversity('simpson_e', relative_abundance_matrix, ids = data_species_full_qc_2.index)
# observed_features = skbio.diversity.alpha_diversity('observed_features', relative_abundance_matrix, ids = data_species_full_qc_2.index)

# generate the new data frame
alpha_div = pd.DataFrame({
    'shannon_skbio': shannon_skbio,
    'simpson_skbio': simpson_skbio,
    'simpson_evenness': simpson_evenness,
#    'observed_features': observed_features
})
alpha_div


# In[24]:


# check distribution
alpha_div_copy = alpha_div.copy()

alpha_div_copy.loc[:,"shannon_skbio_log"] = np.log(alpha_div_copy['shannon_skbio'])
alpha_div_copy

alpha_div_copy.loc[:,"simpson_skbio_log"] = np.log(alpha_div_copy['simpson_skbio'])
alpha_div_copy

import seaborn as sns
import matplotlib.pyplot as plt
fig,(ax1, ax2) = plt.subplots(1, 2, figsize = (8,4))

sns.histplot(data = alpha_div_copy, x = "shannon_skbio_log", ax = ax1)
sns.histplot(data = alpha_div_copy, x = "simpson_skbio_log", ax = ax2, color = "lightgreen")

ax1.set_title("Shannon Diversity")
ax2.set_title("Simpson Diversity")

# Adjust the layout
plt.tight_layout()
# Show the plot
plt.show()


# In[29]:


fig,(ax1, ax2) = plt.subplots(1, 2, figsize = (8,4))

sns.histplot(data = alpha_div_copy, x = "shannon_skbio", ax = ax1)
sns.histplot(data = alpha_div_copy, x = "simpson_skbio", ax = ax2, color = "lightgreen")

ax1.set_title("Shannon Index")
ax1.set_xlabel("")
ax2.set_title("Simpson Index")
ax2.set_xlabel("")


# Adjust the layout
plt.tight_layout()

plt.savefig("./Results/Shannon_Simpson_Indx_Distribution.pdf")
# Show the plot
plt.show()



# In[30]:


data_species_full_qc_2 = pd.read_csv('./Data/data_species_qc.csv')
print(data_species_full_qc_2.shape)

DPs = pd.read_csv("../01_DPs_Diet_patterns/Adj/DPs_Final_score_outlieradj_v2.csv")
DPs.shape

species_DP = pd.merge(data_species_full_qc_2, DPs,  on = "participant_id", how = "left")
species_DP

species_DP = species_DP.dropna(subset = ["AHEI_2010_score_eadj", "hPDI_score_eadj", "rDII_score_eadj",
                                        "AMED_score_eadj", "rEDIH_score_all_eadj"])
species_DP


# In[31]:


# Exclude the columns
species_DP_new = species_DP.loc[:, ~species_DP.columns.str.contains("eadj")].set_index("participant_id")
# print(species_DP_new)
species_DP_new


# In[33]:


relative_abundance_matrix_2 = species_DP_new.to_numpy()
beta_diversity_2 = skbio.diversity.beta_diversity('braycurtis', relative_abundance_matrix_2, ids = species_DP_new.index)

# Rewrite the function to calculate the percentile
def create_groups(df, column_name, q = 5):
    """
    Create groups for a numerical column in a pandas dataframe
    """
    temp = pd.qcut(df[column_name], q, labels = [f"Q{i}" for i in range(1, q+1)])
    return temp.astype(str)

# Define the columns
col_to_groups = ["AHEI_2010_score_eadj", 
                 "hPDI_score_eadj", 
                 "rDII_score_eadj",
                 "AMED_score_eadj",
                 "rEDIH_score_all_eadj"]

# Apply function to each column and add to dataframe
for col in col_to_groups:
    species_DP[col + "_quintile"] = create_groups(species_DP, col, q = 5)
    
species_DP


# In[34]:


quintile_cols = [col for col in species_DP.columns if "_quintile" in col]
quintile_cols


# In[35]:


species_DP = species_DP.set_index("participant_id")


# In[37]:


species_DP["rDII_score_eadj"].describe()


# In[38]:


# Perform the PERMONVA TEST
from skbio.stats.distance import permanova

np.random.seed(42)

# store the results
permanova_results = []

for diet_score in quintile_cols:
    
    # perform PERMANOVA
    results = permanova(
        beta_diversity_2,
        grouping= species_DP[diet_score],
        permutations=999
    )

    # Calculate the R2 (R2 = SS_Between / SS_total)
    # R2 = F/(F + df_residual)
    pseudo_F = results["test statistic"]
    nsamples = results["sample size"]
    n_groups = results["number of groups"]

    # Degree of freedom
    df_between = n_groups - 1
    df_residual = nsamples - n_groups

    # Calculate R2 (proportion of variance explained)
    r_squared = (df_between * pseudo_F) / (df_between * pseudo_F + df_residual)

    # extract results
    permanova_results.append(
        {
            "diet_score" : diet_score,
            "method_name" : results["method name"],
            "p_value" : results["p-value"],
            "number_of_permutation" : results["number of permutations"],
            "R_squared": r_squared,
            "variance_explained_pct" : r_squared * 100
            
        })
    

# convert to DataFrame
permanova_df = pd.DataFrame(permanova_results)
permanova_df


# In[39]:


pcoa_results_2 = skbio.stats.ordination.pcoa(beta_diversity_2)
pcoa_results_2.proportion_explained


# In[40]:


beta_diversity_results_2 = pd.DataFrame(pcoa_results_2.samples[['PC1', 'PC2', "PC3"]].reset_index())
# beta_diversity_results.columns[0] = "participant_id"
beta_diversity_results_2.rename(columns = {'index':'participant_id'}, inplace = True)
beta_diversity_results_2 


# In[41]:


# plot the PCA 
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data = beta_diversity_results_2, x = "PC1", y = "PC2")

# Add x-axis and y-axis label
plt.xlabel("PC1 ({:.2%})".format(pcoa_results_2.proportion_explained.values[0]))
plt.ylabel("PC2 ({:.2%})".format(pcoa_results_2.proportion_explained.values[1]))


# In[36]:


# data_species_full_qc_2 = data_species_full_qc_2.fillna(0)
# relative_abundance_matrix = data_species_full_qc_2.to_numpy()


# In[37]:


# beta diversity
# beta_diversity = skbio.diversity.beta_diversity('braycurtis', relative_abundance_matrix, ids = data_species_full_qc_2.index)


# In[38]:


# pcoa_results = skbio.stats.ordination.pcoa(beta_diversity)


# In[39]:


# pcoa_results.proportion_explained
# PC1: 10.9%
# PC2: 7.55%


# In[40]:


# # table_output configuration option using skbio.set_config(), change the global behavior of all scikit-bio functions
# from skbio import set_config
# # Set output format to pandas
# set_config("table_output", "pandas")


# In[41]:


# beta_diversity_results = pd.DataFrame(pcoa_results.samples[['PC1', 'PC2', "PC3"]].reset_index())
# # beta_diversity_results.columns[0] = "participant_id"
# beta_diversity_results.rename(columns = {'index':'participant_id'}, inplace = True)
# beta_diversity_results 


# In[42]:


# beta_diversity_results["PC1"].describe()


# In[42]:


beta_diversity_results_2.to_csv("./Data/beta_diversity_results_2.csv", index = False)


# In[44]:


# plot the PCA 
# beta_diversity_results = pd.read_csv("./Data/beta_diversity_results.csv")
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.scatterplot(data = beta_diversity_results, x = "PC1", y = "PC2")

# # Add x-axis and y-axis label
# plt.xlabel("PC1 ({:.2%})".format(pcoa_results.proportion_explained[0]))
# plt.ylabel("PC2 ({:.2%})".format(pcoa_results.proportion_explained[1]))


# In[43]:


# Combined with DPs
DPs_Div = pd.merge(beta_diversity_results_2, DPs, on = "participant_id", how = "left")
DPs_Div = pd.merge(DPs_Div, alpha_div.reset_index(), on = "participant_id", how = "left")
DPs_Div


# In[44]:


DPs_Div["AHEI_2010_score_eadj"].isnull().sum()
DPs_Div["rDII_score_eadj"].isnull().sum()

# Discharge the rows with NA for diet scores
DPs_Div = DPs_Div.dropna(subset = ["AHEI_2010_score_eadj", "hPDI_score_eadj", "rDII_score_eadj", "AMED_score_eadj", "rEDIH_score_all_eadj"], how = "all")
DPs_Div.shape


# In[47]:


# Function to generate the quintile for each column of diet score
# bins = DPs_Div["AHEI_2010_score_eadj"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# DPs_Div.loc[:, "AHEI_quintile"] = pd.cut(DPs_Div["AHEI_2010_score_eadj"], bins = bins, 
#                                                  labels= ["Q1", "Q2", "Q3", "Q4", "Q5"], include_lowest = True)


# In[45]:


# Create a function to calculate the quintile
def create_quintile(data, column_name):
    bins = data[column_name].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return pd.cut(data[column_name],
                 bins = bins, 
                 labels = ["Q1", "Q2", "Q3", "Q4", "Q5"], 
                 include_lowest = True)

# Apply to multiple columns
for col in ["AHEI_2010_score_eadj", "hPDI_score_eadj", "rDII_score_eadj", "AMED_score_eadj", "rEDIH_score_all_eadj"]:
    short_name = col.split("_")[0]
    DPs_Div[f'{short_name}_quintile'] = create_quintile(DPs_Div, col)


# In[46]:


# Add the function for ellipse
from matplotlib.patches import Ellipse

def confidence_ellipse(ax, x, y, n_std = 2, facecolor = "none", **kwargs):
    # draw an ellipse around the data
    # 1. calculate covariance matrix
    cov_matrix = np.cov(x, y)
    # 2. Pearson correlation coefficient
    pearson = cov_matrix[0,1]/np.sqrt(cov_matrix[0,0] * cov_matrix[1,1])

    # Ellipse radii (based on Standard deviations)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # create ellipse
    ellipse = Ellipse((0,0), width = ell_radius_x * 2, 
                      height = ell_radius_y * 2, 
                      facecolor = facecolor, **kwargs)
    # calculate scaling factors
    scale_x = np.sqrt(cov_matrix[0,0]) * n_std
    scale_y = np.sqrt(cov_matrix[1,1]) * n_std

    # Apply transformation to scale and translate ellipse
    transf = plt.matplotlib.transforms.Affine2D().scale(scale_x, scale_y).translate(x.mean(), y.mean())
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# In[47]:


# Draw ellipse for each group
# Create scatterplot with hue for group variable
fig, ax = plt.subplots(figsize = (5,5))

# ax: Axes project (actual plot area where you draw)
sns.scatterplot(data = DPs_Div, x = "PC1", y = "PC2", hue = "AHEI_quintile", ax = ax, palette="coolwarm")

# Get the color
groups = DPs_Div["AHEI_quintile"].unique()
palette = sns.color_palette("coolwarm", len(groups))

# Draw ellipse for each group with matching hue colors
for i, group in enumerate(groups):
    group_data = DPs_Div[DPs_Div["AHEI_quintile"] == group]
    confidence_ellipse(ax, group_data['PC1'].values, 
                       group_data["PC2"].values, 
                       edgecolor = palette[i], linewidth = 2, facecolor = "none")

# Add x-axis and y-axis label
ax.set_xlabel("PC1 ({:.2%})".format(pcoa_results_2.proportion_explained.values[0]))
ax.set_ylabel("PC2 ({:.2%})".format(pcoa_results_2.proportion_explained.values[1]))
plt.legend(bbox_to_anchor = (0.95, 0.3))
plt.tight_layout()
plt.show()


# In[48]:


permanova_df


# In[49]:


# Define the dietary pattern scores
diet_patterns = {
    "AHEI": "AHEI_quintile",
    "hPDI" : "hPDI_quintile",
    "rDII" : "rDII_quintile",
    "AMED" : "AMED_quintile",
    "rEDIH" : "rEDIH_quintile"
}

# Create the figure with 5 subplots
fig, axes = plt.subplots(1, 5, figsize = (30,6), gridspec_kw = {'wspace': 0.3})

# Color for Q1 and Q5
color_extreme = {"Q1": "#2b83ba", 
                 "Q5": "#d7191c"}

for idx, (diet_name, diet_column) in enumerate(diet_patterns.items()):
    
    ax = axes[idx]
    
    # plot other quintiles as grey
    for quintile in ["Q2", "Q3", "Q4"]:
        
        mask = (DPs_Div[diet_column] == quintile).values
        
        ax.scatter(
            pcoa_results_2.samples.loc[mask, "PC1"],
            pcoa_results_2.samples.loc[mask, "PC2"],
            c = "lightgrey",
            alpha = 0.15,
            s = 12,
            edgecolors = "none" 
        )

    # plot extreme quintiles (Q1 and Q5)
    for quintile in ["Q1", "Q5"]:
        # subset the data
        mask = (DPs_Div[diet_column] == quintile).values
        ax.scatter(
            pcoa_results_2.samples.loc[mask, "PC1"],
            pcoa_results_2.samples.loc[mask, "PC2"],
            c = color_extreme[quintile],
            label = quintile,
            alpha = 0.7,
            s = 40,
            edgecolors = "white",
            linewidth = 0.6,
            zorder = 5
        )

        confidence_ellipse(
            ax,
            pcoa_results_2.samples.loc[mask, "PC1"].values,
            pcoa_results_2.samples.loc[mask, "PC2"].values,
            n_std= 2,
            edgecolor = color_extreme[quintile],
            linewidth = 2.5,
            facecolor= "none",
            zorder = 5
        )
        
    # formating
    ax.set_xlabel(f"PC1({pcoa_results_2.proportion_explained.iloc[0]:.2%})", fontsize = 10)
    if idx == 0:
        ax.set_ylabel(f"PC2({pcoa_results_2.proportion_explained.iloc[1]:.2%})", fontsize = 10)
        
    ax.set_title(diet_name, fontsize = 11, fontweight = "bold")
    ax.grid(True, alpha = 0.3)
    ax.legend(bbox_to_anchor = (0.95, 0.3))

# Overall legend
# handles = [
#     plt.Line2D([0], [0], 
#                marker = "o",
#                color = "w", 
#                markerfacecolor= "#2b83ba", 
#                markersize - 10, 
#                label = "Q1(lowest)"),
    
#     plt.Line2D([0], [0], 
#                marker = "o", 
#                color = "w", 
#                markerfacecolor= "#d7191c", 
#                markersize - 10, 
#                label = "Q5(heighest)")
# ]

# fig.legend(handles = handles, 
#           # loc = "uppercenter", 
#           # ncols = 2, 
#           bbox_to_anchor = (0.5, 1.05), frameon = True, fontsize = 10)

plt.savefig("./Results/PCA_results_2.pdf", dpi = 300)
plt.show()


# In[50]:


# Add Test pvalue and R2
diet_patterns = {
    "AHEI": "AHEI_quintile",
    "hPDI" : "hPDI_quintile",
    "rDII" : "rDII_quintile",
    "AMED" : "AMED_quintile",
    "rEDIH" : "rEDIH_quintile"
}

# Stats
stats_data = {
    "AHEI" : {"p_value": 0.001, "r2": "0.45%"},
    "hPDI" : {"p_value": 0.001, "r2": "0.42%"},
    "rDII" : {"p_value": 0.001, "r2": "0.55%"},
    "AMED" : {"p_value": 0.001, "r2": "0.42%"},
    "rEDIH" : {"p_value": 0.001, "r2": "0.31%"}
}

# Create the figure with 5 subplots
fig, axes = plt.subplots(1, 5, figsize = (24,4), gridspec_kw = {'hspace': 0.5})

# Color for Q1 and Q5
color_extreme = {"Q1": "#2b83ba", 
                 "Q5": "#d7191c"}

for idx, (diet_name, diet_column) in enumerate(diet_patterns.items()):
    
    ax = axes[idx]
    # plot other quintiles as grey
    for quintile in ["Q2", "Q3", "Q4"]: 
        mask = (DPs_Div[diet_column] == quintile).values
        ax.scatter(pcoa_results_2.samples.loc[mask, "PC1"],
                   pcoa_results_2.samples.loc[mask, "PC2"],
                   c = "lightgrey",
                   alpha = 0.15,
                   s = 12,
                   edgecolors = "none")

    # plot extreme quintiles (Q1 and Q5)
    for quintile in ["Q1", "Q5"]:
        # subset the data
        mask = (DPs_Div[diet_column] == quintile).values
        ax.scatter(pcoa_results_2.samples.loc[mask, "PC1"],
                   pcoa_results_2.samples.loc[mask, "PC2"],
                   c = color_extreme[quintile],
                   label = quintile,
                   alpha = 0.7,
                   s = 40,
                   edgecolors = "white",
                   linewidth = 0.6,
                   zorder = 5)

        confidence_ellipse(ax, 
                           pcoa_results_2.samples.loc[mask, "PC1"].values,
                           pcoa_results_2.samples.loc[mask, "PC2"].values,
                           n_std= 2,
                           edgecolor = color_extreme[quintile],
                           linewidth = 2.5,
                           facecolor= "none",
                           zorder = 5)
        
    # formating
    ax.set_xlabel(f"PC1({pcoa_results_2.proportion_explained.iloc[0]:.2%})", fontsize = 10)
    if idx == 0:
        ax.set_ylabel(f"PC2({pcoa_results_2.proportion_explained.iloc[1]:.2%})", fontsize = 10)
        
    ax.set_title(diet_name, fontsize = 11, fontweight = "bold")
    # ax.grid(True, alpha = 0.3)
    ax.grid(False)
    # add text in the figure
    # .get function, if the diet name is not found, then return the default value 0.05, and 0.1
    stats = stats_data.get(diet_name, {"p_value" : 0.05, "r2": 0.1})
    
    ax.text(0.02, 0.98, f"p = {stats['p_value']:.3f}\nR2 = {stats['r2']}", 
            transform = ax.transAxes, fontsize = 9, verticalalignment = "top", bbox = dict(boxstyle = "round", facecolor = "white", alpha = 0.7))

    ax.legend(bbox_to_anchor = (0.98, 0.05), loc = "lower right")

# adjust the margins around the figure
plt.savefig("./Results/PCA_results_2.pdf", dpi = 300, bbox_inches = "tight")
plt.show()


# In[ ]:


# hue_variables = ["AHEI_quintile", "hPDI_quintile", "rDII_quintile", "AMED_quintile", "rEDIH_quintile"]

# # Create subplots
# n_hues = len(hue_variables)
# fig, axes = plt.subplots(1, 5, figsize = (36, 6),
#                         # Extend the width 
#                         gridspec_kw = {'wspace': 0.3})
# axes = axes.flatten()

# # Get color palette
# palette = "coolwarm"

# # Create a plot for each hue variable
# for idx, hue_var in enumerate(hue_variables):
#     ax = axes[idx]

#     # Scatter plot with current hue variable
#     sns.scatterplot(data = DPs_Div, x = "PC1", y = "PC2", hue = hue_var, ax = ax, palette=palette)

#     # Get groups and draw ellipse
#     groups = DPs_Div[hue_var].unique()
#     palette_colors = sns.color_palette(palette, len(groups))

#     for i, group in enumerate(groups):
#         group_data = DPs_Div[DPs_Div[hue_var] == group]
#         confidence_ellipse(ax, 
#                            group_data["PC1"].values, 
#                            group_data["PC2"].values,
#                           edgecolor = palette_colors[i], linewidth = 2, facecolor = "none")
#         ax.set_title(f'PCA: {hue_var}', fontsize = 16, fontweight = "bold")
#         ax.set_xlabel("PC1 ({:.2%})".format(pcoa_results.proportion_explained.iloc[0]), fontsize = 16)
#         ax.set_ylabel("PC2 ({:.2%})".format(pcoa_results.proportion_explained.iloc[1]), fontsize = 16)
#         ax.legend(bbox_to_anchor = (0.95, 0.3))

# # plt.tight_layout()
# plt.savefig("./Results/PCA_results.pdf", dpi = 300)
# plt.show()


# # Boxplot for comparing the Shannon index

# In[51]:


# Test for shannon index across different dietary pattern quintile 

def plot_boxplot_for_group(data, method = "anova", x_col = None , y_col = None, group_col = None, show_plot = True, ax = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    if ax is None:
        fig, ax = plt.subplots(figsize = (4,4))
    else:
        fig = ax.get_figure()
    # Prepare groups for test
    groups = [group_data[y_col].values for _, group_data in data.groupby(group_col, observed = True)]
    
    # Perform test
    if method.lower() == "anova":
        statistic, p_val = stats.f_oneway(*groups)
        test_name = "One-way ANOVA"
    elif method.lower() == "kruskal":
        statistic, p_val = stats.kruskal(*groups)
        test_name = "Kruskal-Wallis Test"
    else:
        raise ValueError("Method must be either anova or kruskal")

    print(f'{test_name} statistic: {statistic:.4f}, p-value: {p_val:.4f}')

    # Createt the boxplot figure
    sns.boxplot(data = data, x = x_col, y = y_col, hue = x_col, ax = ax, palette="coolwarm")
    ax.set_title(f"{y_col} by {x_col}\n{test_name}: p = {p_val:.4f}")

    # Add the results in the figure
    # if show_plot:
    #     plt.figure(figsize=(4,4))
    #     sns.boxplot(data = data, x = x_col, y = y_col, hue = x_col, ax = ax, palette="rocket")
    #     plt.title(f"{y_col} by {x_col}\n{test_name}: p = {p_val:.4f}")
        # plt.tight_layout()
        # plt.show()

    return fig, ax, statistic, p_val


# In[52]:


# Rename the columns
DPs_Div = DPs_Div.rename(
    columns = {'shannon_skbio': 'Shannon Index',
              'simpson_skbio': 'Simpson Index',
              'simpson_evenness': 'Simpson Evenness'}
)


# In[53]:


# Create a figure with subplots
fig, axes = plt.subplots(2, 5, figsize = (20, 8),
                         # Add a bit space for two rows
                        gridspec_kw = {'hspace': 0.5})

# Plot1
plot_boxplot_for_group(
    data = DPs_Div,
    method = "kruskal",
    x_col = "AHEI_quintile",
    y_col = "Shannon Index",
    group_col = "AHEI_quintile",
    ax = axes[0, 0]
)

# Plot2
plot_boxplot_for_group(
    data = DPs_Div,
    method = "kruskal",
    x_col = "hPDI_quintile",
    y_col = "Shannon Index",
    group_col = "hPDI_quintile",
    ax = axes[0, 1]
)

# Plot3
plot_boxplot_for_group(
    data = DPs_Div,
    method = "kruskal",
    x_col = "rDII_quintile",
    y_col = "Shannon Index",
    group_col = "rDII_quintile",
    ax = axes[0, 2]
)

# Plot4
plot_boxplot_for_group(
    data = DPs_Div,
    method = "kruskal",
    x_col = "AMED_quintile",
    y_col = "Shannon Index",
    group_col = "AMED_quintile",
    ax = axes[0, 3]
)

# Plot5
plot_boxplot_for_group(
    data = DPs_Div,
    method = "kruskal",
    x_col = "rEDIH_quintile",
    y_col = "Shannon Index",
    group_col = "rEDIH_quintile",
    ax = axes[0, 4]
)

# Plot6
plot_boxplot_for_group(
    data = DPs_Div,
    method = "kruskal",
    x_col = "AHEI_quintile",
    y_col = "Simpson Index",
    group_col = "AHEI_quintile",
    ax = axes[1, 0]
)

# Plot7
plot_boxplot_for_group(
    data = DPs_Div,
    method = "kruskal",
    x_col = "hPDI_quintile",
    y_col = "Simpson Index",
    group_col = "hPDI_quintile",
    ax = axes[1, 1]
)

# Plot8
plot_boxplot_for_group(
    data = DPs_Div,
    method = "kruskal",
    x_col = "rDII_quintile",
    y_col = "Simpson Index",
    group_col = "rDII_quintile",
    ax = axes[1, 2]
)

# Plot9
plot_boxplot_for_group(
    data = DPs_Div,
    method = "kruskal",
    x_col = "AMED_quintile",
    y_col = "Simpson Index",
    group_col = "AMED_quintile",
    ax = axes[1, 3]
)

# Plot10
plot_boxplot_for_group(
    data = DPs_Div,
    method = "kruskal",
    x_col = "rEDIH_quintile",
    y_col = "Simpson Index",
    group_col = "rEDIH_quintile",
    ax = axes[1, 4]
)

# Add labels
axes[0,0].text(-0.1, 1.1, "B", transform = axes[0,0].transAxes, fontsize = 16, fontweight = "bold", va = "top", ha = "right")
axes[1,0].text(-0.1, 1.1, "C", transform = axes[1,0].transAxes, fontsize = 16, fontweight = "bold", va = "top", ha = "right")

# Save the figure
plt.savefig("./Results/Shannon_Diversity_Comparison.pdf", bbox_inches = "tight", dpi = 300)
# plt.show()


# ## CLR transformation

# In[54]:


## clr transformation
## data_species_full_2_2 = data_species_full_2_2.apply(lambda x: x/100)
data_species_full_qc_2 = data_species_full_qc_2.set_index("participant_id")
from skbio.stats.composition import clr
data_species_full_qc_clr = clr(data_species_full_qc_2)


# In[55]:


data_species_full_qc_clr


# In[56]:


colnames_qc = data_species_full_qc_2.columns
index_qc = data_species_full_qc_2.index
data_species_full_qc_clr = pd.DataFrame(data_species_full_qc_clr, 
                                        index = index_qc, 
                                        columns = colnames_qc)


# In[57]:


data_species_full_qc_clr.columns[:5]


# In[58]:


import seaborn as sns
# sns.histplot(data=data_species_full_qc_clr, x="g__Bifidobacterium|s__Bifidobacterium_adolescentis")
# s__GGB9059_SGB13976
# sns.histplot(data=data_species_full_qc_clr, x="g__Bifidobacterium|s__Bifidobacterium_angulatum")
# s__Methanobrevibacter_smithii
sns.histplot(data=data_species_full_qc_clr, x="f__Methanobacteriaceae|g__Methanobrevibacter|s__Methanobrevibacter_smithii")


# In[59]:


data_species_full_qc_clr.describe()

mean = data_species_full_qc_clr.mean()
std = data_species_full_qc_clr.std()


# In[60]:


data_species_full_qc_clr_scale = (data_species_full_qc_clr - mean)/std
data_species_full_qc_clr_scale

# print(data_species_full_qc_clr_scale.describe())
sns.histplot(data=data_species_full_qc_clr_scale, x = "f__Methanobacteriaceae|g__Methanobrevibacter|s__Methanobrevibacter_smithii")


# In[61]:


# scale using standardScaler to normalize

# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# def scale_df(df, method = "standard"):
#     # select the numeric columns
#     numeric_columns = df.select_dtypes(include = ['float64', 'int64']).columns
#     # create a copy of the Dataframe
#     df_scaled = df.copy()
    
#     # select method
#     if method.lower() == "standard":
#         scaler = StandardScaler()
#     elif method.lower() == "minmax":
#         scaler = MinMaxScaler()
#     else:
#         raise ValueError("No methods available")
        
#     # scale columns
#     if len(numeric_columns) > 0:
#         df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
#     return df_scaled


# In[62]:


data_species_full_qc_clr_scale


# In[63]:


data_species_full_qc_clr_scale.to_csv('./Data/Species_level_clr_scaled.csv', index = True)
data_species_full_qc_clr.to_csv("./Data/Species_level_clr.csv", index = True)


# # Including the covariates into model

# In[64]:


lifestyle_factors = pd.read_csv("../02_Lifestyle factors/lifestyle_factor_all_disease_V2.csv")
# lifestyle_factors
data_species_full_qc_clr = pd.read_csv("./Data/Species_level_clr_scaled.csv")
data_species_full_qc_clr


# In[65]:


gut_life_df = pd.merge(data_species_full_qc_clr, lifestyle_factors, on = "participant_id", how = "left")
gut_life_df


# In[66]:


DPs_gut_life_df = pd.merge(gut_life_df, DPs, on = "participant_id", how = "left")
DPs_gut_life_df


# In[67]:


alpha_div_copy = alpha_div_copy.reset_index()

DPs_gut_life_df_diversity = pd.merge(DPs_gut_life_df, alpha_div_copy, on = "participant_id", how = "left")
DPs_gut_life_df_diversity


# In[68]:


# beta_diversity_results.to_csv("beta_diversity_results.csv", index = False)
beta_diversity_results = pd.read_csv("./Data/beta_diversity_results_2.csv")
beta_diversity_results


# In[69]:


DPs_gut_life_df_diversity2 = pd.merge(DPs_gut_life_df_diversity, beta_diversity_results, on = "participant_id", how = "left")
DPs_gut_life_df_diversity2.shape

DPs_gut_life_df_diversity2.to_csv("./Data/DPs_gut_life_df_diversity_species_20251023_scaled.csv", index = False)


# ## DPs_gut_life_df_diversity2 -> for analysis

# In[70]:


import pandas as pd

DPs_gut_life_df_diversity2 = pd.read_csv("./Data/DPs_gut_life_df_diversity_species_20251023_scaled.csv")

DPs_gut_life_df_diversity2.loc[:, DPs_gut_life_df_diversity2.columns.str.contains("sex")]


# In[71]:


print(DPs_gut_life_df_diversity2['sex'].value_counts())
print(DPs_gut_life_df_diversity2['sex_y'].value_counts())
DPs_gut_life_df_diversity2['sex_x'].value_counts() # sex_x have the most information about sex


# In[72]:


# focus on the complete case
# Male = 1
DPs_gut_life_df_diversity2['sex_all'] = DPs_gut_life_df_diversity2[DPs_gut_life_df_diversity2['sex_x'].notnull()]['sex_x'].apply(lambda x: 1 if x == "Male" else 0) 
DPs_gut_life_df_diversity2['sex_all'].value_counts()


# In[73]:


#DPs (scaled) the age(scaled), bmi(scaled), and sex
def scaling(df, col_to_scaling):
    df_scaling = df.copy()
    
    for column in col_to_scaling:
        # calculate the mean and std first
        mean_val = df_scaling[column].mean()
        std_val = df_scaling[column].std()
        
        df_scaling[f"{column}_scaled"] = (df_scaling[column] - mean_val)/std_val
        
    return df_scaling  

col_to_scaling = ['AHEI_2010_score_eadj', 'hPDI_score_eadj', 'rDII_score_eadj', 'AMED_score_eadj', 'rEDIH_score_all_eadj']
df_ana_scaled = scaling(DPs_gut_life_df_diversity2, col_to_scaling = col_to_scaling)
df_ana_scaled.head()


# In[74]:


print(df_ana_scaled['age'].isnull().sum()) # 222 without age 
print(df_ana_scaled['bmi'].isnull().sum()) # 231 withou bmi
df_ana_scaled['AHEI_2010_score_eadj_scaled'].isnull().sum() # 1158 withou diet information


# In[75]:


df_ana_scaled.shape
df_ana_scaled.to_csv("./Data/Data_analysis_without_filtering_missingDPs_Species_20251023_scaled.csv", index = False)


# In[76]:


# remove the rows if variable are all missing in the columns
cols_check = ["AHEI_2010_score_eadj_scaled", "hPDI_score_eadj_scaled", "rDII_score_eadj_scaled", 
              "AMED_score_eadj_scaled", "rEDIH_score_all_eadj_scaled"]
df_ana_scaled2 = df_ana_scaled.dropna(subset = cols_check, how = "all")
df_ana_scaled2.shape # 7917 obs


# In[77]:


df_ana_scaled2.to_csv("./Data/Data_species_analysis_20251023_scaled.csv", index = False)
df_ana_scaled2.to_csv("./Data/Data_species_analysis_20251023_scaled.csv", index = False)


# # Using the formula way to perform the analysis
# - See another python file

# In[ ]:


## V2, sm.from_formula() without dummies determined

import statsmodels.api as sm


# In[ ]:


# All columns (except medv, which is our response)
# model = sm.OLS.from_formula('g__Bifidobacterium_s__Bifidobacterium_adolescentis ~ ' + '+'.join(["AHEI_2010_score_eadj_scaled", 'sex_all', 'age', 'smoking_status', 'edu_status', 
#          'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']), df_ana_scaled2, missing = "drop")
# result = model.fit()
# result.summary()


# In[ ]:


import pandas as pd
df_ana_scaled2 = pd.read_csv("./Data/Data_species_analysis_20251023_scaled.csv")
df_ana_scaled2.head()


# In[ ]:


# sex (1: male)
df_ana_scaled2.loc[:,'sex_all'] =  df_ana_scaled2['sex_all'].astype('category')
df_ana_scaled2.loc[:,"sex_status"] = pd.get_dummies(df_ana_scaled2['sex_all'], prefix='sex', drop_first=True)
df_ana_scaled2["sex_status"]


# In[ ]:


Predictors = [item for item in data_species_full_qc_clr_scale.columns if item != 'participant_id']
len(Predictors)

# outcomes
Outcomes = [item for item in data_species_full_qc_clr_scale.columns if item != 'participant_id']
len(Outcomes)


# In[ ]:


df_ana_scaled2_1 = df_ana_scaled2.dropna(subset = ['smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"])
df_ana_scaled2_1.shape 


# In[ ]:


smoking_status = pd.get_dummies(df_ana_scaled2_1['smoking_status'], prefix='smoking_status', drop_first=False)
smoking_status = smoking_status.drop('smoking_status_Never', axis=1) 
smoking_status

education_status = pd.get_dummies(df_ana_scaled2_1['edu_status'], prefix='education', drop_first=False)
education_status = education_status.drop('education_Low', axis=1)
education_status.shape

Vit_use_status = pd.get_dummies(df_ana_scaled2_1['Vit_use'], prefix='Vit_use', drop_first=False)
Vit_use_status.columns = Vit_use_status.columns.str.replace(' ', '_')
Vit_use_status = Vit_use_status.drop('Vit_use_not_use', axis = 1)
Vit_use_status.head()

Hormone_use_status = pd.get_dummies(df_ana_scaled2_1['Hormone_use'], prefix="Hormone_use", drop_first = False)
Hormone_use_status.columns = Hormone_use_status.columns.str.replace(' ', '_')
Hormone_use_status = Hormone_use_status.drop('Hormone_use_not_use', axis = 1)
Hormone_use_status

df_ana_scaled2_2 = pd.concat([df_ana_scaled2_1, smoking_status, education_status, 
                                           Vit_use_status, Hormone_use_status], axis = 1)
df_ana_scaled2_2


# In[ ]:


# including the alcohol_use
#  hPDI should be adjusted also for alcohol intakes

Nutrients = pd.read_csv("Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
Nutrients_2.info()

# merge with alcohol intake
df_ana_scaled2_2 = pd.merge(df_ana_scaled2_2, Nutrients_2, on = "participant_id", how = 'left')
df_ana_scaled2_2


# In[ ]:


# model4
df_ana_scaled2_model4 = df_ana_scaled2_2.dropna(subset = ['NSAID_use','PPI_use', "Antibio_use", 'T2D_f_history', "CVD_f_history"])
print(df_ana_scaled2_model4.shape) #3983 

df_ana_scaled2_model4['PPI_use'].value_counts()


# In[ ]:


# NSAID use 
NSAID_use_status = pd.get_dummies(df_ana_scaled2_model4['NSAID_use'], prefix="NSAID_use", drop_first = False)
NSAID_use_status.columns = NSAID_use_status.columns.str.replace(' ', '_')
NSAID_use_status = NSAID_use_status.drop(columns = "NSAID_use_not_use")
NSAID_use_status.head()


# In[ ]:


# PPI_use
PPI_use_status = pd.get_dummies(df_ana_scaled2_model4['PPI_use'], prefix="PPI_use")
PPI_use_status = PPI_use_status.drop(columns = "PPI_use_not use")
PPI_use_status.head()

# Antibio_use
Antibio_use_status = pd.get_dummies(df_ana_scaled2_model4['Antibio_use'], prefix="Antibio_use")
Antibio_use_status = Antibio_use_status.drop(columns = "Antibio_use_not use")
Antibio_use_status.head()


# In[ ]:


# CVD_f_history
CVD_f_history_status = pd.get_dummies(df_ana_scaled2_model4['CVD_f_history'], prefix="CVD_f_history", drop_first = True)
CVD_f_history_status.head()

# T2D_f_history
T2D_f_history_status = pd.get_dummies(df_ana_scaled2_model4['T2D_f_history'], prefix="T2D_f_history", drop_first = True)
T2D_f_history_status.head()

df_ana_scaled2_model4 = pd.concat([df_ana_scaled2_model4, NSAID_use_status, CVD_f_history_status, T2D_f_history_status, 
                                   PPI_use_status, Antibio_use_status],
                          axis = 1)
df_ana_scaled2_model4.shape


# In[ ]:


# model1 = ['sex_status', 'bmi_scaled', 'age_scaled']
model1 = None

model2 = ['sex_status', 'age', 'smoking_status_Current', 'smoking_status_Former', 'education_High', 
         'Vit_use_Use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use_Use']

# for hPDI index adjusted for alcohol intake 
model2_2 = ['sex_status', 'age', 'smoking_status_Current', 'smoking_status_Former', 'education_High', 
         'Vit_use_Use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use_Use', "alcohol_g"]

model3 = ['sex_status', 'age', 'smoking_status_Current', 'smoking_status_Former', 'education_High', 
         'Vit_use_Use', 'Hormone_use_Use', 'MET_hour', 'sleep_hours_daily', 'bmi']

# for hPDI index adjusted for alcohol intake 
model3_2 = ['sex_status', 'age', 'smoking_status_Current', 'smoking_status_Former', 'education_High', 
         'Vit_use_Use', 'Hormone_use_Use', 'MET_hour', 'sleep_hours_daily', 'alcohol_g', 'bmi']

model4 = ['sex_status', 'age', 'smoking_status_Current', 'smoking_status_Former', 'education_High', 
         'Vit_use_Use', 'Hormone_use_Use', 'NSAID_use_Use', 'MET_hour', 'sleep_hours_daily', 
          "CVD_f_history_yes", 'T2D_f_history_yes', 'PPI_use_Use', 'Antibio_use_Use']

# for hPDI index adjusted for alcohol intake 
model4_2 = ['sex_status', 'age', 'smoking_status_Current', 'smoking_status_Former', 'education_High', 
         'Vit_use_Use', 'Hormone_use_Use', 'NSAID_use_Use', 'MET_hour', 'sleep_hours_daily', 
          "CVD_f_history_yes", 'T2D_f_history_yes', 'alcohol_g', 'PPI_use_Use', 'Antibio_use_Use']

Predictors = ["AHEI_2010_score_eadj_scaled", "hPDI_score_eadj_scaled", 
            "rDII_score_eadj_scaled", 
            "AMED_score_eadj_scaled", "rEDIH_score_all_eadj_scaled"]


# In[ ]:


# Add run_multiple_regression function
# Stop here ==================


# In[ ]:


results_df_model1_test = pd.DataFrame()

# test model
for predict_var in Predictors:
    print(predict_var)
    for out in Outcomes:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_multiple_regressions(df = df_ana_scaled2_2, predictors = [predict_var], outcomes = [out], control_vars = model1)
        else:
            lm_fit = run_multiple_regressions(df = df_ana_scaled2_2, predictors = [predict_var], outcomes = [out], control_vars = model1)
        results_df_model1_test = pd.concat([results_df_model1_test, lm_fit], ignore_index=True)
        

results_df_model1_test['p_adj'] = stats.false_discovery_control(results_df_model1_test['P_Value'])
results_df_model1_test


# In[ ]:


results_df_model1_test_sig = results_df_model1_test.loc[results_df_model1_test['p_adj'] < 0.05] # select based on FDR < 0.05
results_df_model1_test_sig


# In[ ]:


results_df_model2_test = pd.DataFrame()

# test model
for predict_var in Predictors:
    print(predict_var)
    for out in Outcomes:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_multiple_regressions(df = df_ana_scaled2_2, predictors = [predict_var], outcomes = [out], control_vars = model2_2)
        else:
            lm_fit = run_multiple_regressions(df = df_ana_scaled2_2, predictors = [predict_var], outcomes = [out], control_vars = model2)
        results_df_model2_test = pd.concat([results_df_model2_test, lm_fit], ignore_index=True)
        
results_df_model2_test
results_df_model2_test['p_adj'] = stats.false_discovery_control(results_df_model2_test['P_Value'])

results_df_model2_test.sort_values(by = ["Variable", "Coefficient"]).to_csv('dps_to_microbiomes_model2_full.csv', index = False)

# multiple correction
results_df_model2_test_sig = results_df_model2_test.loc[results_df_model2_test['p_adj'] < 0.05] # select based on FDR < 0.05
results_df_model2_test_sig = results_df_model2_test_sig.sort_values(by = ["Coefficient", "Variable"])
results_df_model2_test_sig.to_csv('dps_to_microbiomes_model2_sig.csv', index = False)


# In[ ]:


results_df_model2_test_sig


# In[ ]:


# model3
results_df_model3_test = pd.DataFrame()

# test model
for predict_var in Predictors:
    print(predict_var)
    for out in Outcomes:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_multiple_regressions(df = df_ana_scaled2_2, predictors = [predict_var], outcomes = [out], control_vars = model3_2)
        else:
            lm_fit = run_multiple_regressions(df = df_ana_scaled2_2, predictors = [predict_var], outcomes = [out], control_vars = model3)
        results_df_model3_test = pd.concat([results_df_model3_test, lm_fit], ignore_index=True)
        
results_df_model3_test
results_df_model3_test['p_adj'] = stats.false_discovery_control(results_df_model3_test['P_Value'])

results_df_model3_test_sig = results_df_model3_test.loc[results_df_model3_test['p_adj'] < 0.05]
results_df_model3_test_sig = results_df_model3_test_sig.sort_values(by = ["Coefficient", "Variable"])
results_df_model3_test_sig


# In[ ]:


# model4
results_df_model4_test = pd.DataFrame()

# test model
for predict_var in Predictors:
    print(predict_var)
    for out in Outcomes:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_multiple_regressions(df = df_ana_scaled2_model4, predictors = [predict_var], outcomes = [out], control_vars = model4_2)
        else:
            lm_fit = run_multiple_regressions(df = df_ana_scaled2_model4, predictors = [predict_var], outcomes = [out], control_vars = model4)
        results_df_model4_test = pd.concat([results_df_model4_test, lm_fit], ignore_index=True)
        
results_df_model4_test
results_df_model4_test['p_adj'] = stats.false_discovery_control(results_df_model4_test['P_Value'])

results_df_model4_test_sig = results_df_model4_test.loc[results_df_model4_test['p_adj'] < 0.05]
results_df_model4_test_sig = results_df_model4_test_sig.sort_values(by = ["Coefficient", "Variable"])
results_df_model4_test_sig


# In[ ]:


results_df_model1_test.sort_values(by = ["Variable", "Coefficient"]).to_csv('dps_to_microbiomes_model1_full.csv', index = False)
results_df_model1_test_sig.sort_values(by = ["Variable", "Coefficient"]).to_csv('dps_to_microbiomes_model1_sig.csv', index = False)
results_df_model2_test.sort_values(by = ["Variable", "Coefficient"]).to_csv('dps_to_microbiomes_model2_full.csv', index = False)
results_df_model2_test_sig.sort_values(by = ["Variable", "Coefficient"]).to_csv('dps_to_microbiomes_model2_sig.csv', index = False)
results_df_model3_test.sort_values(by = ["Variable", "Coefficient"]).to_csv('dps_to_microbiomes_model3_full.csv', index = False)
results_df_model3_test_sig.sort_values(by = ["Variable", "Coefficient"]).to_csv('dps_to_microbiomes_model3_sig.csv', index = False)
results_df_model4_test.sort_values(by = ["Variable", "Coefficient"]).to_csv('dps_to_microbiomes_model4_full.csv', index = False)
results_df_model4_test_sig.sort_values(by = ["Variable", "Coefficient"]).to_csv('dps_to_microbiomes_model4_sig.csv', index = False)


# In[ ]:


print(results_df_model2_test.shape)
results_df_model2_test_sig[['genus', "species"]] = results_df_model2_test_sig['Outcome'].str.split("|", expand = True)
results_df_model2_test_sig


# In[ ]:


# Reorder columns
results_df_model2_test_sig = results_df_model2_test_sig[["N", "Variable", "Outcome", "genus", "species"] + 
                                                        [col for col in results_df_model2_test_sig.columns if col not in ["N", "Variable", "Outcome", "genus", "species"]]]

results_df_model2_test_sig


# In[ ]:


# counts on the genus
results_df_model2_test_sig["genus"].value_counts().sort_values(ascending=False)[0:20]


# In[ ]:


# plot value_counts of Series
ax = results_df_model2_test_sig["genus"].value_counts().sort_values(ascending=False)[0:20].plot(kind='barh')
ax

import matplotlib.pyplot as plt
plt.savefig("genus_plot.pdf", format = "pdf", bbox_inches = "tight")
plt.show()


# In[ ]:


results_df_model2_test_sig


# In[ ]:


ct = pd.crosstab(results_df_model2_test_sig['Variable'], results_df_model2_test_sig['Outcome'])
filter_ct = ct.loc[:, ct.sum(axis = 0) == 5]
filter_columns = filter_ct.columns


# In[ ]:


filter_columns


# In[ ]:


results_df_model2_test_sig


# In[ ]:


results_df_model2_test_sig2 = results_df_model2_test_sig.loc[results_df_model2_test_sig["Outcome"].isin(filter_columns), :]
results_df_model2_test_sig2


# In[ ]:


print(len(results_df_model2_test_sig2.nlargest(60, 'Coefficient')['Outcome'].unique()))
# 
len(results_df_model2_test_sig2.nsmallest(68, 'Coefficient')['Outcome'].unique())


# In[ ]:


# outcomes 2 (considering diversity)
DPs = ["AHEI_2010_score_eadj_scaled", "hPDI_score_eadj_scaled", 
       "rDII_score_eadj_scaled", "AMED_score_eadj_scaled", "rEDIH_score_all_eadj_scaled"]

outcomes_2 = ['shannon_skbio_log', 'simpson_skbio_log', 'PC1', 'PC2']

results_df_model2_2 = pd.DataFrame()

# test model
for diversity in outcomes_2:
    print(diversity)
    for predict_var in DPs:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_multiple_regressions(df = df_ana_scaled2_2, predictors = [predict_var], outcomes = [diversity], control_vars = model2_2)
        else:
            lm_fit = run_multiple_regressions(df = df_ana_scaled2_2, predictors = [predict_var], outcomes = [diversity], control_vars = model2)
        results_df_model2_2 = pd.concat([results_df_model2_2, lm_fit], ignore_index=True)
        
results_df_model2_2

from scipy import stats
results_df_model2_2['p_adj'] = stats.false_discovery_control(results_df_model2_2['P_Value'])

results_df_model2_2_sig = results_df_model2_2.loc[results_df_model2_2['p_adj'] < 0.05] # select based on FDR < 0.05
results_df_model2_2_sig = results_df_model2_2_sig.sort_values(by = ["Coefficient", "Variable"])


# In[ ]:


results_df_model2_2_sig


# In[ ]:


# results_df_model2 = pd.DataFrame()

# # test model
# for DPs_out in outcomes:
#     print(DPs_out)
#     for predict_var in Predictors:
#         if DPs_out == "hPDI_score_eadj_scaled":
#             lm_fit = run_multiple_regressions(df = df_ana_scaled2_2, predictors = [predict_var], outcomes = [DPs_out], control_vars = model2_2)
#         else:
#             lm_fit = run_multiple_regressions(df = df_ana_scaled2_2, predictors = [predict_var], outcomes = [DPs_out], control_vars = model2)
#         results_df_model2 = pd.concat([results_df_model2, lm_fit], ignore_index=True)

# p.adjusted value
# from scipy import stats
# results_df_model2['p_adj'] = stats.false_discovery_control(results_df_model2['P_Value'])
# results_df_model2


# In[ ]:


# results_Micro_DPs.sort_values(by = ["Coefficient", "Variable"]).to_csv("results_Micro_DPs.csv", index = False)
# results_df_model2.sort_values(by = ["Coefficient", "Variable"]).to_csv("results_Micro_DPs_model2.csv", index = False)


# In[ ]:


## select significant microbes
# results_df_model2_sig = results_df_model2.loc[results_df_model2['p_adj'] < 0.05] # select based on FDR < 0.05
# results_df_model2_sig = results_df_model2_sig.sort_values(by = ["Coefficient", "Variable"])
# results_df_model2_sig.sort_values(by = ["Coefficient", "Variable"]).to_csv("results_Micro_DPs_model2_sig.csv", index = False)


# In[ ]:


# results_df_model2_sig


# In[ ]:


results_df_model2_test_sig2


# In[ ]:


# get the top 20 and bottom 20 variables
top20 = results_df_model2_test_sig2.nlargest(60, 'Coefficient')['Outcome'].unique()
top20


# In[ ]:


Least20 = results_df_model2_test_sig2.nsmallest(68, 'Coefficient')['Outcome'].unique()
Least20
len(Least20)


# In[ ]:


# plot the results
results_df_model2_test_sig2 = results_df_model2_test_sig2.assign(
    Variable = results_df_model2_test_sig2['Variable'].astype('str'),
    Outcome = results_df_model2_test_sig2['Outcome'].astype('str')
)

# plot the heatmap for the coefficient
Microbe_heatmap = results_df_model2_test_sig2.pivot(index = "Outcome", columns= "Variable", values = "Coefficient")
Microbe_heatmap


# In[ ]:


Microbe_heatmap.rename(index = {'rEDIH_score_all_eadj_scaled':'rEDIH_score_eadj_scaled'}, inplace = True)
Microbe_heatmap 


# In[ ]:


# # Selection the most significant ones
Microbe_heatmap_t = Microbe_heatmap.transpose()
Microbe_heatmap_t


# In[ ]:


TOP20_plot = Microbe_heatmap_t[top20]
TOP20_plot

Least20_plot = Microbe_heatmap_t[Least20]
Least20_plot


# In[ ]:


# plt.figure(figsize=(120,60))
import seaborn as sns
# sns.set_style('whitegrid')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,10))

hp_model1 = sns.heatmap(TOP20_plot.iloc[:,1:10], 
            cmap='RdBu_r',  # Red-Blue diverging colormap
            center=0,       # Center the colormap at 0
            annot=True,     # Show values in cells
            fmt='.2f',      # Format for the annotations (2 decimal places)
            cbar_kws={'label': 'Coefficient',
                     'fraction': 0.05,
                     'shrink': 0.5,
                     'aspect': 40,
                     'pad': 0.01,
                     "orientation": "horizontal"}, # pad change the distance between heatmap and legend
            square=True,
            linewidths = 1,
            annot_kws = {'size':16},
            linecolor='black',
#             xticklabels=False,
            ax = ax)    # Make cells square

# legend = plt.colorbar(hm.collections[0])
# legend.set_label('Coefficient')
hp_model1.set_xticklabels(hp_model1.get_xticklabels(), size=14)  # Change size as needed
hp_model1.set_yticklabels(hp_model1.get_yticklabels(), size=14)  # Change size as needed

# increase axis title (larger than tick labels)
hp_model1.set_ylabel('', fontsize=16)  # Axis title bigger

# Rotate x-axis labels for better readability
plt.xlabel('') # remove the x-axis title
plt.tick_params(axis='x', top=True, bottom=False,
                labeltop = True, labelbottom=False)
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)

# Adjust layout to prevent label cutoff
# Gives more space for y-axis labels
# plt.subplots_adjust(left=0.2)  
plt.tight_layout()

# plt.savefig("heatmap_model1_crude.jpg", bbox_inches = 'tight')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))

hp_model1 = sns.heatmap(Least20_plot.iloc[:,0:10], 
            cmap='RdBu_r',  # Red-Blue diverging colormap
            center=0,       # Center the colormap at 0
            annot=True,     # Show values in cells
            fmt='.2f',      # Format for the annotations (2 decimal places)
            cbar_kws={'label': 'Coefficient',
                     'fraction': 0.05,
                     'shrink': 0.5,
                     'aspect': 40,
                     'pad': 0.01,
                     "orientation": "horizontal"}, # pad change the distance between heatmap and legend
            square=True,
            linewidths = 1,
            annot_kws = {'size':16},
            linecolor='black',
#             xticklabels=False,
            ax = ax)    # Make cells square

# legend = plt.colorbar(hm.collections[0])
# legend.set_label('Coefficient')
hp_model1.set_xticklabels(hp_model1.get_xticklabels(), size=14)  # Change size as needed
hp_model1.set_yticklabels(hp_model1.get_yticklabels(), size=14)  # Change size as needed

# increase axis title (larger than tick labels)
hp_model1.set_ylabel('', fontsize=16)  # Axis title bigger

# Rotate x-axis labels for better readability
plt.xlabel('') # remove the x-axis title
plt.tick_params(axis='x', top=True, bottom=False,
                labeltop = True, labelbottom=False)
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)

# Adjust layout to prevent label cutoff
# Gives more space for y-axis labels
# plt.subplots_adjust(left=0.2)  
plt.tight_layout()

# plt.savefig("heatmap_model1_crude.jpg", bbox_inches = 'tight')
plt.show()


# # Calculate the tertile for five dietary pattern scores

# In[ ]:


cols_check = ["AHEI_2010_score_eadj_scaled", "hPDI_score_eadj_scaled", "rDII_score_eadj_scaled", 
              "AMED_score_eadj_scaled", "rEDIH_score_all_eadj_scaled"]
df_ana_scaled_pca = df_ana_scaled.dropna(subset = cols_check)


# In[ ]:


df_ana_scaled_pca_copy = df_ana_scaled_pca.copy()
for i in cols_check:
    print(i)
    # generate the tertile
    df_ana_scaled_pca_copy[f"{i}_tertile"] = pd.qcut(df_ana_scaled_pca_copy[i], q=3, labels=['Low', 'Medium', 'High'])

df_ana_scaled_pca_copy


# In[ ]:


df_ana_scaled_pca_copy_2 = df_ana_scaled_pca.copy()
for i in cols_check:
    print(i)
    # generate the tertile
    df_ana_scaled_pca_copy_2[f"{i}_quintile"] = pd.qcut(df_ana_scaled_pca_copy_2[i], q=5, labels=['Q1', "Q2", 'Q3', "Q4",'Q5'])

df_ana_scaled_pca_copy_2


# In[ ]:


import matplotlib.pyplot as plt
# create a figure with 2*3 subplots
fig, axes = plt.subplots(2, 3, figsize = (20,10))

sns.scatterplot(data=df_ana_scaled_pca_copy_2, x="PC1", y="PC2", hue = "AHEI_2010_score_eadj_scaled_quintile", ax = axes[0,0])
sns.scatterplot(data=df_ana_scaled_pca_copy_2, x="PC1", y="PC2", hue = "rDII_score_eadj_scaled_quintile", ax = axes[0,1])
sns.scatterplot(data=df_ana_scaled_pca_copy_2, x="PC1", y="PC2", hue = "hPDI_score_eadj_scaled_quintile", ax = axes[0,2])
sns.scatterplot(data=df_ana_scaled_pca_copy_2, x="PC1", y="PC2", hue = "AMED_score_eadj_scaled_quintile", ax = axes[1,0])
sns.scatterplot(data=df_ana_scaled_pca_copy_2, x="PC1", y="PC2", hue = "rEDIH_score_all_eadj_scaled_quintile", ax = axes[1,1])

plt.tight_layout()


# In[ ]:


import matplotlib.pyplot as plt
# create a figure with 2*3 subplots
fig, axes = plt.subplots(2, 3, figsize = (20,10))

sns.scatterplot(data=df_ana_scaled_pca_copy, x="PC1", y="PC2", hue = "AHEI_2010_score_eadj_scaled_tertile", ax = axes[0,0])
sns.scatterplot(data=df_ana_scaled_pca_copy, x="PC1", y="PC2", hue = "rDII_score_eadj_scaled_tertile", ax = axes[0,1])
sns.scatterplot(data=df_ana_scaled_pca_copy, x="PC1", y="PC2", hue = "hPDI_score_eadj_scaled_tertile", ax = axes[0,2])
sns.scatterplot(data=df_ana_scaled_pca_copy, x="PC1", y="PC2", hue = "AMED_score_eadj_scaled_tertile", ax = axes[1,0])
sns.scatterplot(data=df_ana_scaled_pca_copy, x="PC1", y="PC2", hue = "rEDIH_score_all_eadj_scaled_tertile", ax = axes[1,1])

plt.tight_layout()


# In[ ]:


# print(df_ana_scaled.columns.tolist())
df_ana_scaled["Vit_use"].value_counts()


# In[ ]:


# DPs_gut_life_df_diversity2


# ### Use LASSO-CV selection the feature selection (clr-LASSO)

# In[ ]:


from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression


# In[ ]:


# df_ana_scaled2


# In[ ]:


# X = df_ana_scaled2.iloc[:, 1:380]
# Y = df_ana_scaled2['rDII_score_eadj_scaled']


# In[ ]:


# X1 = pd.concat([X, df_ana_scaled2[['sex_status', 'bmi_scaled', 'age_scaled']]], axis = 1)
# X1             


# In[ ]:


# X1.shape, Y.shape


# In[ ]:


# # Create a LassoCV object
# lasso_cv = LassoCV(cv=10, random_state=123)  # cv=10 means 10-fold cross-validation

# # Fit the model to the training data
# reg = lasso_cv.fit(X, Y)


# In[ ]:


# print(f"Number of non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)}") # 255 still in the model


# In[ ]:


# len(X1.columns) # 379 -> 255 still significant


# ## Change variable into tertiles

# In[ ]:


df_ana_scaled2 = df_ana_scaled2.set_index("participant_id")


# In[ ]:


relative_abundance_matrix =  df_ana_scaled2.iloc[:, 0:379]
relative_abundance_matrix


# In[ ]:


# pcoa_results = pcoa(beta_diversity)


# In[ ]:


beta_div = pd.DataFrame({
    'participant_id': pcoa_results.samples['PC1'].index,
    'PC1': pcoa_results.samples['PC1'].values,
    'PC2': pcoa_results.samples['PC2'].values
})
beta_div

beta_div.to_csv("beta_PC1AND2.csv", index = False)


# In[ ]:


df_ana_scaled2_copy = df_ana_scaled2_copy.reset_index()
df_ana_scaled2_copy


# In[ ]:


# merge with PC
df_ana_scaled2_merge = pd.merge(df_ana_scaled2_copy, beta_div,
                               on = "participant_id", how = "left")
df_ana_scaled2_merge



# ============================================================
Source: 02_01_MLR_analysis_Species.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Using formula way to perform analysis in python
# Author: Keyong Deng

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
    
import pandas as pd
import importlib # Import the importlib library

# --- Make changes to A0_helpfunction.py and save them ---
# importlib.reload(A0_helpfunction) # Reload the module
from A0_helpfunction import *
from sklearn.preprocessing import StandardScaler


# In[2]:


# remove the rows if variable are all missing in the columns
df_ana_scaled = pd.read_csv("./Data/Data_species_analysis_20251023_scaled.csv")
df_ana_scaled.shape


# In[77]:


df_ana_scaled.columns


# In[78]:


df_ana_scaled["rDII_score_eadj"].describe()


# In[3]:


df_ana_scaled.head()


# In[4]:


print(df_ana_scaled.columns[df_ana_scaled.columns.str.contains(r'use')])
df_ana_scaled["Vit_use"].value_counts()

Nutrients = pd.read_csv("Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
Nutrients_2.info()

# merge with alcohol intake
df_ana_scaled = pd.merge(df_ana_scaled, Nutrients_2, on = "participant_id", how = 'left')
df_ana_scaled


# In[5]:


print(df_ana_scaled["age"].isnull().sum())
df_ana_scaled["sex_all"].isnull().sum()
df_ana_scaled["alcohol_g"].describe()


# In[6]:


temp = pd.read_csv("../02_Lifestyle factors/lifestyle_factor_all_disease_V2.csv")
temp

df_ana_scaled = pd.merge(df_ana_scaled, temp[["participant_id", "age_all"]], on = "participant_id", how = "left")

df_ana_scaled["age"].isnull().sum()


# In[7]:


df_ana_scaled["CVD_f_history"].isnull().sum()
df_ana_scaled["CVD_f_history"].value_counts()


# ## Calculate the associations between DPs and Microbiome diversity

# In[8]:


model2 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']

# for hPDI index adjusted for alcohol intake 
model2_2 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use',"alcohol_g"]

model3 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "bmi"]

model3_2 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "bmi", 'alcohol_g']

# we don't adjust for the family history fo disease here 
model4 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'Hormone_use', 'NSAID_use', 'MET_hour', 'sleep_hours_daily', 
          'PPI_use', 'Antibio_use']

model4_2 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'Hormone_use', 'NSAID_use', 'MET_hour', 'sleep_hours_daily', 
          'PPI_use', 'Antibio_use', "alcohol_g"]


# In[9]:


def run_models(model_name, control_vars1, control_vars2, df_data, predictors, outcomes):
    """Run regression tests for a given model"""
    results_df = pd.DataFrame()
    for predict_var in predictors:
        print(predict_var)
        for out in outcomes:
            if predict_var == "hPDI_score_eadj_scaled":
                lm_fit = run_regression_formula(df=df_data, 
                                              predictors=[predict_var], 
                                              outcomes=out,
                                              control_vars=control_vars1)
            else:
                lm_fit = run_regression_formula(df=df_data,
                                              predictors=[predict_var],
                                              outcomes=out,
                                              control_vars=control_vars2)
            
            lm_fit_df = pd.DataFrame(lm_fit)
            
            # Add the model name into the results
            lm_fit_df["model_name"] = model_name
            # lm_fit_df["predictor"] = predict_var
            # lm_fit_df["outcome"] = out
            # concatenate results
            results_df = pd.concat([results_df, lm_fit_df], axis=0, ignore_index=True)
    
    return results_df


# In[10]:


Predictors = ["AHEI_2010_score_eadj_scaled", "hPDI_score_eadj_scaled", 
            "rDII_score_eadj_scaled", 
            "AMED_score_eadj_scaled", "rEDIH_score_all_eadj_scaled"]

# Outcomes = ["shannon_skbio_scaled"]
# Outcomes = ["shannon_skbio_scaled", "simpson_skbio_scaled"]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_ana_scaled['shannon_skbio_scaled'] = scaler.fit_transform(
    np.log(df_ana_scaled[["shannon_skbio"]])
)
df_ana_scaled['simpson_skbio_scaled'] = scaler.fit_transform(
    np.log(df_ana_scaled[["simpson_skbio"]])
)


# In[11]:


Outcomes = ["shannon_skbio_scaled", "simpson_skbio_scaled", "PC1", "PC2"]
DP_Div_model1 = run_models("model1", control_vars1= None, control_vars2 = None, df_data=df_ana_scaled, predictors = Predictors, outcomes = Outcomes)
DP_Div_model2 = run_models("model2", control_vars1= model2_2, control_vars2 = model2, df_data=df_ana_scaled, predictors = Predictors, outcomes = Outcomes)
DP_Div_model3 = run_models("model3", control_vars1= model3_2, control_vars2 = model3, df_data=df_ana_scaled, predictors = Predictors, outcomes = Outcomes)
DP_Div_model4 = run_models("model4", control_vars1= model4_2, control_vars2 = model4, df_data=df_ana_scaled, predictors = Predictors, outcomes = Outcomes)


# In[12]:


from statsmodels.stats.multitest import multipletests

# Combine all the models
# Results_DP_Div = pd.concat([DP_Div_model1, DP_Div_model2, DP_Div_model3, DP_Div_model4])
# display(Results_DP_Div.shape)

DP_Div_model4['p_adj'] = multipletests(DP_Div_model4['P_value'], 
                                                 alpha = 0.05, method = "fdr_bh")[1]

DP_Div_model4_sig = DP_Div_model4.loc[DP_Div_model4['p_adj'] < 0.05]
# display(DP_Div_model4_sig)

# DP_Div_model2['p_adj'] = multipletests(DP_Div_model2['P_value'], 
#                                                  alpha = 0.05, method = "fdr_bh")[1]
# DP_Div_model2_sig = DP_Div_model2.loc[DP_Div_model2['p_adj'] < 0.05]
# DP_Div_model2_sig
DP_Div_model4_sig


# In[13]:


# # Plot the results
# DP_Div_model4["Variable"] = DP_Div_model4["Variable"].map({
#     "AHEI_2010_score_eadj_scaled":"AHEI",
#     "hPDI_score_eadj_scaled" : "hPDI",
#     "rDII_score_eadj_scaled" : "rDII",
#     "AMED_score_eadj_scaled" : "AMED",
#     "rEDIH_score_all_eadj_scaled" : "rEDIH"
# })

# # replace function
# DP_Div_model4["Outcome"] = DP_Div_model4["Outcome"].replace({
#     "shannon_skbio_scaled":"Shannon_index",
#     "simpson_skbio_scaled" : "Simpson_index",
# })


# In[14]:


DP_Div_model4
# change the long format to the matrix format
DP_Div_model4_matrix = DP_Div_model4.pivot(
    index = "Variable",
    columns = "Outcome",
    values = "Coefficient"
)

DP_Div_model4_matrix


# In[15]:


# plot the results
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (4, 6))
sns.heatmap(
    DP_Div_model4_matrix,
    center = 0,
    annot = True,
    cmap = "coolwarm_r",
    # yticklabels = DP_Div_model4["Variable"],
    # xticklabels = DP_Div_model4["Outcome"],
    cbar_kws = {
        "orientation" : "horizontal",
        "pad" : 0.05,
        "location" : "top",
        "label": "Coefficient"
    },
    ax = ax
)


# # Exclude those with self reported MASLD 
# - Updated 2025-05-28

# In[16]:


df_MAFLD = pd.read_csv("../05_Medication_status/Medication_baseline.csv")

merge_df_MAFLD = pd.merge(df_ana_scaled, 
                          df_MAFLD[["participant_id", "NAFLD_baseline_diagnosis"]],
                          on = "participant_id", how = "left")

merge_df_MAFLD["MAFLD_baseline_diagnosis"] = merge_df_MAFLD["NAFLD_baseline_diagnosis"].map({"Yes":1, "No": 0})
merge_df_MAFLD["MAFLD_baseline_diagnosis"].value_counts(dropna = False) # 352 with self-reported NAFLD (base on ATC code)


# In[17]:


merge_df_MAFLD_filter = merge_df_MAFLD.loc[~(merge_df_MAFLD["MAFLD_baseline_diagnosis"] == 1)]
merge_df_MAFLD_filter.shape


# In[18]:


df_ana_scaled = merge_df_MAFLD_filter
df_ana_scaled


# In[19]:


df_ana_scaled.columns = df_ana_scaled.columns.str.replace('|', '_')


# In[20]:


df_ana_scaled["f__Methanobacteriaceae_g__Methanobrevibacter_s__Methanobrevibacter_smithii"].describe()


# In[21]:


df_ana_scaled.columns[df_ana_scaled.columns.str.contains(r"g__")]
df_ana_scaled.columns.get_loc("f__Akkermansiaceae_g__Akkermansia_s__Akkermansia_muciniphila")


# In[22]:


cols_check = ["AHEI_2010_score_eadj_scaled", "hPDI_score_eadj_scaled", "rDII_score_eadj_scaled", 
              "AMED_score_eadj_scaled", "rEDIH_score_all_eadj_scaled"]
df_ana_scaled2 = df_ana_scaled.dropna(subset = cols_check)
df_ana_scaled2.shape # 7565 obs


# In[23]:


Outcomes = ["shannon_skbio_scaled", "simpson_skbio_scaled", "PC1", "PC2"]
DP_Div_model1 = run_models("model1", control_vars1= None, control_vars2 = None, df_data=df_ana_scaled, predictors = Predictors, outcomes = Outcomes)
DP_Div_model2 = run_models("model2", control_vars1= model2_2, control_vars2 = model2, df_data=df_ana_scaled, predictors = Predictors, outcomes = Outcomes)
DP_Div_model3 = run_models("model3", control_vars1= model3_2, control_vars2 = model3, df_data=df_ana_scaled, predictors = Predictors, outcomes = Outcomes)
DP_Div_model4 = run_models("model4", control_vars1= model4_2, control_vars2 = model4, df_data=df_ana_scaled, predictors = Predictors, outcomes = Outcomes)


# In[24]:


DP_Div_model4['p_adj'] = multipletests(DP_Div_model4['P_value'], 
                                                 alpha = 0.05, method = "fdr_bh")[1]


# In[25]:


DP_Div_model4


# ## Categorize the Different DPs

# In[26]:


# Calculat the different categories for Dietary pattern score
cols_process = ["AHEI_2010_score_eadj", 
                "hPDI_score_eadj", 
                "rDII_score_eadj", 
                "AMED_score_eadj", 
                "rEDIH_score_all_eadj"]

for col in cols_process:
    col_new = f'{col}_quintile'
    # generate the quintile
    df_ana_scaled2[col_new] = pd.qcut(df_ana_scaled2[col], 5, labels = ["Q1","Q2", "Q3", "Q4", "Q5"])


# In[27]:


# Nutrients = pd.read_csv("Food log.csv")
# Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
# Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

# Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
# Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
# Nutrients_2.info()

# # merge with alcohol intake
# df_ana_scaled2 = pd.merge(df_ana_scaled2, Nutrients_2, on = "participant_id", how = 'left')
# df_ana_scaled2


# In[28]:


# run multiple LR
Predictors = ["AHEI_2010_score_eadj_scaled", "hPDI_score_eadj_scaled", 
            "rDII_score_eadj_scaled", 
            "AMED_score_eadj_scaled", "rEDIH_score_all_eadj_scaled"]

Predictors_2 = ["AHEI_2010_score_eadj_quintile", "hPDI_score_eadj_quintile", "rDII_score_eadj_quintile",
               "AMED_score_eadj_quintile", "rEDIH_score_all_eadj_quintile"]

Outcomes = df_ana_scaled2.columns[df_ana_scaled2.columns.str.contains('g__')].tolist()
len(Outcomes)


# In[29]:


df_ana_scaled2.columns[df_ana_scaled2.columns.str.contains(r'use')]


# In[30]:


print(df_ana_scaled2["Hormone_use"].value_counts())

df_ana_scaled2.dropna(subset = ['smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"]).shape


# In[31]:


df_ana_scaled2[['smoking_status', 'edu_status','Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']]


# In[32]:


df_ana_scaled2.smoking_status.value_counts(dropna= False)


# In[33]:


smoking_levels = ['Never', 'Current', 'Former']  
df_ana_scaled2['smoking_status'] = pd.Categorical(df_ana_scaled2['smoking_status'], categories = smoking_levels, ordered=True)


# In[34]:


edu_levels = ['Low', 'High'] 
df_ana_scaled2['edu_status'] = pd.Categorical(df_ana_scaled2['edu_status'], categories = edu_levels, ordered=True)

# Vitamine_use
Vit_use_level = ["Use", "not use"]
df_ana_scaled2['Vit_use'] = pd.Categorical(df_ana_scaled2['Vit_use'], categories = Vit_use_level, ordered=True)

# Hormone use
Hormone_use_level = ["Use", "not use"]
df_ana_scaled2['Hormone_use'] = pd.Categorical(df_ana_scaled2['Hormone_use'], categories = Hormone_use_level, ordered=True)

# PPI use
PPI_use_level = ["Use", "not use"]
df_ana_scaled2['PPI_use'] = pd.Categorical(df_ana_scaled2['PPI_use'], categories = PPI_use_level, ordered=True)

# Antibio_use
Antibio_use_level = ["Use", "not use"]
df_ana_scaled2['Antibio_use'] = pd.Categorical(df_ana_scaled2['Antibio_use'], categories = Antibio_use_level, ordered=True)


# In[35]:


model2 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']

# for hPDI index adjusted for alcohol intake 
model2_2 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use',"alcohol_g"]

model3 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "bmi"]

model3_2 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "bmi", 'alcohol_g']

# we don't adjust for the family history fo disease here
model4 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'Hormone_use', 'NSAID_use', 'MET_hour', 'sleep_hours_daily', 
          'PPI_use', 'Antibio_use']

model4_2 = ['sex_all', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'Hormone_use', 'NSAID_use', 'MET_hour', 'sleep_hours_daily', 
          'PPI_use', 'Antibio_use', "alcohol_g"]


# In[36]:


df_ana_scaled2.columns


# In[37]:


# run_regression_formula(df = df_ana_scaled, predictors = ["AHEI_2010_score_eadj_scaled"], 
#                                             outcomes = "g__Bifidobacterium_s__Bifidobacterium_bifidum", 
#                                             control_vars = model2)


# In[38]:


# def run_models(model_name, control_vars1, control_vars2, df_data, predictors, outcomes):
#     """Run regression tests for a given model"""
    
#     results_df = pd.DataFrame()
    
#     for predict_var in predictors:
#         print(predict_var)
#         for out in outcomes:
#             if predict_var == "hPDI_score_eadj_scaled":
#                 lm_fit = run_regression_formula(df=df_data, 
#                                               predictors=[predict_var], 
#                                               outcomes=out,
#                                               control_vars=control_vars1)
#             else:
#                 lm_fit = run_regression_formula(df=df_data,
#                                               predictors=[predict_var],
#                                               outcomes=out,
#                                               control_vars=control_vars2)
            
#             lm_fit = pd.DataFrame(lm_fit)
#             results_df = pd.concat([results_df, lm_fit], axis=0, ignore_index=True)
    
#     return results_df
Results_continue_model1 = run_models("model1", control_vars1= None, control_vars2 = None, df_data=df_ana_scaled2, predictors = Predictors, outcomes = Outcomes)
Results_continue_model2 = run_models("model2", control_vars1= model2_2, control_vars2 = model2, df_data=df_ana_scaled2, predictors = Predictors, outcomes = Outcomes)
Results_continue_model3 = run_models("model3", control_vars1= model3_2, control_vars2 = model3, df_data=df_ana_scaled2, predictors = Predictors, outcomes = Outcomes)
Results_continue_model4 = run_models("model4", control_vars1= model4_2, control_vars2 = model4, df_data=df_ana_scaled2, predictors = Predictors, outcomes = Outcomes)


# In[39]:


from statsmodels.stats.multitest import multipletests
Results_continue_model2
Results_continue_model2['p_adj'] = multipletests(Results_continue_model2['P_value'], 
                                                 alpha = 0.05, method = "fdr_bh")[1]
Results_continue_model2_sig = Results_continue_model2.loc[Results_continue_model2['p_adj'] < 0.05]


# In[40]:


Results_continue_model2_sig
Results_continue_model2_sig.loc[Results_continue_model2_sig["Outcome"].str.contains(r"plautii")]


# In[41]:


# Calculate the p_adjust for each model
from scipy import stats
def fdr_correction_ (df, p_col = "P_value", fdr_threshold = 0.05):
    df_copy = df.copy()
    df_copy["p_adj"] = stats.false_discovery_control(df_copy[p_col])
    df_sig = df_copy.loc[df_copy['p_adj'] < fdr_threshold]
    return df_copy, df_sig

df_models = [Results_continue_model1, Results_continue_model2, Results_continue_model3, Results_continue_model4]
corrected_dfs = []
significant_dfs = []

for i, df in enumerate(df_models):
    df_corrected, df_sig = fdr_correction_(df)
    corrected_dfs.append(df_corrected)
    significant_dfs.append(df_sig)

# Access to results
Results_continue_model1_corrected, Results_continue_model2_corrected, Results_continue_model3_corrected, Results_continue_model4_corrected = corrected_dfs
Results_continue_model1_sig, Results_continue_model2_sig, Results_continue_model3_sig, Results_continue_model4_sig = significant_dfs


# In[42]:


Results_continue_model1_corrected["Model"] = "Model1"
Results_continue_model2_corrected["Model"] = "Model2"
Results_continue_model3_corrected["Model"] = "Model3"
Results_continue_model4_corrected["Model"] = "Model4"

# Combine all the Results and export 
merged_DPs_to_Species = pd.concat([Results_continue_model1_corrected, Results_continue_model2_corrected,
                          Results_continue_model3_corrected, Results_continue_model4_corrected]).sort_values(["Model", "Coefficient"])
merged_DPs_to_Species.shape


# In[43]:


merged_DPs_to_Species.to_csv("./Results/DPs_to_Species_cont_1207.csv", index = False)


# # Categorical Diet pattern score

# In[44]:


# Also check the categorical DPs
def run_GLM_Group(df, predictors, outcomes, control_vars = None):
    
    results_list = pd.DataFrame()

    # check predictors if is a list
    if isinstance(predictors, str):
        predictors = [predictors]

    for var in predictors:
        # generate the formula string for the current predictor
        if control_vars is not None:
            formula = f'{outcomes} ~ {var} + {" + ".join(control_vars)}'
        else: 
            formula = f"{outcomes} ~ {var}"

        # fit the model using the formula
        model = smf.ols(formula, data = df, missing='drop').fit()
        
        N = model.nobs
        DPs_run = var.split("_")[0]
        coeffs = model.params
        
        quintile_coeffs = coeffs[coeffs.index.str.contains(var)]
        p_vals = model.pvalues[model.pvalues.index.str.contains(var)]
        conf_int = model.conf_int().loc[model.conf_int().index.str.contains(var)]

        # create a new dataframe
        df_coefficients = pd.DataFrame({
                    "N": N,
                    "DPs": DPs_run,
                    "Categories": quintile_coeffs.index,
                    "Outcome": outcomes,
                    "Coefficient" : quintile_coeffs.values,
                    "P_val" : p_vals.values,
                    "ci_lower" : conf_int.iloc[:, 0].values,
                    "ci_upper" : conf_int.iloc[:, 1].values
                    })

        df_coefficients["Categories"] = ["Q2", "Q3", "Q4", "Q5"]

        # combine the reference group
        reference_row = pd.DataFrame({
                        "N": N,
                        "DPs" : DPs_run,
                        "Categories" : ["Q1"],
                        "Outcome": outcomes,
                        "Coefficient" : [0.0],
                        "P_val" : np.nan,
                        "ci_lower" : np.nan,
                        "ci_upper" : np.nan
                        })
        
        # concat two dataframe
        df_results = pd.concat([reference_row, df_coefficients], ignore_index=True)
        # results_list.append(df_results)
        results_list = pd.concat([results_list, df_results], ignore_index=True)
    return results_list
    # change it into dataframe
    # results_df = pd.DataFrame(results_list)
    # set two decimals
    # results_df['CI'] = results_df.apply(lambda row: f"({row['ci_lower']:.2f}, {row['ci_upper']:.2f})", axis=1)
    # return results_df

import numpy as np


# In[45]:


# Outcomes
import statsmodels.formula.api as smf


# In[46]:


results_cat_model1 = pd.DataFrame()

for guts in Outcomes:
    lm_fit = run_GLM_Group(df = df_ana_scaled2, 
                          predictors = Predictors_2, 
                          outcomes = guts, 
                          control_vars = None)
    
    results_cat_model1 = pd.concat([results_cat_model1, lm_fit], ignore_index=True)
    
results_cat_model1

from statsmodels.stats.multitest import multipletests

p_clean = [p for p in results_cat_model1["P_val"] if not np.isnan(p)]

multipletests(
    p_clean,
    alpha=0.05, 
    method='fdr_bh'
)[1]

results_cat_model1_clean = results_cat_model1.loc[~results_cat_model1["P_val"].isnull(),:]
results_cat_model1_clean.shape

results_cat_model1_clean["P_adj"] = multipletests(
        p_clean,
        alpha=0.05, 
        method='fdr_bh'
)[1]

results_cat_model1_clean
results_cat_model1_clean[results_cat_model1_clean["P_adj"] < 0.05]


# In[47]:


results_cat_model2 = pd.DataFrame()
for guts in Outcomes:
    # print(metabolites)
    for predict_var in Predictors_2:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_GLM_Group(df = df_ana_scaled2, predictors = predict_var, outcomes = guts, control_vars = model2_2)
        else:
            lm_fit = run_GLM_Group(df = df_ana_scaled2, predictors = predict_var, outcomes = guts, control_vars = model2)
        results_cat_model2 = pd.concat([results_cat_model2, lm_fit], ignore_index=True)
    
results_cat_model3 = pd.DataFrame()
for guts in Outcomes:
    # print(metabolites)
    for predict_var in Predictors_2:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_GLM_Group(df = df_ana_scaled2, predictors = predict_var, outcomes = guts, control_vars = model3_2)
        else:
            lm_fit = run_GLM_Group(df = df_ana_scaled2, predictors = predict_var, outcomes = guts, control_vars = model3)
        results_cat_model3 = pd.concat([results_cat_model3, lm_fit], ignore_index=True)

results_cat_model4 = pd.DataFrame()
for guts in Outcomes:
    # print(metabolites)
    for predict_var in Predictors_2:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_GLM_Group(df = df_ana_scaled2, predictors = predict_var, outcomes = guts, control_vars = model4_2)
        else:
            lm_fit = run_GLM_Group(df = df_ana_scaled2, predictors = predict_var, outcomes = guts, control_vars = model4)
        results_cat_model4 = pd.concat([results_cat_model4, lm_fit], ignore_index = True)

DP_cat_to_species = pd.concat([results_cat_model1, results_cat_model2, results_cat_model3, results_cat_model4])
display(DP_cat_to_species.shape)

DP_cat_to_species.to_csv("./Results/DP_cat_to_Species_1207.csv", index = False)


# # Identify the diet specific gut microbiome
# - Model2 as the main model

# In[48]:


Results_continue_model2_sig_AHEI = Results_continue_model2_sig[Results_continue_model2_sig["Variable"] == "AHEI_2010_score_eadj_scaled"]
print(Results_continue_model2_sig_AHEI.shape)

Results_continue_model2_sig_hPDI = Results_continue_model2_sig[Results_continue_model2_sig["Variable"] == "hPDI_score_eadj_scaled"]
print(Results_continue_model2_sig_hPDI.shape)

Results_continue_model2_sig_rDII = Results_continue_model2_sig[Results_continue_model2_sig["Variable"] == "rDII_score_eadj_scaled"]
print(Results_continue_model2_sig_rDII.shape)

Results_continue_model2_sig_AMED = Results_continue_model2_sig[Results_continue_model2_sig["Variable"] == "AMED_score_eadj_scaled"]
print(Results_continue_model2_sig_AMED.shape)

Results_continue_model2_sig_rEDIH = Results_continue_model2_sig[Results_continue_model2_sig["Variable"] == "rEDIH_score_all_eadj_scaled"]
Results_continue_model2_sig_rEDIH.shape


# In[49]:


AHEI_gut = set(Results_continue_model2_sig_AHEI["Outcome"].unique())
print(len(AHEI_gut))

hPDI_gut = set(Results_continue_model2_sig_hPDI["Outcome"].unique())
print(len(hPDI_gut))

rDII_gut = set(Results_continue_model2_sig_rDII["Outcome"].unique())
print(len(rDII_gut))

AMED_gut = set(Results_continue_model2_sig_AMED["Outcome"].unique())
print(len(AMED_gut))

rEDIH_gut = set(Results_continue_model2_sig_rEDIH["Outcome"].unique())
len(rEDIH_gut)

AHEI_only = AHEI_gut - hPDI_gut - rDII_gut - AMED_gut - rEDIH_gut
hPDI_only = hPDI_gut - AHEI_gut - rDII_gut - AMED_gut - rEDIH_gut
rDII_only = rDII_gut - AHEI_gut - hPDI_gut - AMED_gut - rEDIH_gut
AMED_only = AMED_gut - AHEI_gut - hPDI_gut - rDII_gut - rEDIH_gut # None
rEDIH_only = rEDIH_gut - AHEI_gut - hPDI_gut - rDII_gut - AMED_gut 

AHEI_only_list = sorted(list(AHEI_only))
hPDI_only_list = sorted(list(hPDI_only))
rDII_only_list = sorted(list(rDII_only))
AMED_only_list = sorted(list(AMED_only))
rEDIH_only_list = sorted(list(rEDIH_only))

Specific_guts = pd.DataFrame([
    {'Diet': 'AHEI', 'Specific_gut': guts} 
    for guts in AHEI_only_list
] + [
    {'Diet': 'hPDI', 'Specific_gut': guts} 
    for guts in hPDI_only_list
] + [
    {'Diet': 'rDII', 'Specific_gut': guts} 
    for guts in rDII_only_list
] + [
    {'Diet': 'rEDIH', 'Specific_gut': guts} 
    for guts in rEDIH_only_list
])

Specific_guts 


# In[50]:


freq_counts = Specific_guts["Diet"].value_counts()
Specific_guts['frequency'] = Specific_guts['Diet'].map(freq_counts)


# In[51]:


Specific_guts


# In[52]:


# Export the results
Specific_guts.to_csv("./Results/DPs_specific_gut_1207.csv", index = False)


# # Select the microbiota with the same effect direction in all the dietary pattern score

# In[53]:


test_tbl = Results_continue_model2_sig.groupby("Outcome").count()
test_tbl_filter = test_tbl.loc[test_tbl["N"] == 5]
keep_model2 = test_tbl_filter.index
print(len(keep_model2))

Results_df_model2_test_sig_filter = Results_continue_model2_sig[Results_continue_model2_sig["Outcome"].isin(keep_model2)]
Results_df_model2_test_sig_filter


# In[54]:


Results_df_model2_test_sig_filter.loc[Results_df_model2_test_sig_filter["Outcome"].str.contains(r"plautii")]


# In[55]:


Results_df_model2_test_sig_filter["Outcome"].nunique()


# In[56]:


Results_df_model2_test_sig_filter.head(10)


# In[57]:


Results_df_model2_test_sig_filter.loc[:,'Coefficient_sign'] = np.sign(Results_df_model2_test_sig_filter['Coefficient'])
Results_df_model2_test_sig_filter


# In[58]:


def same_direction(group):
    coeffs = group['Coefficient']
    # Check if all positive or all negative (excluding zeros)
    non_zero_coeffs = coeffs[coeffs != 0]
    if len(non_zero_coeffs) == 0:
        return False
    return (non_zero_coeffs > 0).all() or (non_zero_coeffs < 0).all()


# In[59]:


consistent_outcomes = Results_df_model2_test_sig_filter.groupby(['Outcome'])[["Coefficient"]].apply(same_direction)


# In[60]:


consistent_outcomes.dtypes
outcomes_to_keep = consistent_outcomes[consistent_outcomes].index
outcomes_to_keep # 138 species


# In[61]:


# Need to create the mask
mask = Results_df_model2_test_sig_filter.set_index(['Outcome']).index.isin(outcomes_to_keep)


# In[62]:


Results_df_model2_test_sig_filter2 = Results_df_model2_test_sig_filter[mask]
Results_df_model2_test_sig_filter2


# In[63]:


Results_df_model2_test_sig_filter2["Outcome"].nunique()


# In[64]:


# results_df_model2_test_sig_filter["Outcome"].nunique()
# make sure all the coeffiicient direction are the same 
# import numpy as np
# group_outcome = Results_df_model2_test_sig_filter.groupby("Outcome")["Coefficient"].prod()

# # get the sign of the product
# result_sign = np.sign(group_outcome)
# result_sign


# In[65]:


# Results_df_model2_test_sig_filter['Coefficient_sign'] = Results_df_model2_test_sig_filter['Outcome'].map(result_sign)
# Results_df_model2_test_sig_filter


# In[66]:


# Results_df_model2_test_sig_filter_sig = Results_df_model2_test_sig_filter.loc[Results_df_model2_test_sig_filter["Coefficient_sign"] == -1]
# Results_df_model2_test_sig_filter_sig.head()


# In[67]:


print(Results_df_model2_test_sig_filter2.Outcome.nunique())
# Selected_micrbes = Results_df_model2_test_sig_filter2.Outcome.unique()


# In[68]:


# pd.Series(Selected_micrbes).str.contains(r"plautii").sum()


# ## Check for Model4
# - Update 2025-05-28, I did not adjust for the family history of T2DM and CVD, because they seems not the confounders here
#   Also I would like to substain the power.

# In[69]:


Results_continue_model4_sig.head()


# In[70]:


test_tbl = Results_continue_model4_sig.groupby("Outcome").count()
test_tbl_filter = test_tbl.loc[test_tbl["N"] == 5]
keep_model4 = test_tbl_filter.index
print(len(keep_model4))
Results_df_model4_test_sig_filter = Results_continue_model4_sig[Results_continue_model4_sig["Outcome"].isin(keep_model4)]
Results_df_model4_test_sig_filter

# group_outcome = Results_df_model4_test_sig_filter.groupby("Outcome")["Coefficient"].prod()

# # get the sign of the product
# result_sign = np.sign(group_outcome)
# result_sign

# Results_df_model4_test_sig_filter['Coefficient_sign'] = Results_df_model4_test_sig_filter['Outcome'].map(result_sign)
# Results_df_model4_test_sig_filter

# Results_df_model4_test_sig_filter_sig = Results_df_model4_test_sig_filter.loc[Results_df_model4_test_sig_filter["Coefficient_sign"] == -1]
# Results_df_model4_test_sig_filter_sig.head()

# Results_df_model4_test_sig_filter_sig.shape

# Results_df_model4_test_sig_filter_sig.Outcome.nunique()
# Selected_micrbes = Results_df_model4_test_sig_filter_sig.Outcome.unique()
# len(Selected_micrbes)

consistent_outcomes_model4 = Results_df_model4_test_sig_filter.groupby(['Variable','Outcome'])[["Coefficient"]].apply(same_direction)


# In[71]:


consistent_outcomes_model4
outcomes_to_keep_model4 = consistent_outcomes_model4[consistent_outcomes_model4].index
mask2 = Results_df_model4_test_sig_filter.set_index(['Variable', 'Outcome']).index.isin(outcomes_to_keep_model4)

Results_df_model4_test_sig_filter2 = Results_df_model4_test_sig_filter[mask2]
Results_df_model4_test_sig_filter2


# In[72]:


outcomes_to_keep_model4 = consistent_outcomes_model4[consistent_outcomes_model4].index
outcomes_to_keep_model4


# In[73]:


# # Selected_micrbes
# # Export the 71 microbes into local
# with open("Significant_Microbes_in_all_DPs_v2.txt", mode= "w") as file:
#     file.write(str(Selected_micrbes)) # change the list to str


# In[74]:


# Selected_micrbes
# # save it as pandas dataframe
# Selected_micrbes_Df = pd.DataFrame(Selected_micrbes, columns= ["Species"])
# Selected_micrbes_Df

# Selected_micrbes_Df.to_csv("Significant_Microbes_in_all_DPs.csv", index = False)


# In[75]:


# Results_continue_model4_sig.loc[Results_continue_model4_sig["Outcome"].str.contains(r'Roseburia_intestinalis')]


# # Cluster analysis for the Species abundance
# - Exploratory analysis
# - Using the clr transformed data

# In[76]:


data_clr = pd.read_csv(filepath_or_buffer= "./Data/Species_level_clr.csv")
data_clr
data_clr.columns = data_clr.columns.str.replace("|", "_")


# In[ ]:


data_clr_select = data_clr.iloc[:,data_clr.columns.isin(Selected_micrbes)]
data_clr_select.shape
data_clr_select.head()


# In[ ]:


# Calculate the corralation matrix and cluster analysis
pearson_matrix_cor = data_clr_select.corr(method = "pearson")


# In[ ]:


pearson_matrix_cor


# In[ ]:


from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (12, 20))
sns.heatmap(pearson_matrix_cor,
           annot = False, cmap = "coolwarm", fmt = ".1f", linewidths=.5)
plt.show()


# In[ ]:


# CLUSTER ANALYSIS
distance_matrix = 1 - pearson_matrix_cor.abs()
condensed_distance_matrix = squareform(distance_matrix, checks=False)
linked_clusters = linkage(condensed_distance_matrix, method="ward")


# In[ ]:


plt.figure(figsize = (15, 8))
dendrogram(linked_clusters,
           orientation="top",
           labels= data_clr_select.columns.tolist(),
           distance_sort = "descending",
           show_leaf_counts= True)
plt.tight_layout()
plt.show()


# In[ ]:


# Reorder the heatmap based on the clustering information
fig, ax = plt.subplots(figsize = (12, 10))
dendro_info = dendrogram(linked_clusters, ax = ax, orientation= "left", labels= data_clr_select.columns.tolist(),
                        no_plot = False)

reordered_indicies = dendro_info["leaves"]
reordered_column_names = [data_clr_select.columns[i] for i in reordered_indicies]

reordered_matrix_cor = pearson_matrix_cor.loc[reordered_column_names,reordered_column_names]
reordered_matrix_cor

plt.figure(figsize=(12,10))
sns.heatmap(reordered_matrix_cor,
            annot=False,
            cmap = "coolwarm",
            center=0,
            vmin = -1,
            vmax = 1, 
            fmt = ".1f",
            linewidth = 0.5)
plt.show()


# In[ ]:


Results_continue_model2_corrected.loc[Results_continue_model2_corrected["Outcome"].str.contains(r'Dysosmobacter_welbionis'),:]
Results_continue_model2_corrected.loc[Results_continue_model2_corrected["Outcome"].str.contains(r'Haemophilus_parainfluenzae'),:]
Results_continue_model2_corrected.loc[Results_continue_model2_corrected["Outcome"].str.contains(r'smithii'),:]
Results_continue_model2_corrected.loc[Results_continue_model2_corrected["Outcome"].str.contains(r'torques'),:]
Results_continue_model2_corrected.loc[Results_continue_model2_corrected["Outcome"].str.contains(r'gnavus'),:]


# In[ ]:






# ============================================================
Source: 03_Microbiome_to_Liver_revised.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ### Microbiome to Liver measures 
# 
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
    
import pandas as pd
import importlib # Import the importlib library
import A0_helpfunction # Import the module itself

# --- Make changes to A0_helpfunction.py and save them ---

importlib.reload(A0_helpfunction) # Reload the module
from A0_helpfunction import *
from sklearn.preprocessing import StandardScaler


# In[2]:


# read the microbiome species data
# data_species = pd.read_csv("Gut_species_abundance.csv")
# print(data_species.shape)

df_analysis = pd.read_csv("./Data/Data_analysis_without_filtering_missingDPs_Species_20251023_scaled.csv") 
df_analysis


# In[3]:


# df_analysis["rDII_score_eadj"].describe()


# In[4]:


data_liver_all = pd.read_csv("../03_Liver_Ultrasound_Related/Data/liver_ultrasound_aggre_base3_dps_scaled_life.csv")
data_liver_all


# In[5]:


cols_select = data_liver_all.columns[data_liver_all.columns.str.contains(r'participant_id|elasticity|velocity|viscosity|dispersion|attenuation|speed')]
data_liver_select = data_liver_all[cols_select]
data_liver_select


# In[6]:


df_analysis_liver = pd.merge(df_analysis, data_liver_select, on = "participant_id", how = "left")
df_analysis_liver


# # Exclude those with self reported MASLD 
# - update on 2025-05-28

# In[7]:


df_MAFLD = pd.read_csv("../05_Medication_status/Medication_baseline.csv")

df_analysis_liver = pd.merge(df_analysis_liver, 
                          df_MAFLD[["participant_id", "NAFLD_baseline_diagnosis"]],
                          on = "participant_id", how = "left")

df_analysis_liver["MAFLD_baseline_diagnosis"] = df_analysis_liver["NAFLD_baseline_diagnosis"].map({"Yes":1, "No": 0})
df_analysis_liver["MAFLD_baseline_diagnosis"].value_counts(dropna = False) # 405 obs with baseline self-reported MASLD


# In[8]:


df_analysis_liver_filter = df_analysis_liver.loc[~(df_analysis_liver["MAFLD_baseline_diagnosis"] == 1)]
df_analysis_liver_filter.shape


# In[9]:


sum(df_analysis_liver_filter.speed_of_sound_qbox.isnull())


# In[10]:


# df_analysis_liver_filter.to_csv("./Data/data_for_mediation_exclude_reportMASLD.csv", index = False)


# In[11]:


df_analysis_liver_filter["speed_of_sound_qbox"].describe()


# In[12]:


# df_analysis_liver_filter[df_analysis_liver_filter["speed_of_sound_qbox"]>1600]
# percentile_99 = df_analysis_liver_filter["speed_of_sound_qbox"].quantile(0.99)
# percentile_99


# In[13]:


# Clip the values for liver measure 
def clip_outliers(df, columns, lower_pct = 0.01, upper_pct = 0.99, inplace = False):
    """
    inplace: bool, modify original columns or create new ones
    """
    df_result = df.copy() if not inplace else df

    for col in columns:
        lower = df[col].quantile(lower_pct)
        upper = df[col].quantile(upper_pct)

        if inplace:
            df_result[col] = df_result[col].clip(lower, upper)
        else:
            df_result[f"{col}_clipped"] = df_result[col].clip(lower, upper)

    return df_result

# perform also the rank-based reverse tranformation
# https://github.com/edm1/rank-based-INT/blob/master/rank_based_inverse_normal_transformation.py
import numpy as np
import pandas as pd
import scipy.stats as ss

def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return ss.norm.ppf(x)

def rank_INT(series, c=3.0/8, stochastic=True):
    """ Perform rank-based inverse normal transformation on pandas series.
        If stochastic is True ties are given rank randomly, otherwise ties will
        share the same value. NaN values are ignored.

        Args:
            param1 (pandas.Series):   Series of values to transform
            param2 (Optional[float]):  Constand parameter (Bloms constant)
            param3 (Optional[bool]):  Whether to randomise rank of ties
        
        Returns:
            pandas.Series
    """

    # Check input
    assert(isinstance(series, pd.Series))
    assert(isinstance(c, float))
    assert(isinstance(stochastic, bool))

    # Set seed
    np.random.seed(123)

    # Take original series indexes
    orig_idx = series.index

    # Drop NaNs
    series = series.loc[~ pd.isnull(series)]

    # Get ranks
    if stochastic == True:
        # Shuffle by index
        series = series.loc[np.random.permutation(series.index)]
        # Get rank, ties are determined by their position in the series (hence
        # why we randomised the series)
        rank = ss.rankdata(series, method="ordinal")
    else:
        # Get rank, ties are averaged
        rank = ss.rankdata(series, method="average")

    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)

    # Convert rank to normal distribution
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
    
    return transformed[orig_idx]


# In[14]:


# Exclude those with speed of sound measures more than 99% percentile
# df_analysis_liver_filter = df_analysis_liver_filter[df_analysis_liver_filter["speed_of_sound_qbox"] <= percentile_99]
# df_analysis_liver_filter

# clip_columns
outcomes_liver_clip = ["elasticity_median_median_of_qboxes", 
                  "velocity_median_median_of_qboxes", 
                  "viscosity_median_median_of_qboxes", 
                  "dispersion_median_median_of_qboxes", 
                  "attenuation_coefficient_qbox", 
                  "speed_of_sound_qbox"]

df_analysis_liver_clipped = clip_outliers(df_analysis_liver_filter, outcomes_liver_clip, lower_pct=0.01, upper_pct=0.99)
df_analysis_liver_clipped


# In[15]:


# Apply Inverse 
# Create the clip columns
outcomes_liver_clip_cols = [col + "_clipped" for col in outcomes_liver_clip]
transformed_cols = [f"{col}_rank_INT" for col in outcomes_liver_clip_cols]
transformed_cols
outcomes_liver_clip_cols

df_analysis_liver_clipped[outcomes_liver_clip_cols]


# In[16]:


df_analysis_liver_clipped[outcomes_liver_clip_cols].describe()

# Drop the missingness
df_analysis_liver_clipped_whole = df_analysis_liver_clipped.dropna(subset=outcomes_liver_clip_cols).copy()
df_analysis_liver_clipped_whole


# In[17]:


# df_analysis_liver_clipped_whole
df_analysis_liver_clipped_whole[transformed_cols] = df_analysis_liver_clipped_whole[outcomes_liver_clip_cols].transform(rank_INT)


# In[23]:


# print(df_analysis_liver_clipped_whole.columns.tolist())


# # Perform New Analysis

# In[18]:


import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


# In[19]:


import matplotlib.pyplot as plt

# Create a figure with two subplots side by side
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df_analysis_liver_filter, x = "viscosity_mean_mean_of_qboxes", color="red")
sns.histplot(df_analysis_liver_filter, x = "viscosity_median_median_of_qboxes")
plt.xlabel("Viscosity_Measures_qboxes")
# Display the figure with both histograms
plt.show()


# ## Merge with alcohol intake

# In[20]:


df_analysis_liver_filter = df_analysis_liver_clipped_whole.copy()


# In[22]:


Nutrients = pd.read_csv("Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
Nutrients_2.info()

# merge with alcohol intake
df_analysis_liver_complete_2 = pd.merge(df_analysis_liver_filter, Nutrients_2, on = "participant_id", how = 'left')
df_analysis_liver_complete_2


# # Performing the analysis of GLM based on the formula

# In[23]:


df_analysis_liver_complete_2.columns


# In[22]:


# Write the DPs associated Microbes in
# with open("Significant_Microbes_in_all_DPs.txt", "r") as file:
#     sig_microbes = file.read()
# sig_microbes


# In[24]:


# change the column names first
df_analysis_liver_complete_2.columns = df_analysis_liver_complete_2.columns.str.replace('|', '_', regex=False)
microbes = df_analysis_liver_complete_2.columns[df_analysis_liver_complete_2.columns.str.contains(r'g__')]
microbes


# In[25]:


df_models = pd.read_csv("./Results/DPs_to_Species_cont_1207.csv")
df_model2 = df_models[(df_models["Model"] == "Model2") & (df_models["p_adj"] < 0.05)]
df_model2['Variable'].value_counts()

test_tbl = df_model2.groupby("Outcome").count()
test_tbl_filter = test_tbl.loc[test_tbl["N"] == 5]
keep_model2 = test_tbl_filter.index
print(len(keep_model2))

df_model2 = df_model2[df_model2["Outcome"].isin(keep_model2)]
df_model2

def same_direction(group):
    coeffs = group['Coefficient']
    # Check if all positive or all negative (excluding zeros)
    non_zero_coeffs = coeffs[coeffs != 0]
    if len(non_zero_coeffs) == 0:
        return False
    return (non_zero_coeffs > 0).all() or (non_zero_coeffs < 0).all()

consistent_outcomes = df_model2.groupby(['Outcome'])[["Coefficient"]].apply(same_direction)  

sum(consistent_outcomes)
outcomes_to_keep = consistent_outcomes[consistent_outcomes].index
outcomes_to_keep

# mask
mask = df_model2.set_index(['Outcome']).index.isin(outcomes_to_keep)

df_model2_filter = df_model2[mask]
df_model2_filter["Outcome"].nunique()


# In[26]:


microbes = df_model2_filter["Outcome"].unique()


# In[27]:


# # What if only consider the 138 species into the model
# Species_138 = pd.read_csv("./Results/Species_Prevalent_MASLD.csv")
# microbes = Species_138["Variable"]
# microbes


# In[28]:


# Main model used here 
model1 = ['sex_all', 'age', "bmi",'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', "alcohol_g",
         'Hormone_use', 'PPI_use', 'Antibio_use']

outcomes_liver = ["elasticity_median_median_of_qboxes_rank_INT", 
                  "velocity_median_median_of_qboxes_rank_INT", 
                  "viscosity_median_median_of_qboxes_rank_INT", 
                  "dispersion_median_median_of_qboxes_rank_INT", 
                  "attenuation_coefficient_qbox_rank_INT", 
                  "speed_of_sound_qbox_rank_INT"]

# outcomes_liver = ['elasticity_median_median_of_qboxes_clipped_rank_INT',
#        'velocity_median_median_of_qboxes_clipped_rank_INT',
#        'viscosity_median_median_of_qboxes_clipped_rank_INT',
#        'dispersion_median_median_of_qboxes_clipped_rank_INT',
#        'attenuation_coefficient_qbox_clipped_rank_INT',
#        'speed_of_sound_qbox_clipped_rank_INT']

results_df_model = pd.DataFrame()

# Save model results
for predict_var in microbes:
    # print(predict_var)
    for out in outcomes_liver:
        # print(out)
        lm_fit = run_regression_formula(df = df_analysis_liver_complete_2, 
                                              predictors = [predict_var], 
                                              outcomes = out, 
                                              control_vars = model1)
        lm_fit = pd.DataFrame(lm_fit)  
        results_df_model = pd.concat([results_df_model, lm_fit], axis = 0, ignore_index=True)


# In[29]:


results_df_model


# ## Adjusted for the Multiple test

# In[30]:


results_df_model["Outcome"].value_counts()


# In[31]:


results_df_model.sort_values(by = ["Outcome", "Coefficient"])


# In[32]:


from statsmodels.stats.multitest import multipletests

results_df_model['p_adj'] = multipletests(results_df_model['P_value'], alpha = 0.05, method = "fdr_bh")[1]
results_df_model_sig = results_df_model.loc[results_df_model['p_adj'] < 0.05]
results_df_model_sig


# In[33]:


results_df_model_sig["Outcome"].value_counts()


# In[34]:


# results_df_model.to_csv("./Results/Microbiome_to_Liver_metrics.csv", index=False)


# In[35]:


result_speed = results_df_model.loc[results_df_model["Outcome"] == "speed_of_sound_qbox_clipped_rank_INT"].sort_values(by = ["Outcome", "Coefficient"])


# In[36]:


# with pd.option_context('display.max_rows', None):
#     display(result_speed)


# In[37]:


# results_df_model.to_csv("./Results/Gut_to_Liver_update_20251205.csv", index = False)


# In[38]:


print(results_df_model_sig["Outcome"].value_counts())

crosstab = pd.crosstab(results_df_model_sig['Variable'], 
                       results_df_model_sig['Outcome'])
crosstab


# In[39]:


results_sig_Micro_speed = results_df_model_sig.loc[results_df_model_sig["Outcome"] == "speed_of_sound_qbox_rank_INT"].sort_values(["Coefficient"])
results_sig_Micro_speed


# In[40]:


display(results_sig_Micro_speed.loc[results_sig_Micro_speed["Variable"].str.contains(r'plautii')])
display(results_sig_Micro_speed.loc[results_sig_Micro_speed["Variable"].str.contains(r'prausnitzii')])
display(results_sig_Micro_speed.loc[results_sig_Micro_speed["Variable"].str.contains(r'welbionis')])
# display(results_sig_Micro_speed.loc[results_sig_Micro_speed["Variable"].str.contains(r'bifidum')])


# # What if only focuse on the diet associated gut species

# In[43]:


# species_diet = pd.read_csv("./Results/Archive/DPs_to_Species_cont.csv")
# species_diet_model2 = species_diet[(species_diet["Model"] == "Model2") & (species_diet["p_adj"] < 0.05)]
# species_diet_model2

# test_tbl = species_diet_model2.groupby("Outcome").count()
# test_tbl_filter = test_tbl.loc[test_tbl["N"] == 5]
# keep_model2 = test_tbl_filter.index
# print(len(keep_model2))

# species_diet_model2_filter = species_diet_model2[species_diet_model2["Outcome"].isin(keep_model2)]
# # species_diet_model2_filter


# In[44]:


# len(species_diet_model2_filter["Outcome"].unique())


# In[48]:


# results_df_model_2 = pd.DataFrame()

# # Save model results
# for predict_var in species_diet_model2_filter["Outcome"].unique():
#     # print(predict_var)
#     for out in outcomes_liver:
#         # print(out)
#         lm_fit = run_regression_formula(df = df_analysis_liver_complete_2, 
#                                               predictors = [predict_var], 
#                                               outcomes = out, 
#                                               control_vars = model1)
#         lm_fit = pd.DataFrame(lm_fit)  
#         results_df_model_2 = pd.concat([results_df_model_2, lm_fit], axis = 0, ignore_index=True)


# In[49]:


# # Multiple test
# from statsmodels.stats.multitest import multipletests

# results_df_model_2['p_adj'] = multipletests(results_df_model_2['P_value'], alpha = 0.05, method = "fdr_bh")[1]
# results_df_model_2_sig = results_df_model_2.loc[results_df_model_2['p_adj'] < 0.05]
# results_df_model_2_sig


# In[50]:


# len(results_df_model_2_sig["Variable"].unique()) # 24 with significant associations


# In[52]:


# results_sig_Micro_speed = results_df_model_2_sig.loc[results_df_model_2_sig["Outcome"] == "speed_of_sound_qbox_rank_INT"].sort_values(["Coefficient"])
# results_sig_Micro_speed


# In[53]:


# results_sig_Micro_speed.shape


# In[44]:


# results_sig_Micro_speed.to_csv("./Results/sig_microbes_speedofsound.csv", index=False)


# # How about those specific species

# In[41]:


species_diet_specific = pd.read_csv("./Results/DPs_specific_gut_1207.csv")
species_diet_specific["Specific_gut"].nunique()


# In[42]:


results_df_model_2 = pd.DataFrame()

# Save model results
for predict_var in species_diet_specific["Specific_gut"].unique():
    # print(predict_var)
    for out in outcomes_liver:
        # print(out)
        lm_fit = run_regression_formula(df = df_analysis_liver_complete_2, 
                                              predictors = [predict_var], 
                                              outcomes = out, 
                                              control_vars = model1)
        lm_fit = pd.DataFrame(lm_fit)  
        results_df_model_2 = pd.concat([results_df_model_2, lm_fit], axis = 0, ignore_index=True)


# In[43]:


# Multiple test
from statsmodels.stats.multitest import multipletests

pd.set_option("display.max_colwid", None)

results_df_model_2['p_adj'] = multipletests(results_df_model_2['P_value'], alpha = 0.05, method = "fdr_bh")[1]
results_df_model_2_sig = results_df_model_2.loc[results_df_model_2['p_adj'] < 0.05]
results_df_model_2_sig


# In[44]:


# Plot the results
results_df_model_2


# In[45]:


# Heatmap
# recode the variables 
import re

def format_index(idx):
    # extract the genus
    genus_match = re.search(r'g__(.+?)(?:_s__)', idx)
    genus = genus_match.group(1) if genus_match else "Unknown"
    # extract the species
    species_match = re.search(r's__(.+)', idx)
    species = species_match.group(1) if species_match else "Unknown"
    return f"s_{species} (g_{genus})"
    
results_df_model_2 = results_df_model_2.copy()

results_df_model_2["Variable_new"] = results_df_model_2["Variable"].map(format_index)
results_df_model_2

recode_dict = {
    "speed_of_sound_qbox_rank_INT" : "Speed of Sound",
    "attenuation_coefficient_qbox_rank_INT" : "Attenuation coefficient",
    "dispersion_median_median_of_qboxes_rank_INT": "Liver Dispersion",
    "elasticity_median_median_of_qboxes_rank_INT": "Liver Elasticity",
    "velocity_median_median_of_qboxes_rank_INT": "Liver Velocity",
    "viscosity_median_median_of_qboxes_rank_INT": "Liver Viscosity"
}

results_df_model_2["Outcome_new"] = results_df_model_2["Outcome"].map(recode_dict)
results_df_model_2


# In[46]:


results_df_model_2_heatmap = results_df_model_2.pivot(index = "Variable_new", columns= "Outcome_new", values = "Coefficient")
results_df_model_2_heatmap


# In[59]:


# plt.figure(figsize=(120,60))
fig, ax = plt.subplots(figsize=(20,15))

# add the annotation and asterisks
p_adj_dic = {}
for _, row in results_df_model_2.iterrows():
    key = (row["Variable_new"], row["Outcome_new"])
    p_adj_dic[key] = row["p_adj"]
    
results_df_model_2_heatmap_matrix = results_df_model_2_heatmap.copy().astype(str)

for i, row_label in enumerate(results_df_model_2_heatmap_matrix.index):
    for j, col_label in enumerate(results_df_model_2_heatmap_matrix.columns):
        value = results_df_model_2_heatmap.iloc[i, j]

        # check if the combination is sig
        key = (row_label, col_label)
        is_sig = p_adj_dic.get(key, 1) < 0.05

        if is_sig:
            results_df_model_2_heatmap_matrix.iloc[i, j] = f"{value:.2f}*"
        else:
            results_df_model_2_heatmap_matrix.iloc[i, j] = f"{value:.2f}" 

hp_model1 = sns.heatmap(results_df_model_2_heatmap.T, 
            cmap='RdBu_r',  # Red-Blue diverging colormap
            center=0,       # Center the colormap at 0
            annot=results_df_model_2_heatmap_matrix.T,     # Show values in cells
            fmt='',      # Format for the annotations (2 decimal places)
            cbar_kws={'label': 'Coefficient',
                      "orientation": "horizontal",
                     'fraction': 0.05,
                     'shrink': 0.5,
                     'aspect': 40,
                     'pad': 0.01}, # pad change the distance between heatmap and legend
            square=True,
            linewidths = 1,
            linecolor='black',
#             xticklabels=False,
            ax = ax)    # Make cells square

# legend = plt.colorbar(hm.collections[0])
# legend.set_label('Coefficient')

# Rotate x-axis labels for better readability
plt.xlabel('') # remove the x-axis title
plt.ylabel('') # remove the x-axis title
plt.tick_params(axis='x', top=True, bottom=False,
                labeltop = True, labelbottom=False)
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)

plt.savefig("./Results/exclusive_species_HepaticMetrics.pdf", dpi = 500, bbox_inches = "tight")


# In[53]:


results_df_model_2.to_csv("./Results/specific_species_to_liver_metrics.csv", index = False)


# # Plot the prevalent MASLD

# In[61]:


final_results_spec = pd.read_csv("./Results/Specific_species_MASLD.csv")
final_results_spec


# In[62]:


import re

def format_index(idx):
    # extract the genus
    genus_match = re.search(r'g__(.+?)(?:_s__)', idx)
    genus = genus_match.group(1) if genus_match else "Unknown"
    # extract the species
    species_match = re.search(r's__(.+)', idx)
    species = species_match.group(1) if species_match else "Unknown"
    return f"s_{species} (g_{genus})"
final_results_spec = final_results_spec.copy()

final_results_spec["Variable_new"] = final_results_spec["Variable"].map(format_index)
final_results_spec


# In[67]:


# Forest plot p_adj
def assign_color (p_adj):
    # assign color based on OR
    if p_adj < 0.05:
        return "darkred"
    else:
        return "grey"

final_results_spec = final_results_spec.sort_values("OR", ascending= True)

colors = [assign_color(row["p_adj"])
         for _, row in final_results_spec.iterrows()]

fig, ax = plt.subplots(figsize = (12, 10))
# make room so the outside text is visible
plt.subplots_adjust(right = 0.8)

y_pos = range(len(final_results_spec))

# assign color
for i, (idx, row) in enumerate(final_results_spec.iterrows()):
    ax.scatter(row["OR"], i, s = 80, color = colors[i], edgecolors = "black", linewidth = 0.5, zorder = 3)
    ax.hlines(i, row["lower_OR_ci"], row["upper_OR_ci"], color =  colors[i], linewidth = 2, alpha = 0.7, zorder = 2)

    # generate the labels
    label = f'{row["OR"]:.2f} ({row["lower_OR_ci"]:.2f},{row["upper_OR_ci"]:.2f})'
    # ax.transAxes: using axis-fraction coordinates,
    # clip_on: control if the text is clipped to the axes rectangle
    ax.text(1.23, i, label, transform = ax.transData, ha = "left", va = "center", fontsize = 10, clip_on= False)
    
ax.axvline(1, color = "red", linestyle = "--", alpha = 0.5,  linewidth = 1.5, zorder = 1)

ax.set_yticks(y_pos)
ax.set_yticklabels(final_results_spec["Variable_new"], fontsize = 10)
ax.set_xlabel("Odds Ratio (95%CI) for Prevalent MASLD", fontsize = 10, fontweight = "bold")
ax.invert_yaxis()


# plt.tight_layout()
# plt.savefig("./Results/Forest_plot_of_species.png", dpi = 300, bbox_inches = "tight")
plt.savefig("./Results/Forest_plot_of_Specific_species_MASLD.pdf", bbox_inches = "tight")


# In[45]:


# import pandas as pd
# DP_microbioes = pd.read_csv(filepath_or_buffer= "dps_to_microbiomes_model2_sig.csv")
# display(DP_microbioes.head())

# grouped = DP_microbioes.groupby(DP_microbioes['Variable'])
# df_rEDIH = grouped.get_group("rEDIH_score_all_eadj_scaled")
# df_AHEI_2010 = grouped.get_group("AHEI_2010_score_eadj_scaled")
# df_AMED = grouped.get_group("AMED_score_eadj_scaled")
# df_hPDI = grouped.get_group("hPDI_score_eadj_scaled")
# df_rDII = grouped.get_group("rDII_score_eadj_scaled")

# # set intersection
# lists = [df_AHEI_2010['Outcome'],df_AMED['Outcome'], df_hPDI['Outcome'],df_rDII['Outcome'], df_rEDIH['Outcome']]
# overlapped = set.intersection(*map(set, lists))
# len(overlapped) # 145 


# In[46]:


# # microb_liver_sig["Variable"].intersection(DP_microbioes["Variable"])
# overlapped = set(microb_liver_sig["Variable"]) & set(DP_microbioes["Outcome"])
# microb_liver_sig.loc[microb_liver_sig['Variable'].isin(overlapped)].Outcome.value_counts()


# In[47]:


# microb_liver_sig.loc[microb_liver_sig['Variable'].isin(overlapped)]


# In[48]:


# select the rows
# microb_liver_sig.loc[microb_liver_sig['Variable'].isin(overlapped)].query('Outcome == "speed_of_sound_qbox_rank_INT"')


# In[49]:


# test = microb_liver_sig.loc[microb_liver_sig['Variable'].isin(overlapped)].query('Outcome == "speed_of_sound_qbox_rank_INT"')
# test[['genus', "species"]] = test['Variable'].str.split("|", expand = True)
# test


# In[50]:


# test.to_csv("Overlapped_microbes_between_DPs_Microbes_Liver.csv", index = False)


# # Plot the results

# In[51]:


results_df_model_sig


# In[54]:


results_df_model_sig["Outcome"].unique()


# In[56]:


# recode the variables 
import re

def format_index(idx):
    # extract the genus
    genus_match = re.search(r'g__(.+?)(?:_s__)', idx)
    genus = genus_match.group(1) if genus_match else "Unknown"
    # extract the species
    species_match = re.search(r's__(.+)', idx)
    species = species_match.group(1) if species_match else "Unknown"
    return f"s_{species} (g_{genus})"
results_df_model_sig = results_df_model_sig.copy()

results_df_model_sig["Variable_new"] = results_df_model_sig["Variable"].map(format_index)
results_df_model_sig

recode_dict = {
    "speed_of_sound_qbox_rank_INT" : "Speed of Sound",
    "attenuation_coefficient_qbox_rank_INT" : "Attenuation coefficient",
    "dispersion_median_median_of_qboxes_rank_INT": "Liver Dispersion",
    "elasticity_median_median_of_qboxes_rank_INT": "Liver Elasticity",
    "velocity_median_median_of_qboxes_rank_INT": "Liver Velocity"
}

results_df_model_sig["Outcome_new"] = results_df_model_sig["Outcome"].map(recode_dict)
results_df_model_sig


# In[58]:


results_df_model_sig.to_csv("./Results/Sig_Microbes_LiverMetrics_clean.csv", index = False)


# In[ ]:





# In[ ]:






# ============================================================
Source: 03_Species_MASLD.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# # Microbiome to prevalent MASLD
# 
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
    
import pandas as pd
import importlib # Import the importlib library
import A0_helpfunction # Import the module itself

# --- Make changes to A0_helpfunction.py and save them ---

importlib.reload(A0_helpfunction) # Reload the module
from A0_helpfunction import *
from sklearn.preprocessing import StandardScaler


# In[3]:


def multiple_glm_model (data, 
                        dependent_var, 
                        covariate_sets, 
                        family = sm.families.Binomial()):
    """
    data: dataframe
    dependent_var: outcome
    covariate_sets: dict- {"model": [list of the covariates]}
    family: model
    """
    models = {}
    # covariate_sets (is a dictionary)
    # Iterate through each model
    for model_name, covariates in covariate_sets.items():

        # Create the formula
        formula = f"{dependent_var} ~ {'+'.join(covariates)}"
        
        # print(f"Fitting {model_name} with formula: {formula}")

        # fit the model (remove missing)
        model = sm.GLM.from_formula(formula, family = family, data = data, missing = "drop")

        # results
        result = model.fit()

        # store the results
        # models[model_name] = result.summary()
        models[model_name] = result
        
    return models

def logist_model_df2(data, covariate_sets, dependent_var, exposure_only = True):
    
    '''
    run logistic regression model andd extract the results in dataframe
    '''
    all_results = []

    # Fit all models
    model_dict = multiple_glm_model(data = data, 
                                    dependent_var = dependent_var, 
                                    covariate_sets = covariate_sets)
    
    for model_name, model_results in model_dict.items():

        covariates = covariate_sets[model_name]

        # Get exposure
        exposure_var = covariates[-1]

        # Determine which variable to extract
        if exposure_only:
            vars_to_extract = [exposure_var]
        else:
            vars_to_extract = covariates

        # Get the result
        params = model_results.params
        pvalue = model_results.pvalues
        conf_int = model_results.conf_int(alpha = 0.05)

        for var in vars_to_extract:
            if var in params.index:
                result_row = pd.DataFrame(
                    {
                        "Model": [model_name],
                        "Variable": [var],
                        "Outcome": [dependent_var],
                        "Coef": [params[var]],
                        "pvalue": [pvalue[var]],
                        "lower_ci" : [conf_int.loc[var, 0]],
                        "upper_ci" : [conf_int.loc[var, 1]],
                        "OR" : [np.exp(params[var])],
                        "lower_OR_ci" : [np.exp(conf_int.loc[var, 0])],
                        "upper_OR_ci" : [np.exp(conf_int.loc[var, 1])],
                    }
                )
                all_results.append(result_row)
            
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    


# In[4]:


df_analysis = pd.read_csv("./Data/Data_analysis_without_filtering_missingDPs_Species_20251023_scaled.csv") 
df_analysis


# In[5]:


# df_analysis.columns.tolist()
# Merge with the prevalent MASLD at baseline
# Based on another definition of MAFLD(speed of sound) + ICD-11
df_MAFLD = pd.read_csv("../05_Medication_status/MAFLD_baseline.csv")
merge_df_MAFLD = pd.merge(df_analysis, df_MAFLD[["participant_id", "mafld_diagnosed"]],
                          on = "participant_id", how = "left")

merge_df_MAFLD["MAFLD_baseline_diagnosis"] = merge_df_MAFLD["mafld_diagnosed"].map({"Yes":1, "No": 0})
merge_df_MAFLD["MAFLD_baseline_diagnosis"].value_counts(dropna = False)


# In[6]:


# Reorder smokeing status
merge_df_MAFLD["smoking_status"] = pd.Categorical(
    merge_df_MAFLD["smoking_status"],
    categories =  ["Never", "Former", "Current"],
    ordered = True
)


# In[7]:


merge_df_MAFLD["age"].isnull().sum()


# In[8]:


merge_df_MAFLD.shape


# In[9]:


merge_df_MAFLD["age_all"].isnull().sum()


# In[10]:


# Age update in the V2
# lifestyles = pd.read_csv("../02_Lifestyle factors/lifestyle_factor_all_disease_V2.csv")
# lifestyles_age = lifestyles[["participant_id", "age_all"]]
# lifestyles_age


# In[11]:


# merge with merge_df_MAFLD
# merge_df_MAFLD2 = pd.merge(merge_df_MAFLD,
#                          lifestyles_age, on="participant_id", how = "left")
# merge_df_MAFLD2["age_all"].isnull().sum()


# In[12]:


# Combined with alcohol intake
Nutrients = pd.read_csv("/home/ec2-user/studies/ruifang/Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
Nutrients_2.shape

merge_df_MAFLD2 = pd.merge(merge_df_MAFLD, Nutrients_2, on = "participant_id", how = "left")


# In[13]:


species_diet = pd.read_csv("./Results/DPs_to_Species_cont_1207.csv")
species_diet_model2 = species_diet[(species_diet["Model"] == "Model2") & (species_diet["p_adj"] < 0.05)]
species_diet_model2


# In[14]:


test_tbl = species_diet_model2.groupby("Outcome").count()
test_tbl_filter = test_tbl.loc[test_tbl["N"] == 5]
keep_model2 = test_tbl_filter.index
print(len(keep_model2))

species_diet_model2_filter = species_diet_model2[species_diet_model2["Outcome"].isin(keep_model2)]
species_diet_model2_filter

def same_direction(group):
    coeffs = group['Coefficient']
    # Check if all positive or all negative (excluding zeros)
    non_zero_coeffs = coeffs[coeffs != 0]
    if len(non_zero_coeffs) == 0:
        return False
    return (non_zero_coeffs > 0).all() or (non_zero_coeffs < 0).all()

consistent_outcomes = species_diet_model2_filter.groupby(['Outcome'])[["Coefficient"]].apply(same_direction)

outcomes_to_keep = consistent_outcomes[consistent_outcomes].index
outcomes_to_keep # 138 species

# Need to create the mask
mask = species_diet_model2_filter.set_index(['Outcome']).index.isin(outcomes_to_keep)

species_diet_model2_filter = species_diet_model2_filter[mask]
species_diet_model2_filter.shape


# In[15]:


species_diet_model2_filter["Outcome"].nunique()


# In[16]:


merge_df_MAFLD2.columns[merge_df_MAFLD2.columns.str.contains(f"bmi")]


# In[17]:


covariates_list = ['age_all','smoking_status', "bmi",'edu_status','sleep_hours_daily','MET_hour',
                   'Vit_use', 'Hormone_use', "sex_x", 'Hormone_use', 'PPI_use', 'Antibio_use', "alcohol_g"]


# In[18]:


# Exposure that I would like to check
variables_to_test = species_diet_model2_filter["Outcome"].unique().tolist()
# variables_to_test


# In[19]:


merge_df_MAFLD2.columns = merge_df_MAFLD2.columns.str.replace('|', '_')


# In[20]:


# Loop through all exposure and run models
all_results = []

for i, exposure_var in enumerate(variables_to_test, 1):
    # print(f"Testing: {exposure_var}")
          
    full_variable = {f"model_{exposure_var}": covariates_list + [exposure_var]}

    # run the model
    results = logist_model_df2(
        data = merge_df_MAFLD2,
        covariate_sets= full_variable,
        dependent_var= "MAFLD_baseline_diagnosis",
        exposure_only= True,
    )

    # Add metadata
    results["test_variable"] = exposure_var
    results["number"] = i

    # Append to results
    all_results.append(results)


# In[21]:


if all_results:
    final_results = pd.concat(all_results, ignore_index=True)

    # Multiple correction
    from statsmodels.stats.multitest import multipletests

    reject, pvalues_correct, _, _ = multipletests(
        final_results["pvalue"],
        method= "fdr_bh"
    )

    final_results["p_adj"] = pvalues_correct


# In[22]:


final_results


# In[23]:


final_results.to_csv("./Results/Species_Prevalent_MASLD.csv", index = False)


# # Plot the results

# In[24]:


final_results = pd.read_csv("./Results/Species_Prevalent_MASLD.csv")


# In[25]:


final_results_sig = final_results[final_results["p_adj"] < 0.05]


# In[26]:


final_results[final_results["pvalue"] < 0.05].shape


# In[27]:


pd.set_option("display.max_colwidth", None)
final_results[(final_results["pvalue"] < 0.05) & (final_results["OR"] < 1)].sort_values("OR")


# In[28]:


display(final_results[(final_results["p_adj"] < 0.05) & (final_results["OR"] > 1)].sort_values("OR"))
final_results[(final_results["p_adj"] < 0.05) & (final_results["OR"] > 1)].sort_values("OR").shape


# In[29]:


final_results_sig.shape


# In[30]:


# rename the species
import re

def format_index(idx):
    # extract the genus
    genus_match = re.search(r'g__(.+?)(?:_s__)', idx)
    genus = genus_match.group(1) if genus_match else "Unknown"
    # extract the species
    species_match = re.search(r's__(.+)', idx)
    species = species_match.group(1) if species_match else "Unknown"
    return f"s_{species} (g_{genus})"
final_results_sig = final_results_sig.copy()

final_results_sig["Variable_new"] = final_results_sig["Variable"].map(format_index)
final_results_sig


# In[31]:


# function to define the color
def assign_color (or_val, ci_lower, ci_upper):
    # assign color based on OR
    if ci_upper < 1:
        return "#4393C3"
    elif ci_lower > 1:
        return "#D6604D"

top_sig = final_results_sig.sort_values("OR", ascending= True)

colors = [assign_color(row["OR"], row["lower_OR_ci"], row["upper_OR_ci"])
         for _, row in top_sig.iterrows()]


# In[32]:


# Forest plot for the top 20 significant
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize = (12, 10))
# make room so the outside text is visible
plt.subplots_adjust(right = 0.8)

y_pos = range(len(top_sig))

# assign color
for i, (idx, row) in enumerate(top_sig.iterrows()):
    ax.scatter(row["OR"], i, s = 80, color = colors[i], edgecolors = "black", linewidth = 0.5, zorder = 3)
    ax.hlines(i, row["lower_OR_ci"], row["upper_OR_ci"], color =  colors[i], linewidth = 2, alpha = 0.7, zorder = 2)

    # generate the labels
    label = f'{row["OR"]:.2f} ({row["lower_OR_ci"]:.2f},{row["upper_OR_ci"]:.2f})'
    # ax.transAxes: using axis-fraction coordinates,
    # clip_on: control if the text is clipped to the axes rectangle
    ax.text(1.35, i, label, transform = ax.transData, ha = "left", va = "center", fontsize = 10, clip_on= False)
    
ax.axvline(1, color = "red", linestyle = "--", alpha = 0.5,  linewidth = 1.5, zorder = 1)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_sig["Variable_new"], fontsize = 10)
ax.set_xlabel("Odds Ratio (95%CI) for Prevalent MASLD", fontsize = 10, fontweight = "bold")
ax.invert_yaxis()

# plt.tight_layout()
# plt.savefig("./Results/Forest_plot_of_species.png", dpi = 300, bbox_inches = "tight")
# plt.savefig("./Results/Forest_plot_of_species.pdf", bbox_inches = "tight")


# In[33]:


# Exposure the data into Data folder
merge_df_MAFLD2.to_csv("./Data/Species_Genus_MAFLD.csv", index = False)


# # Run the analysis for the genus data as well

# In[34]:


# combine with genus data 

# genus_df = pd.read_csv("./Data/data_genus_full_qc_clr_scale.csv")
# genus_df

# rename the column name
# genus_df.columns = genus_df.columns.str.replace('|', '_')


# In[35]:


# combine to merge_df_MAFLD2
# data_all = pd.merge(merge_df_MAFLD2, genus_df, on = "participant_id", how = "left")
# data_all


# In[36]:


# Extract diet-asspciated gut genus
# merged_DPs_to_Genus = pd.read_csv("./Results/Archive/DPs_to_Genus_cont.csv")
# merged_DPs_to_Genus

# merged_DPs_to_Genus_sig = merged_DPs_to_Genus[(merged_DPs_to_Genus["Model"] == "Model2") & (merged_DPs_to_Genus["p_adj"] < 0.05)]
# merged_DPs_to_Genus_sig.shape

# merged_DPs_to_Genus_sig["Variable"].value_counts()


# In[37]:


# test_tbl = merged_DPs_to_Genus_sig.groupby("Outcome").count()
# test_tbl_filter = test_tbl.loc[test_tbl["N"] == 5]
# keep_model2 = test_tbl_filter.index
# print(len(keep_model2))

# Genus_diet_model2_filter = merged_DPs_to_Genus_sig[merged_DPs_to_Genus_sig["Outcome"].isin(keep_model2)]
# Genus_diet_model2_filter


# In[38]:


# variables_to_test = Genus_diet_model2_filter["Outcome"].unique().tolist()

# # Loop through all exposure and run models
# all_results = []

# for i, exposure_var in enumerate(variables_to_test, 1):
#     print(f"Testing: {exposure_var}")
#     full_variable = {f"model_{exposure_var}": covariates_list + [exposure_var]}

#     # run the model
#     results = logist_model_df2(
#         data = data_all,
#         covariate_sets= full_variable,
#         dependent_var= "MAFLD_baseline_diagnosis",
#         exposure_only= True,
#     )

#     # Add metadata
#     results["test_variable"] = exposure_var
#     results["number"] = i

#     # Append to results
#     all_results.append(results)


# In[39]:


# all_results

# if all_results:
#     final_results = pd.concat(all_results, ignore_index=True)

#     # Multiple correction
#     from statsmodels.stats.multitest import multipletests

#     reject, pvalues_correct, _, _ = multipletests(
#         final_results["pvalue"],
#         method= "fdr_bh"
#     )

#     final_results["p_adj"] = pvalues_correct


# In[40]:


# final_results


# In[41]:


# final_results_sig = final_results[final_results["p_adj"] < 0.05]

# # function to define the color
# def assign_color (or_val, ci_lower, ci_upper):
#     # assign color based on OR
#     if ci_upper < 1:
#         return "#4393C3"
#         # return "darkgreen"
#     elif ci_lower > 1:
#         return "#D6604D"
#         # return "darkred"


# top_sig = final_results_sig.sort_values("OR", ascending= True)

# colors = [assign_color(row["OR"], row["lower_OR_ci"], row["upper_OR_ci"])
#          for _, row in top_sig.iterrows()]


# In[42]:


# top_sig.shape # 17 genus
# top_sig


# In[43]:


# def format_index2(idx):
#     # extract the family
#     family_match = re.search(r'f__(.+)(?:_g__)', idx)
#     family = family_match.group(1) if family_match else "Unknown"
#     # extract the genus
#     genus_match = re.search(r'g__(.+)', idx)
#     genus = genus_match.group(1) if genus_match else "Unknown"
#     return f"g_{genus} (f_{family})"
# top_sig = top_sig.copy()

# top_sig["Variable_new"] = top_sig["Variable"].map(format_index2)
# top_sig


# In[44]:


# # Forest plot 
# import matplotlib.pyplot as plt
# import seaborn as sns

# fig, ax = plt.subplots(figsize = (12, 10))
# # make room so the outside text is visible
# plt.subplots_adjust(right = 0.8)

# y_pos = range(len(top_sig))

# # assign color
# for i, (idx, row) in enumerate(top_sig.iterrows()):
#     ax.scatter(row["OR"], i, s = 80, color = colors[i], edgecolors = "black", linewidth = 0.5, zorder = 3)
#     ax.hlines(i, row["lower_OR_ci"], row["upper_OR_ci"], color =  colors[i], linewidth = 2, alpha = 0.7, zorder = 2)

#     # generate the labels
#     label = f'{row["OR"]:.2f} ({row["lower_OR_ci"]:.2f},{row["upper_OR_ci"]:.2f})'
#     # ax.transAxes: using axis-fraction coordinates,
#     # clip_on: control if the text is clipped to the axes rectangle
#     ax.text(1.35, i, label, transform = ax.transData, ha = "left", va = "center", fontsize = 10, clip_on= False)
    
# ax.axvline(1, color = "red", linestyle = "--", alpha = 0.5,  linewidth = 1.5, zorder = 1)

# ax.set_yticks(y_pos)
# ax.set_yticklabels(top_sig["Variable_new"], fontsize = 10)
# ax.set_xlabel("Odds Ratio (95%CI) for Prevalent MASLD", fontsize = 10, fontweight = "bold")
# ax.invert_yaxis()

# # plt.tight_layout()
# # plt.savefig("./Results/Forest_plot_of_species.png", dpi = 300, bbox_inches = "tight")
# plt.savefig("./Results/Forest_plot_of_Genus.pdf", bbox_inches = "tight")


# # Run for the 116 microbes (significant after sensitivity analysis)

# In[45]:


# Read in the for 116 microbiome
common_species = pd.read_csv("./Results/Archive/Common_species.csv")
common_species


# In[46]:


variables_to_test = common_species["Common_species"].unique().tolist()
# variables_to_test


# In[47]:


# Loop through all exposure and run models
all_results_common = []

for i, exposure_var in enumerate(variables_to_test, 1):
    # print(f"Testing: {exposure_var}")  
    full_variable = {f"model_{exposure_var}": covariates_list + [exposure_var]}
    # run the model
    results = logist_model_df2(
        data = merge_df_MAFLD2,
        covariate_sets= full_variable,
        dependent_var= "MAFLD_baseline_diagnosis",
        exposure_only= True,
    )

    # Add metadata
    results["test_variable"] = exposure_var
    results["number"] = i

    # Append to results
    all_results_common.append(results)

# all_results_common

if all_results_common:
    final_results_common = pd.concat(all_results_common, ignore_index=True)

    # Multiple correction
    from statsmodels.stats.multitest import multipletests

    reject, pvalues_correct, _, _ = multipletests(
        final_results_common["pvalue"],
        method= "fdr_bh"
    )

    final_results_common["p_adj"] = pvalues_correct

final_results_common


# In[48]:


final_results_common_sig = final_results_common[final_results_common["p_adj"] < 0.05]

# function to define the color
def assign_color (or_val, ci_lower, ci_upper):
    # assign color based on OR
    if ci_upper < 1:
        return "#4393C3"
    elif ci_lower > 1:
        return "#D6604D"

top_sig = final_results_common_sig.sort_values("OR", ascending= True)

colors = [assign_color(row["OR"], row["lower_OR_ci"], row["upper_OR_ci"])
         for _, row in top_sig.iterrows()]


# In[49]:


top_sig


# In[50]:


import re

def format_index(idx):
    # extract the genus
    genus_match = re.search(r'g__(.+?)(?:_s__)', idx)
    genus = genus_match.group(1) if genus_match else "Unknown"
    # extract the species
    species_match = re.search(r's__(.+)', idx)
    species = species_match.group(1) if species_match else "Unknown"
    return f"s_{species} (g_{genus})"
top_sig = top_sig.copy()

top_sig["Variable_new"] = top_sig["Variable"].map(format_index)
top_sig


# In[51]:


# Forest plot 
fig, ax = plt.subplots(figsize = (12, 10))
# make room so the outside text is visible
plt.subplots_adjust(right = 0.8)

y_pos = range(len(top_sig))

# assign color
for i, (idx, row) in enumerate(top_sig.iterrows()):
    ax.scatter(row["OR"], i, s = 80, color = colors[i], edgecolors = "black", linewidth = 0.5, zorder = 3)
    ax.hlines(i, row["lower_OR_ci"], row["upper_OR_ci"], color =  colors[i], linewidth = 2, alpha = 0.7, zorder = 2)

    # generate the labels
    label = f'{row["OR"]:.2f} ({row["lower_OR_ci"]:.2f},{row["upper_OR_ci"]:.2f})'
    # ax.transAxes: using axis-fraction coordinates,
    # clip_on: control if the text is clipped to the axes rectangle
    ax.text(1.35, i, label, transform = ax.transData, ha = "left", va = "center", fontsize = 10, clip_on= False)
    
ax.axvline(1, color = "red", linestyle = "--", alpha = 0.5,  linewidth = 1.5, zorder = 1)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_sig["Variable_new"], fontsize = 10)
ax.set_xlabel("Odds Ratio (95%CI) for Prevalent MASLD", fontsize = 10, fontweight = "bold")
ax.invert_yaxis()


# plt.tight_layout()
# plt.savefig("./Results/Forest_plot_of_species.png", dpi = 300, bbox_inches = "tight")
plt.savefig("./Results/Forest_plot_of_Common_species_MASLD_v2.pdf", bbox_inches = "tight")


# In[52]:


final_results_common_sig.shape


# ##  Plot all result in one Figure

# In[76]:


# read in the species and liver metrics results
species_liver_metrics = pd.read_csv("./Results/Sig_Microbes_LiverMetrics_clean.csv")
species_liver_metrics.columns

species_liver_metrics.head(10)


# In[80]:


len(species_liver_metrics["Variable_new"])


# ## Combining 
# -- For Figure 4

# In[114]:


species_liver_metrics_all = pd.read_csv("./Results/Microbiome_to_Liver_metrics.csv")
species_liver_metrics_all.head()

species_liver_metrics_all["Variable_new"] =  species_liver_metrics_all["Variable"].map(format_index)
species_liver_metrics_all.head()


# In[115]:


final_results = pd.read_csv("./Results/Species_Prevalent_MASLD.csv")

final_results["Variable_new"] = final_results["Variable"].map(format_index)

final_results_sig = final_results[final_results["p_adj"] < 0.05]

import re

# def format_index(idx):
#     # extract the genus
#     genus_match = re.search(r'g__(.+?)(?:_s__)', idx)
#     genus = genus_match.group(1) if genus_match else "Unknown"
#     # extract the species
#     species_match = re.search(r's__(.+)', idx)
#     species = species_match.group(1) if species_match else "Unknown"
#     return f"s_{species} (g_{genus})"
final_results_sig = final_results_sig.copy()

final_results_sig["Variable_new"] = final_results_sig["Variable"].map(format_index)
final_results_sig

top_sig = final_results_sig.sort_values("OR", ascending= True)
top_sig.columns


# In[116]:


top_sig.shape

# Union of the species
Union_species = set(top_sig["Variable_new"]) | set(species_liver_metrics_select["Variable_new"])
len(Union_species)


# In[124]:


species_liver_metrics_all["Outcome"].unique()


# In[117]:


species_liver_metrics_select = species_liver_metrics_all.loc[species_liver_metrics_all["Variable_new"].isin(Union_species),:].copy()

# recode the outcome
recode_dict = {
    "speed_of_sound_qbox_rank_INT" : "Speed of Sound",
    "attenuation_coefficient_qbox_rank_INT" : "Attenuation coefficient",
    "dispersion_median_median_of_qboxes_rank_INT": "Liver Dispersion",
    "elasticity_median_median_of_qboxes_rank_INT": "Liver Elasticity",
    "velocity_median_median_of_qboxes_rank_INT": "Liver Velocity",
    "viscosity_median_median_of_qboxes_rank_INT": "Liver Viscosity"
}

species_liver_metrics_select["Outcome_new"] = species_liver_metrics_select["Outcome"].map(recode_dict)


# In[118]:


final_results_select = final_results.loc[final_results["Variable_new"].isin(Union_species), :]
final_results_select.shape
final_results_select.head()


# In[119]:


# Change the columan names of top_sig 
final_results_select = final_results_select.rename(columns = {"OR" : "Coefficient",
                         "lower_OR_ci": "CI_lower",
                         "upper_OR_ci": "CI_upper",
                         "Outcome" : "Outcome_new"})

final_results_select = final_results_select[["Variable_new", "Outcome_new", "Coefficient", "CI_lower", "CI_upper", "p_adj"]]

species_liver_metrics_select = species_liver_metrics_select[["Variable_new", "Outcome_new", "Coefficient", "CI_lower", "CI_upper", "p_adj"]]

df_select = pd.concat([species_liver_metrics_select, final_results_select], ignore_index=True)
df_select.head()


# In[126]:


# species_liver_metrics_select.loc[species_liver_metrics_select["Outcome_new"] == "Speed of Sound",]


# In[120]:


df_select["Outcome_new"].unique()


# In[121]:


df_select["Outcome_new"] = df_select["Outcome_new"].replace(
    {"MAFLD_baseline_diagnosis": "Prevalent MASLD"}
)

df_select


# In[122]:


df_select["Variable_new"] = df_select["Variable_new"].astype(str)
df_select["Outcome_new"] = df_select["Outcome_new"].astype(str)

# Get all outcomes and variables
outcomes = df_select["Outcome_new"].unique().tolist()
all_vars = df_select["Variable_new"].unique().tolist()

first_outcome = outcomes[0]
base_order = df_select.loc[df_select["Outcome_new"] == first_outcome, "Variable_new"].tolist()
extras = [v for v in all_vars if v not in base_order]
global_order = base_order + sorted(extras)
global_order_rev = global_order[::-1] # reverse the order

# map variables to y-axis
pos = {v: i for i, v in enumerate(global_order_rev)}

# Expand to full template
template = pd.MultiIndex.from_product(
    [outcomes, global_order], names=["Outcome_new", "Variable_new"]
).to_frame(index = False)
template


# In[57]:


# color_pos = "#D6604D"
# color_neg = "#4393C3"

color_pos = "#EF4757"
color_neg = "#1E90FF"


# In[127]:


df_select_full = template.merge(df_select, on = ["Outcome_new", "Variable_new"], how = "left")
df_select_full["y"] = df_select_full["Variable_new"].map(pos)
df_select_full["present"] = ~ df_select_full["Coefficient"].isna()

df_select_full


# In[129]:


df_select_full.loc[df_select_full["Outcome_new"] == "Speed of Sound",]


# In[135]:


masld_sub = df_select_full[df_select_full["Outcome_new"] == "Prevalent MASLD"].copy()
masld_sub = masld_sub.sort_values("Coefficient", ascending = False)
global_order = masld_sub["Variable_new"].tolist()

# create facetGrid
n_vars = len(global_order)
# height = max(4, 0.4 *n_vars)

g = sns.FacetGrid(
    df_select_full,
    col = "Outcome_new",
    sharey = True,
    sharex = False,
    height= 12 ,
    aspect = 0.85
)

for idx, (ax, outcome) in enumerate(zip(g.axes.flat, outcomes)):
   
    # subset for the outcome
    sub = df_select_full[df_select_full["Outcome_new"] == outcome].copy()

    # set reference value and label based on outcome type
    if outcome == "Prevalent MASLD":
        ref = 1
        xlabel = "OR (95% CI)"
        sub = sub.sort_values("Coefficient", ascending = False)
    else:
        ref = 0
        xlabel = "\u03B2 (95% CI)"

    # sub = sub.reset_index(drop = True)
    # sub["y"] = range(len(sub))
    sub["y"] = sub["Variable_new"].map({v: i for i, v in enumerate(global_order)})
    
    # vertical line at ref
    ax.axvline(ref, color = "grey", lw = 1, ls = "--", zorder = 0)

    for _, row in sub.iterrows():
        if pd.isna(row["Coefficient"]):
            continue

        if row["p_adj"] > 0.05:
            color = "#B0B0B0"
            # color = "#999999"
        else:
            color = color_pos if row["Coefficient"] >= ref else color_neg
        
        y = row["y"]
        est = row["Coefficient"]
        # y = sub.loc[present, "y"].to_numpy()
        # est = sub.loc[present, "Coefficient"].to_numpy()
        # lo = sub.loc[present, "CI_lower"].to_numpy()
        # hi = sub.loc[present, "CI_upper"].to_numpy()
        # xerr = np.vstack([est-lo, hi-est])
        xerr = [[est - row["CI_lower"]], [row["CI_upper"] - est]]

        ax.errorbar(
            est, y, 
            xerr = xerr,
            fmt = "o", 
            color = color, ecolor = color,
            elinewidth = 1, 
            # set the capsize value to remove the end caps on the error bars
            capsize = 0, 
            markersize = 8, 
            # zorder, controls the drawing order of plot elements, zorder = 2, (errorbars _ points)
            zorder = 2
        )

        ax.set_xlabel(xlabel, fontsize = 14, labelpad = 6)
    
    # set the y-ticks to full list (blank rows where missing)
    ax.set_yticks(range(len(global_order)))
    ax.set_yticklabels(global_order, fontsize = 14)
    ax.set_ylim(-0.5, len(global_order) - 0.5)
    ax.invert_yaxis()
    
    # ax.grid(axis = "y", linestyle= ":", alpha = 0.3)
     # ax.set_yticks(range(len(sub)))
     # ax.set_yticklabels(sub["Variable_new"].tolist(), fontsize = 12)
     # ax.set_ylim(-0.5, len(sub) - 0.5)

    # x-axis tick labels
    ax.tick_params(axis = "x", labelsize = 10)

    # Facet title
    short_title = outcome.replace("_qbox_rank_INT", "").replace("_", " ")#.title()
    ax.set_title(short_title, fontsize = 14, fontweight = "bold")

    # Add panel annotation only the first and last facet
    if idx == 0:
        panel_label = "A"
    elif idx == len(outcomes) - 1:
        panel_label = "B"
    else:
        panel_label = None

    if panel_label:
        ax.text(
            -0.1, 1.05,
            panel_label,
            transform = ax.transAxes,
            fontsize = 18,
            fontweight = "bold",
            va = "bottom",
            ha = "right"
        )
    # Define the xlim for each subplots
    if idx <= 5:
        ax.set_xlim(-0.1, 0.1)
        ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
    elif idx == 6:
        ax.set_xlim(0.7, 1.3)
        ax.set_xticks([0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3])

# Axis labels 
# g.set_axis_labels("Coefficient (95%CI)", "", fontsize = 12)
# g.fig.suptitle("Forest plot by outcome", y = 1.03, fontsize = 12, fontweight = "bold")

g.fig.set_size_inches(24, 8)
plt.tight_layout()
plt.show

g.fig.savefig("./Results/Species_Liver_Metrics_plot_v5.pdf", bbox_inches = "tight")


# In[59]:


# Combine the results for the liver metrics and MASLD


# # Combine results 
# 1. Association with prevalent MASLD
# 2. Association with Liver parameter

# In[60]:


# df1 = pd.read_csv("./Results/Microbiome_to_Liver.csv")
# df1 = df1.drop(columns = ["N"])
# df1


# In[61]:


# # Convert long format to wide format
# df1_wide = df1.pivot_table(
#     columns = "Outcome", # outcome become columns
#     index = "Variable",
#     values = ["Coefficient", "CI_lower", "CI_upper", "P_value", "p_adj"]
# )

# # Flatten multi-level columns
# df1_wide.columns = ["_".join(col).strip() for col in df1_wide.columns.values]
# df1_wide =  df1_wide.reset_index()
# df1_wide


# In[62]:


# df1_wide.columns


# In[63]:


# df2 = pd.read_csv("./Results/Species_Prevalent_MASLD.csv")
# df2


# In[64]:


# df2.columns


# In[65]:


# df2 = df2.drop(columns = ["Model", "test_variable", "number"])
# df2


# In[66]:


# df1_wide.columns


# In[67]:


# combine the two table
# combined_df = pd.merge(df2, df1_wide, on = "Variable", how="left")
# combined_df


# In[68]:


# # display(combined_df.columns)

# combined_df[combined_df["p_adj"] < 0.05]

# # select need columns
# coefficient_cols = combined_df.loc[combined_df["p_adj"] < 0.05].filter(like = "Coefficient").columns.tolist()

# results =  combined_df.loc[combined_df["p_adj"] < 0.05,["Variable", "OR"] + coefficient_cols]


# In[69]:


# results.to_csv("./Results/Compare_Species.csv", index = False)


# # Looking at those specific diet-related species

# In[70]:


specific_species = pd.read_csv("./Results/Archive/DPs_specific_gut.csv")
spec_variables_to_test = specific_species["Specific_gut"].unique().tolist()


# In[71]:


# Loop through all exposure and run models
all_results_spec = []

for i, exposure_var in enumerate(spec_variables_to_test, 1):
    # print(f"Testing: {exposure_var}")  
    full_variable = {f"model_{exposure_var}": covariates_list + [exposure_var]}
    # run the model
    results = logist_model_df2(
        data = merge_df_MAFLD2,
        covariate_sets= full_variable,
        dependent_var= "MAFLD_baseline_diagnosis",
        exposure_only= True,
    )

    # Add metadata
    results["test_variable"] = exposure_var
    results["number"] = i

    # Append to results
    all_results_spec.append(results)

if all_results_spec:
    final_results_spec = pd.concat(all_results_spec, ignore_index=True)

    # Multiple correction
    from statsmodels.stats.multitest import multipletests

    reject, pvalues_correct, _, _ = multipletests(
        final_results_spec["pvalue"],
        method= "fdr_bh"
    )

    final_results_spec["p_adj"] = pvalues_correct

final_results_spec


# In[72]:


final_results_spec.sort_values(by = "OR")


# In[73]:


final_results_spec.to_csv("./Results/Specific_species_MASLD.csv", index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# ============================================================
Source: 04_Overlapped_microbiomes_Liver.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math


# In[17]:


df_models = pd.read_csv("./Results/DPs_to_Species_cont_1207.csv")
df_model2 = df_models[(df_models["Model"] == "Model2") & (df_models["p_adj"] < 0.05)]
df_model2['Variable'].value_counts()


# In[18]:


df_model2


# In[19]:


# df_models = pd.read_csv("./Results/DPs_to_Species_cont.csv")
df_model3 = df_models[(df_models["Model"] == "Model3") & (df_models["p_adj"] < 0.05)]
df_model3['Variable'].value_counts()


# In[20]:


# df_model3 = pd.read_csv("dps_to_microbiomes_model3_sig.csv")
# df_model3['Variable'].value_counts()


# In[21]:


df_model4 = df_models[(df_models["Model"] == "Model4") & (df_models["p_adj"] < 0.05)]
df_model4['Variable'].value_counts()


# In[22]:


test_tbl = df_model2.groupby("Outcome").count()
test_tbl_filter = test_tbl.loc[test_tbl["N"] == 5]
keep_model2 = test_tbl_filter.index
print(len(keep_model2))

df_model2 = df_model2[df_model2["Outcome"].isin(keep_model2)]
df_model2


# In[23]:


def same_direction(group):
    coeffs = group['Coefficient']
    # Check if all positive or all negative (excluding zeros)
    non_zero_coeffs = coeffs[coeffs != 0]
    if len(non_zero_coeffs) == 0:
        return False
    return (non_zero_coeffs > 0).all() or (non_zero_coeffs < 0).all()

consistent_outcomes = df_model2.groupby(['Outcome'])[["Coefficient"]].apply(same_direction)    


# In[24]:


sum(consistent_outcomes)
outcomes_to_keep = consistent_outcomes[consistent_outcomes].index
outcomes_to_keep

# mask
mask = df_model2.set_index(['Outcome']).index.isin(outcomes_to_keep)

df_model2_filter = df_model2[mask]
df_model2_filter


# In[25]:


df_model2_filter["Outcome"].nunique()


# In[26]:


# plot the results
df_model2_filter = df_model2_filter.assign(
    Variable=df_model2_filter['Variable'].astype('str'),
    Outcome=df_model2_filter['Outcome'].astype('str')
)
df_model2_filter


# In[27]:


# plot the heatmap for the coefficient
Microbe_heatmap_2 = df_model2_filter.pivot(index = "Outcome", columns= "Variable", values = "Coefficient")
Microbe_heatmap_2


# In[28]:


Microbe_heatmap_2.rename(columns = {
    'AHEI_2010_score_eadj_scaled' : 'AHEI',
    "AMED_score_eadj_scaled" : 'AMED',
    "hPDI_score_eadj_scaled" : 'hPDI',
    'rDII_score_eadj_scaled' : 'rDII',
    'rEDIH_score_all_eadj_scaled':'rEDIH'}, inplace = True)
Microbe_heatmap_2


# In[29]:


Microbe_heatmap_2.index


# In[30]:


# recode the species name
import re

def format_index(idx):
    # extract the genus
    genus_match = re.search(r'g__(.+?)(?:_s__)', idx)
    genus = genus_match.group(1) if genus_match else "Unknown"
    # extract the species
    species_match = re.search(r's__(.+)', idx)
    species = species_match.group(1) if species_match else "Unknown"
    return f"s_{species} (g_{genus})"

Microbe_heatmap_2.index = Microbe_heatmap_2.index.map(format_index)
Microbe_heatmap_2


# In[31]:


Microbe_heatmap_2 = Microbe_heatmap_2[["AHEI", "hPDI", "rDII", "AMED", "rEDIH"]]
Microbe_heatmap_2


# In[39]:


get_ipython().system('pip install --quiet seaborn')


# In[40]:


import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import seaborn as sns


# In[34]:


# Define the discrete bundaries matching the reference
# bounds = [-0.3, -0.24, -0.18, -0.12, -0.06, 0, 0.06, 0.12, 0.18, 0.24, 0.3]
# norm = mcolors.BoundaryNorm(bounds, ncolors = 256)

# Select the TOP20 +/-
Top20 = (df_model2_filter.sort_values("Coefficient", ascending = False)
    .assign(Outcome_fmt = lambda d: d["Outcome"].map(format_index))
    .drop_duplicates(subset = "Outcome_fmt")
    .head(20)["Outcome_fmt"].values)

least20 = (df_model2_filter.sort_values("Coefficient", ascending = True)
    .assign(Outcome_fmt = lambda d: d["Outcome"].map(format_index))
    .drop_duplicates(subset = "Outcome_fmt")
    .head(20)["Outcome_fmt"].values)


# Subset the heatmap data
TOP20_plot = Microbe_heatmap_2.loc[Top20, :]
Least20_plot = Microbe_heatmap_2.loc[least20, :]


# In[35]:


# top20 = df_model2_filter.nlargest(56, 'Coefficient')['Outcome'].map(format_index).unique()
# top20
# display(len(top20))

# least20 = df_model2_filter.nsmallest(66, 'Coefficient')['Outcome'].map(format_index).unique()
# least20
# len(least20)


# In[67]:


# Create figure: 2 heatmpap and colorbar on the right
fig = plt.figure(figsize = (20, 15))
gs = GridSpec(2, 2, width_ratios = [1, 1], 
              height_ratios = [20, 0.3], 
              hspace = 0.25, 
              wspace = 0.25)

ax_left = fig.add_subplot(gs[0,0])
ax_right = fig.add_subplot(gs[0,1])
# spans both rows on the right
cbar_ax = fig.add_subplot(gs[1,:]) 

# plot the heatmaps with shared norm, no individual colorbars
for ax, data, title, show_x in [
    (ax_left, Least20_plot, "Top 20 Positive Associations", True),
    (ax_right, TOP20_plot, "Top 20 Negative Associations", True)]:
    sns.heatmap(data, 
                ax = ax, 
                cmap='RdBu_r',  # Red-Blue diverging colormap
                center = 0,
                annot = True, # show values in cells
                fmt = ".2f", # format for the annotation 
                # square = True,
                linewidth = 1,
                linecolor = "black",
                cbar = False,
                xticklabels = show_x
               )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title, fontsize = 14, pad = 14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    ax.tick_params(axis = "y", labelsize = 12)
    ax.set_aspect("equal") # keep cells square but lets the axes resize

    if show_x:
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, ha = "right")

# Add annotations
ax_left.text(-0.65, 1.05, "B", transform = ax_left.transAxes, fontsize = 16, 
             fontweight = "bold", va = "bottom", ha = "right")

ax_right.text(-0.65, 1.05, "C", transform = ax_right.transAxes, fontsize = 16, 
             fontweight = "bold", va = "bottom", ha = "right")

# create the color matching
abs_max = max(
    abs(TOP20_plot.min().min()), abs(TOP20_plot.max().max()),
    abs(Least20_plot.min().min()), abs(Least20_plot.max().max()), 0.3)

print(abs_max)

norm = mcolors.TwoSlopeNorm(vmin = -abs_max, vcenter = 0, vmax = abs_max)

sm = cm.ScalarMappable(cmap = "RdBu_r", norm = norm)
sm.set_array([])
# remove the dedicated axes
cbar_ax.remove()
cb = fig.colorbar(sm, # cax = cbar_ax, 
                  ax = [ax_left, ax_right],
                  orientation = "horizontal",
                 shrink = 0.3, aspect = 30, pad = 0.08, 
                  location = "bottom")

# the labelpad set the distance between the colorbar and label text
cb.set_label("Coefficients", fontsize = 14, labelpad = 5)
cb.ax.tick_params(labelsize = 12)

# set the tick positions
cb.set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])

# save the figure 
fig.savefig("./Results/Heatmap_B_AND_C.pdf", bbox_inches = "tight")
plt.show()


# In[83]:


TOP20_plot = Microbe_heatmap_2.loc[top20,:]
TOP20_plot

Least20_plot = Microbe_heatmap_2.loc[least20,:]
Least20_plot

import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def heatmap_plot(data, ax = None):
    fig, ax = plt.subplots(figsize=(15,15)) if ax is None else (None, ax)

    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes('top', size = "4%", pad = 0.6)
    hp_model1 = sns.heatmap(data, 
            cmap='RdBu_r',  # Red-Blue diverging colormap
            center=0,       # Center the colormap at 0
            annot=True,     # Show values in cells
            fmt='.2f',      # Format for the annotations (2 decimal places)
            cbar_kws={'label': 'Coefficient',
                      'orientation': 'horizontal',
                     'fraction': 0.05,
                     'shrink': 0.8,
                     'aspect': 40,
                     'pad': 0.01}, # pad change the distance between heatmap and legend
            square=True,
            cbar_ax = cbar_ax,
            linewidths = 1,
            linecolor='black',
#             xticklabels=False,
            ax = ax) 
    
# Rotate x-axis labels for better readability
    plt.xlabel('') # remove the x-axis title
    plt.ylabel('') # remove the y-axis title
    plt.tick_params(axis='x', top=False, bottom= True,
                labeltop = False, labelbottom=True)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)


# Adjust layout to prevent label cutoff
# Gives more space for y-axis labels
# plt.subplots_adjust(left=0.2)  
#    plt.tight_layout()

# plt.savefig("heatmap_model1_crude.jpg", bbox_inches = 'tight')
#    plt.show()
    return hp_model1


# In[79]:


heatmap_plot(data=TOP20_plot.T)


# In[80]:


heatmap_plot(data=Least20_plot.T)


# In[88]:


# Combine two figure into one
# %matplotlib inline

fig, axes = plt.subplots(1, 2, figsize = (18, 10))

heatmap_plot(data=TOP20_plot, ax = axes[0])
heatmap_plot(data=Least20_plot, ax = axes[1])

# add panel labels
axes[0].text(
    -0.2, 1.1, "B", transform = axes[0].transAxes, fontsize = 24, fontweight = "bold", va = "top", ha = "left"
)

axes[1].text(
    -0.2, 1.1, "C", transform = axes[1].transAxes, fontsize = 24, fontweight = "bold", va = "top", ha = "left"
)

axes[0].set_xlabel('')
axes[0].set_ylabel('')
axes[1].set_xlabel('')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig("./Results/Top20_pos_negative_model2.pdf", bbox_inches = "tight")
plt.show()


# In[99]:


# # set cutoff
# adjp_cutoff = 0.05

# # list for insignifcant results
# x_postive=[]
# x_negative=[]

# y_notsignificant = []
# y_sig = []

# for i in df_model4_full.index:
#     if df_model4_full.loc[i,:][8] <= adjp_cutoff:
        
#         # if the coefficient is positive or negative
#         if df_model4_full.loc[i,:][3] > 0:
#             x_postive.append(df_model4_full.loc[i,:][3])
#         else:
#             x_negative.append(df_model4_full.loc[i,:][3])
#         y_sig.append(-math.log10(df_model4_full.loc[i,:][8]))
        
#     elif df_model4_full.loc[i,:][8] > adjp_cutoff:
#         y_notsignificant.append(-math.log10(df_model4_full.loc[i,:][8]))


# In[100]:


# print(len(x_postive))
# print(len(x_negative))
# print(len(y_sig))
# len(y_notsignificant)


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')

# get the -log10 for the lines
log10_adjp_cutoff = -math.log10(adjp_cutoff)

data = (y_notsignificant, y_sig)
colors = ("grey", "red")


# In[89]:


# s__Haemophilus_parainfluenzae
df_model2.loc[df_model2['Outcome'].str.contains("parainfluenzae")]


# In[90]:


grouped = df_model2_filter.groupby(df_model2_filter['Variable'])
df_rEDIH = grouped.get_group("rEDIH_score_all_eadj_scaled")
df_rEDIH


# In[91]:


df_AHEI_2010 = grouped.get_group("AHEI_2010_score_eadj_scaled")
df_AMED = grouped.get_group("AMED_score_eadj_scaled")
df_hPDI = grouped.get_group("hPDI_score_eadj_scaled")
df_rDII = grouped.get_group("rDII_score_eadj_scaled")
df_rEDIH = grouped.get_group("rEDIH_score_all_eadj_scaled")


# In[92]:


# set intersection
lists = [df_AHEI_2010['Outcome'],df_AMED['Outcome'], df_hPDI['Outcome'],df_rDII['Outcome'], df_rEDIH['Outcome']]
overlapped = set.intersection(*map(set, lists))
len(overlapped) # 137 


# In[84]:


# print(overlapped)


# In[85]:


# cols_check = ["AHEI_2010_score_eadj_scaled", "hPDI_score_eadj_scaled", "rDII_score_eadj_scaled", 
#               "AMED_score_eadj_scaled", "rEDIH_score_all_eadj_scaled"]

# df_analysis = df_analysis.dropna(subset = cols_check)

# df_analysis_copy = df_analysis.copy()
# for i in cols_check:
#     print(i)
#     # generate the tertile
#     df_analysis_copy[f"{i}_tertile"] = pd.qcut(df_analysis_copy[i], q=3, labels=['Low', 'Medium', 'High'])

# df_analysis_copy


# In[86]:


# print(df_analysis_copy.columns.tolist())


# In[87]:


# import seaborn as sns
# sns.boxplot(data=df_analysis_copy, 
#             x = "AHEI_2010_score_eadj_scaled_tertile", 
#             y = "shannon_skbio",
#             fill = False,
#             hue = "AHEI_2010_score_eadj_scaled_tertile")


# In[97]:


# # sns.boxplot(data=df_analysis_copy, x="AMED_score_eadj_scaled_tertile", y="shannon_skbio", hue = "AMED_score_eadj_scaled_tertile", fill = False)
# sns.boxplot(data = df_analysis_copy, 
#             x = "rDII_score_eadj_scaled_tertile", 
#             y = "shannon_skbio", 
#             # hue = "rDII_score_eadj_scaled_tertile",  fill = False)


# In[89]:


# sns.boxplot(data=df_analysis_copy, x="hPDI_score_eadj_scaled_tertile", y="shannon_skbio", hue = "hPDI_score_eadj_scaled_tertile",  fill = False)


# In[90]:


# sns.boxplot(data=df_analysis_copy, x="AMED_score_eadj_scaled_tertile", y="shannon_skbio", hue = "AMED_score_eadj_scaled_tertile",  fill = False)


# In[91]:


# sns.boxplot(data=df_analysis_copy, x="rEDIH_score_all_eadj_scaled_tertile", y="shannon_skbio", hue = "rEDIH_score_all_eadj_scaled_tertile",  fill = False)


# In[92]:


# liver measures
# df_liver = pd.read_csv("liver_ultrasound_baseline_selected.csv")
# df_liver


# In[93]:


# df_analysis_liver = pd.merge(df_analysis_copy, df_liver, on = "participant_id", how = "left")
# df_analysis_liver


# In[94]:


# df_analysis_liver_complete = df_analysis_liver.dropna(subset=["elasticity_qbox_mean", "velocity_qbox_mean", "viscosity_qbox_mean", "dispersion_qbox_mean", "attenuation_coefficient_qbox", "speed_of_sound_qbox"])
# df_analysis_liver_complete # 5317 individuals


# In[95]:


# # RNT transformation for the liver outcomes
# import sys
# import os
# import numpy as np
# import pandas as pd
# import scipy.stats as ss

# def rank_to_normal(rank, c, n):
#     # Standard quantile function
#     x = (rank - c) / (n - 2*c + 1)
#     return ss.norm.ppf(x)

# def rank_INT(series, c=3.0/8, stochastic=True):
#     """ Perform rank-based inverse normal transformation on pandas series.
#         If stochastic is True ties are given rank randomly, otherwise ties will
#         share the same value. NaN values are ignored.

#         Args:
#             param1 (pandas.Series):   Series of values to transform
#             param2 (Optional[float]):  Constand parameter (Bloms constant)
#             param3 (Optional[bool]):  Whether to randomise rank of ties
        
#         Returns:
#             pandas.Series
#     """

#     # Check input
#     assert(isinstance(series, pd.Series))
#     assert(isinstance(c, float))
#     assert(isinstance(stochastic, bool))

#     # Set seed
#     np.random.seed(123)

#     # Take original series indexes
#     orig_idx = series.index

#     # Drop NaNs
#     series = series.loc[~pd.isnull(series)]

#     # Get ranks
#     if stochastic == True:
#         # Shuffle by index
#         series = series.loc[np.random.permutation(series.index)]
#         # Get rank, ties are determined by their position in the series (hence
#         # why we randomised the series)
#         rank = ss.rankdata(series, method="ordinal")
#     else:
#         # Get rank, ties are averaged
#         rank = ss.rankdata(series, method="average")

#     # Convert numpy array back to series
#     rank = pd.Series(rank, index=series.index)

#     # Convert rank to normal distribution
#     transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
    
#     return transformed[orig_idx]

# liver_ultrasound_para = ["elasticity_qbox_mean", "velocity_qbox_mean", "viscosity_qbox_mean", 
#                     "dispersion_qbox_mean", "attenuation_coefficient_qbox", "speed_of_sound_qbox"]

# # df_analysis_liver_complete[liver_ultrasound_para + '_rnd'] = df_analysis_liver_complete[liver_ultrasound_para].apply(rank_INT)

# liver_transformed = df_analysis_liver_complete[liver_ultrasound_para].apply(rank_INT)

# liver_transformed = liver_transformed.rename(columns = {
#     "elasticity_qbox_mean" : "elasticity_qbox_mean_rnd",
#     "velocity_qbox_mean" : "velocity_qbox_mean_rnd",
#     "viscosity_qbox_mean" : "viscosity_qbox_mean_rnd",
#     "dispersion_qbox_mean" : "dispersion_qbox_mean_rnd",
#     "attenuation_coefficient_qbox" : "attenuation_coefficient_qbox_rnd",
#     "speed_of_sound_qbox" : "speed_of_sound_qbox_rnd",
# })

# liver_transformed


# In[96]:


# df_analysis_liver_complete_2 = pd.concat([df_analysis_liver_complete, liver_transformed], axis = 1)
# df_analysis_liver_complete_2


# ## MLR analysis

# In[3]:


# read in function from .py file
# from A0_helpfunction import *


# In[73]:


# # outcomes_liver = [item + "_rnd" for item in liver_ultrasound_para]
# outcomes_liver

# micro_liver_model1 = pd.DataFrame()

# # test model
# model1 = None
# for predict_var in overlapped:
#     # print(predict_var)
#     for out in outcomes_liver:
#         lm_fit = run_multiple_regressions(df = df_analysis_liver_complete_2, predictors = [predict_var], outcomes = [out], control_vars = model1)    
#         micro_liver_model1 = pd.concat([micro_liver_model1, lm_fit], ignore_index=True)


# In[74]:


# from scipy import stats
# micro_liver_model1['p_adj'] = stats.false_discovery_control(micro_liver_model1['P_Value'])
# micro_liver_model1.loc[micro_liver_model1["p_adj"]<0.05].sort_values(["Coefficient", "Outcome"])


# In[75]:


# # model2
# Nutrients = pd.read_csv("Food log.csv")
# Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
# Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

# Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
# Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
# Nutrients_2.info()

# # merge with alcohol intake
# df_analysis_liver_complete_2 = pd.merge(df_analysis_liver_complete_2, Nutrients_2, on = "participant_id", how = 'left')
# df_analysis_liver_complete_2


# In[76]:


# micro_liver_model2 = pd.DataFrame()

# df_analysis_liver_complete_2.loc[:,'sex_all'] =  df_analysis_liver_complete_2['sex_all'].astype('category')
# df_analysis_liver_complete_2.loc[:,"sex_status"] = pd.get_dummies(df_analysis_liver_complete_2['sex_all'], prefix='sex', drop_first=True)
# df_analysis_liver_complete_2["sex_status"]

# smoking_status = pd.get_dummies(df_analysis_liver_complete_2['smoking_status'], prefix='smoking_status', drop_first=False)
# smoking_status = smoking_status.drop('smoking_status_Never', axis=1) 
# smoking_status

# education_status = pd.get_dummies(df_analysis_liver_complete_2['edu_status'], prefix='education', drop_first=False)
# education_status = education_status.drop('education_Low', axis=1)
# education_status.shape

# df_analysis_liver_complete_2 = pd.concat([df_analysis_liver_complete_2, smoking_status, education_status], axis = 1)
# df_analysis_liver_complete_2


# In[77]:


# model2 = ['sex_status', 'age', 'smoking_status_Current', 'smoking_status_Former', 'education_High', 
#          'MET_hour', 'sleep_hours_daily', "alcohol_g"]

# for predict_var in overlapped:
#     # print(predict_var)
#     for out in outcomes_liver:
#         lm_fit = run_multiple_regressions(df = df_analysis_liver_complete_2, predictors = [predict_var], outcomes = [out], control_vars = model2)
#         micro_liver_model2 = pd.concat([micro_liver_model2, lm_fit], ignore_index=True)
        
# # select based on the pvalue
# micro_liver_model2['p_adj'] = stats.false_discovery_control(micro_liver_model2['P_Value'])
# micro_liver_model2.loc[micro_liver_model2["p_adj"]<0.05].sort_values(["Coefficient", "Outcome"])


# In[79]:


# micro_liver_model2

# micro_liver_model2['Variable'] = micro_liver_model2['Variable'].astype('str')
# micro_liver_model2['Outcome'] = micro_liver_model2['Outcome'].astype('str')

# micro_liver_model2_heatmap = micro_liver_model2.pivot(index = "Outcome", columns= "Variable", values = "Coefficient")


# In[85]:


# def heatmap_plot2(data):
#     fig, ax = plt.subplots(figsize=(25,35))
    
#     hp_model1 = sns.heatmap(data, 
#             cmap='RdBu_r',  # Red-Blue diverging colormap
#             center=0,       # Center the colormap at 0
#             annot=True,     # Show values in cells
#             fmt='.2f',      # Format for the annotations (2 decimal places)
#             cbar_kws={'label': 'Coefficient',
#                      'fraction': 0.05,
#                      'shrink': 0.5,
#                      'aspect': 40,
#                      'pad': 0.01}, # pad change the distance between heatmap and legend
#             square=True,
#             linewidths = 1,
#             linecolor='black',
# #             xticklabels=False,
#             ax = ax)   
# # Rotate x-axis labels for better readability
#     plt.xlabel('') # remove the x-axis title
#     plt.tick_params(axis='x', top=True, bottom=False,
#                 labeltop = True, labelbottom=False)
#     plt.xticks(rotation=90, ha='right')
#     plt.yticks(rotation=0)


# # Adjust layout to prevent label cutoff
# # Gives more space for y-axis labels
# # plt.subplots_adjust(left=0.2)  
#     plt.tight_layout()

# # plt.savefig("heatmap_model1_crude.jpg", bbox_inches = 'tight')
#     plt.show()
#     return hp_model1
# heatmap_plot2(data=micro_liver_model2_heatmap.T)


# In[86]:


temp = micro_liver_model2.loc[micro_liver_model2["p_adj"]<0.05].sort_values(["Coefficient", "Outcome"])
temp['Outcome'].value_counts()


# In[87]:


temp.loc[temp['Variable'] == "s__Flavonifractor_plautii"]


# In[88]:


temp.loc[temp['Outcome'] == "elasticity_qbox_mean_rnd"]


# In[89]:


temp.loc[temp['Outcome'] == "viscosity_qbox_mean_rnd"]


# In[90]:


temp.loc[temp['Outcome'] == "dispersion_qbox_mean_rnd"]


# In[91]:


temp.loc[temp['Outcome'] == "velocity_qbox_mean_rnd"]



# ============================================================
Source: 05_Mediation_analysis_for_Microbes-v3.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Mediation analysis for the overlapped species and overlapped genus levels

# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[1]:


# 1. read in the overlapped microbes names
import pandas as pd

df_microbes = pd.read_csv("./Results/Microbiome_to_Liver.csv")
df_microbes


# In[2]:


df_microbes_speed = df_microbes[(df_microbes["Outcome"] == "speed_of_sound_qbox_rank_INT") & (df_microbes["p_adj"] < 0.05)]
df_microbes_speed


# In[3]:


len(df_microbes_speed)


# In[4]:


DP_microbiome_models = pd.read_csv(filepath_or_buffer= "./Results/DPs_to_Species_cont_1207.csv")
DP_microbiome_model2 = DP_microbiome_models[(DP_microbiome_models["Model"] == "Model2") & (DP_microbiome_models["p_adj"]<0.05)]
DP_microbiome_model2.shape


# In[5]:


# 316 significant DP-related microbiome
DP_microbiome_model2.Outcome.nunique()


# In[6]:


DP_microbiome_model2_select = DP_microbiome_model2[DP_microbiome_model2["Outcome"].isin(df_microbes_speed["Variable"])]
# DP_microbiome_model2_select


# In[7]:


# DP_microbiome_model2_select["Outcome"].value_counts()
# DP_microbiome_model2_select[DP_microbiome_model2_select["Variable"] == "AHEI_2010_score_eadj_scaled"]
# print(DP_microbiome_model2_select[DP_microbiome_model2_select["Variable"] == "AHEI_2010_score_eadj_scaled"].shape)

group_dfs = dict(list(DP_microbiome_model2_select.groupby("Variable")))
print(group_dfs["AHEI_2010_score_eadj_scaled"].shape)
print(group_dfs["AMED_score_eadj_scaled"].shape)
print(group_dfs["hPDI_score_eadj_scaled"].shape)
print(group_dfs["rDII_score_eadj_scaled"].shape)
group_dfs["rEDIH_score_all_eadj_scaled"].shape


# In[8]:


group_dfs["AHEI_2010_score_eadj_scaled"].head()


# In[9]:


def find_overlap(lists):
    
    # convert the first list o a set
    common_elements = set(lists[0])

    # Insert with all other lists
    for lst in lists[1:]:
        common_elements = common_elements.intersection(set(lst))

    # convert back to list if desired
    return list(common_elements)

all_lists = [group_dfs["AHEI_2010_score_eadj_scaled"].Outcome,
             group_dfs["AMED_score_eadj_scaled"].Outcome,
             group_dfs["hPDI_score_eadj_scaled"].Outcome,
             group_dfs["rDII_score_eadj_scaled"].Outcome,
             group_dfs["rEDIH_score_all_eadj_scaled"].Outcome]

overlapped = find_overlap(all_lists)
len(overlapped)


# In[10]:


# overlapped
# overlap_genus = [item.split("|")[-2] for item in overlapped]
# overlap_genus
# len(set(overlap_genus))
# pd.DataFrame(overlap_genus).to_csv("Overlapped_genus.csv")


# In[11]:


# print(overlapped)
overlapped_mediators = [item.replace("|", "_") for item in overlapped]
overlapped_mediators[0:5]


# In[12]:


overlapped_mediators


# ## Read in the data for mediation analysis

# In[13]:


df_mediation = pd.read_csv("./Data/data_for_mediation_exclude_reportMASLD.csv")
df_mediation


# ## Mediation 

# In[14]:


print(df_mediation["sex_all"].value_counts())
df_mediation["edu_status"].value_counts()


# In[15]:


print(df_mediation["age"].isnull().sum())

lifestyles = pd.read_csv("../02_Lifestyle factors/lifestyle_factor_all_disease_V2.csv")
lifestyles_age = lifestyles[["participant_id", "age_all"]]
lifestyles_age

# merge with merge_df_MAFLD
df_mediation = pd.merge(df_mediation,
                         lifestyles_age, on="participant_id", how = "left")
df_mediation["age_all"].isnull().sum()


# In[16]:


from statsmodels.stats.mediation import Mediation
import statsmodels.api as sm
from A0_helpfunction import *


# In[17]:


df_mediation.columns = df_mediation.columns.str.replace('|', '_')


# In[18]:


def calculate_missingness(df):
    # Total number of rows
    total_rows = len(df)
    
    # Calculate missing values count
    missing_count = df.isna().sum()
    
    # Calculate missing values percentage
    missing_percentage = (missing_count / total_rows) * 100
    
    # Create a summary DataFrame
    missingness_summary = pd.DataFrame({
        'missing_count': missing_count,
        'missing_percentage': missing_percentage
    })
    
    return missingness_summary

missingness = calculate_missingness(df_mediation)
missingness


# In[19]:


df_mediation.columns[df_mediation.columns.str.contains(r'use')]


# In[20]:


df_mediation_withoutmissing = df_mediation.dropna(subset = ["speed_of_sound_qbox_rank_INT", "AHEI_2010_score_eadj_scaled",
                                                           "age_all", "sex_all", "bmi", "edu_status", "smoking_status",  "sleep_hours_daily", 
                                                            "MET_hour", "Hormone_use", "Vit_use", "Antibio_use", "PPI_use"])
df_mediation_withoutmissing.shape # 3847 obs


# In[21]:


df_mediation_withoutmissing.columns[0:5]


# In[22]:


# outcome model
# outcome_model = sm.OLS.from_formula("speed_of_sound_qbox_rank_INT ~ AHEI_2010_score_eadj_scaled + g__Clostridium_s__Clostridium_fessum + age + sex_all + bmi + edu_status + smoking_status + sleep_hours_daily + MET_hour + Hormone_use + Vit_use",
#                                     df_mediation_withoutmissing)


# In[23]:


# medidation model
# mediator_model = sm.OLS.from_formula("g__Clostridium_s__Clostridium_fessum ~ AHEI_2010_score_eadj_scaled + age + sex_all + bmi + edu_status + smoking_status + sleep_hours_daily + MET_hour + Hormone_use + Vit_use", 
#                                      df_mediation_withoutmissing)


# In[24]:


# med = Mediation(outcome_model, mediator_model, exposure = "AHEI_2010_score_eadj_scaled", mediator = "g__Clostridium_s__Clostridium_fessum").fit()

# med.summary() # ACME : average causal mediated effect; ADE : average direct effect


# In[25]:


# med_summary = pd.DataFrame(med.summary()).reset_index()
# med_summary


# In[26]:


# Estimate = med_summary.loc[med_summary["index"] == "ACME (average)"]["Estimate"].values[0]
# Lower_ci = med_summary.loc[med_summary["index"] == "ACME (average)"]["Lower CI bound"].values[0]
# Upper_ci = med_summary.loc[med_summary["index"] == "ACME (average)"]["Upper CI bound"].values[0]
# # comebine them into a string
# Estimate_ci = f"{Estimate:.3f}[{Lower_ci:.3f}, {Upper_ci:.3f}]"
# Estimate_ci


# In[27]:


# check if the microbes are in to data
# df_mediation.columns.isin(["g__Clostridium_s__Clostridium_fessum"]).sum()


# ## Run the Mediation analysis
# - Write a function for the mediation analysis

# In[28]:


# # x1 = mediation_gut(data = df_mediation_withoutmissing, 
# #               exposure = "AHEI_2010_score_eadj_scaled", 
# #               controls = ["age", "sex_all", "bmi", "edu_status", "smoking_status",
# #                           "sleep_hours_daily", "MET_hour", "Hormone_use", "Vit_use"],
# #               dependent_var = "speed_of_sound_qbox_rank_INT",
# #               mediators = overlapped_mediators[0])

# # all the results for the mediation analysis
# import random
# random.seed(2025)

# all_results_AHEI = []

# for i in range(len(overlapped_mediators)):
    
#     print(overlapped_mediators[i])
#     # Mediation analysis
#     result_df = mediation_gut(data = df_mediation_withoutmissing, 
#               exposure = "AHEI_2010_score_eadj_scaled", 
#               controls = ["age", "sex_all", "bmi", "edu_status", "smoking_status",
#                           "sleep_hours_daily", "MET_hour", "Hormone_use", "Vit_use"],
#               dependent_var = "speed_of_sound_qbox_rank_INT",
#               mediators = overlapped_mediators[i])

#     # Append results into results list
#     all_results_AHEI.append(result_df)

# combine_results = pd.concat(all_results_AHEI, ignore_index = True)
# combine_results.to_csv("AHEI_mediation_results.csv", index = False)


# ## 1. AMED

# In[29]:


# import random
# random.seed(2025)
# all_results_AMED = []

# for i in range(len(overlapped_mediators)): 
#     print(overlapped_mediators[i])
#     # Mediation analysis
#     result_df = mediation_gut(data = df_mediation_withoutmissing, 
#               exposure = "AMED_score_eadj_scaled", 
#               controls = ["age_all", "sex_all", "bmi", "edu_status", "smoking_status",
#                           "sleep_hours_daily", "MET_hour", "Hormone_use", "Vit_use", "Antibio_use", "PPI_use"],
#               dependent_var = "speed_of_sound_qbox_rank_INT",
#               mediators = overlapped_mediators[i])

#     # Append results into results list
#     all_results_AMED.append(result_df)

# combine_results_AMED = pd.concat(all_results_AMED, ignore_index = True)
# combine_results_AMED.to_csv("./Results/AMED_mediation_results_overlappedGut.csv", index = False)


# # rDII

# In[30]:


import random
random.seed(20252)

all_results_rDII = []

for i in range(len(overlapped_mediators)):

    print(i)
    print(overlapped_mediators[i])
    # Mediation analysis
    result_df = mediation_gut(data = df_mediation_withoutmissing, 
              exposure = "rDII_score_eadj_scaled", 
              controls = ["age", "sex_all", "bmi", "edu_status", "smoking_status",
                          "sleep_hours_daily", "MET_hour", "Hormone_use", "Vit_use", "Antibio_use", "PPI_use"],
              dependent_var = "speed_of_sound_qbox_rank_INT",
              mediators = overlapped_mediators[i])

    # Append results into results list
    # remove the NA column
    result_df = result_df.dropna(axis = 1, how = "all")
    all_results_rDII.append(result_df)

all_results_rDII = pd.concat(all_results_rDII, ignore_index = True)
all_results_rDII.to_csv("./Results/rDII_mediation_results_overlappedGut_1215.csv", index = False)


# # rEDIH

# In[31]:


# # Run for the rEDIH
# import random
# random.seed(20254)

# all_results_rEDIH = []
# for i in range(len(overlapped_mediators)):

#     print(i)
#     print(overlapped_mediators[i])
#     # Mediation analysis
#     result_df = mediation_gut(data = df_mediation_withoutmissing, 
#               exposure = "rEDIH_score_all_eadj_scaled", 
#               controls = ["age_all", "sex_all", "bmi", "edu_status", "smoking_status",
#                           "sleep_hours_daily", 
#                           "MET_hour", "Antibio_use", "PPI_use",
#                           "Hormone_use", "Vit_use"
#                          ],
#               dependent_var = "speed_of_sound_qbox_rank_INT",
#               mediators = overlapped_mediators[i])

#     # Append results into results list
#     all_results_rEDIH.append(result_df)

# all_results_rEDIH = pd.concat(all_results_rEDIH, ignore_index = True)

# all_results_rEDIH.to_csv("./Results/rEDIH_mediation_results.csv", index = False)


# # hPDI

# In[32]:


Nutrients = pd.read_csv("Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
Nutrients_2.info()

# merge with alcohol intake
df_mediation_withoutmissing = pd.merge(df_mediation_withoutmissing, Nutrients_2, on = "participant_id", how = 'left')
df_mediation_withoutmissing


# In[33]:


df_mediation_withoutmissing["alcohol_g"].isna().sum()


# In[34]:


# Run for the hPDI
import random
random.seed(20254)

all_results_hPDI1 = []

for i in range(len(overlapped_mediators)):

    print(i)
    print(overlapped_mediators[i])
    # Mediation analysis
    result_df = mediation_gut(data = df_mediation_withoutmissing, 
              exposure = "hPDI_score_eadj_scaled", 
              controls = ["age_all", "sex_all", "bmi", "edu_status", "smoking_status",
                          "sleep_hours_daily", "MET_hour",
                          "Hormone_use", "Vit_use", "alcohol_g", "Antibio_use", "PPI_use"
                         ],
              dependent_var = "speed_of_sound_qbox_rank_INT",
              mediators = overlapped_mediators[i])

    # Append results into results list
    all_results_hPDI1.append(result_df)

combine_hPDI = pd.concat(all_results_hPDI1, ignore_index = True)
combine_hPDI.to_csv("./Results/hPDI_mediation_results.csv", index = False)


# In[35]:


# # Run for the hPDI
# import random
# random.seed(20254)

# all_results_hPDI1 = []

# for i in range(len(overlapped_mediators)):

#     print(i)
#     print(overlapped_mediators[i])
#     # Mediation analysis
#     result_df = mediation_gut(data = df_mediation_withoutmissing, 
#               exposure = "hPDI_score_eadj_scaled", 
#               controls = ["age", "sex_all", "bmi", "edu_status", "smoking_status",
#                           "sleep_hours_daily", "MET_hour",
#                           "Hormone_use", "Vit_use", "alcohol_g", "Antibio_use", "PPI_use"
#                          ],
#               dependent_var = "speed_of_sound_qbox_rank_INT",
#               mediators = overlapped_mediators[i])

#     # Append results into results list
#     all_results_hPDI1.append(result_df)


# In[36]:


# # combine two list together
# all_results_hPDI = pd.concat([all_results_hPDI1, all_results_hPDI2], axis = 0) # stack rows
# all_results_hPDI.to_csv("hPDI_mediation_results.csv", index = False)


# ## AHEI MEDIATION

# In[ ]:


import random
random.seed(2025)

all_results_AHEI = []

for i in range(len(overlapped_mediators)):
    print(i)
    print(overlapped_mediators[i])
    # Mediation analysis
    result_df = mediation_gut(data = df_mediation_withoutmissing, 
              exposure = "AHEI_2010_score_eadj_scaled", 
              controls = ["age_all", "sex_all", "bmi", "edu_status", "smoking_status",
                          "sleep_hours_daily", "MET_hour", "Hormone_use", "Vit_use",
                          "Antibio_use", "PPI_use"
                         ],
              dependent_var = "speed_of_sound_qbox_rank_INT",
              mediators = overlapped_mediators[i])

    # Append results into results list
    all_results_AHEI.append(result_df)

combine_results = pd.concat(all_results_AHEI, ignore_index = True)
combine_results.to_csv("./Results/AHEI_mediation_results_1215.csv", index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# ============================================================
Source: 06_Plot_Mediation_Results-V2.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# # Plot the results for mediation analysis
# Author: Keyong Deng

# In[1]:


get_ipython().system('pip install --quiet statsmodels')


# In[2]:


import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import numpy as np

def add_adjust_p(df, p_col = "prop_mediated_pval", method = "fdr_bh"):
    reject, prop_mediated_pval_adj, _, _ = multipletests(
        df[p_col],
        alpha = 0.05,
        method = method 
    )
    df["prop_mediated_pval_adj"] = prop_mediated_pval_adj
    return df


# In[3]:


AHEI_Mediation = pd.read_csv("./Results/AHEI_mediation_results_1215.csv")
# display(AHEI_Mediation.head())

AMED_Mediation = pd.read_csv("./Results/AMED_mediation_results_overlappedGut.csv")
# display(AMED_Mediation.head())

rDII_Mediation = pd.read_csv("./Results/rDII_mediation_results_overlappedGut_1215.csv")
# display(rDII_Mediation.head())

hPDI_Mediation = pd.read_csv("./Results/hPDI_mediation_results.csv")
# hPDI_Mediation.head()

# Add p_adj into different datasets
AHEI_Mediation = add_adjust_p(AHEI_Mediation)
AMED_Mediation = add_adjust_p(AMED_Mediation)
rDII_Mediation = add_adjust_p(rDII_Mediation)
hPDI_Mediation = add_adjust_p(hPDI_Mediation)


# In[4]:


results_test = pd.concat([AMED_Mediation, AHEI_Mediation, rDII_Mediation, hPDI_Mediation], ignore_index=True)
results_test.head()


# In[5]:


results_test[["Genus", "Species"]] = results_test["Mediator"].str.split("_s_", expand = True)
results_test["Species"] = 's_' + results_test['Species']

# Handle the strings
results_test[["prop_mediated_estimate", "prop_mediated_CI"]] = results_test["prop_mediated"].str.split("[", expand = True)
results_test["prop_mediated_CI"] = "[" + results_test["prop_mediated_CI"]
results_test.head()


# In[6]:


results_test.sort_values(by = ["prop_mediated_estimate"], ascending=False).head()


# In[7]:


# Recode the Exposure
results_test["Exposure"] = results_test["Exposure"].replace({"AHEI_2010_score_eadj_scaled" : "AHEI",
                                                            "AMED_score_eadj_scaled" : "AMED",
                                                            "hPDI_score_eadj_scaled" : "hPDI", 
                                                            "rDII_score_eadj_scaled" : "rDII",
                                                            })
results_test_copy = results_test.copy()
results_test_copy["prop_mediated_estimate"] = results_test_copy["prop_mediated_estimate"].astype(float)


# In[8]:


# AHEI_Mediation.loc[AHEI_Mediation["Mediator"].str.contains(r"welbionis"),]
# AHEI_Mediation["Exposure"] = AHEI_Mediation["Exposure"].replace({"AHEI_2010_score_eadj_scaled" : "AHEI"})
# AHEI_Mediation


# In[9]:


# pattern_data = results_test[results_test["Exposure"] == "AHEI"]
# pattern_data


# In[10]:


# range(len(pattern_data))
# colors["AHEI"]
# pattern_data["prop_mediated_estimate"] = pattern_data["prop_mediated_estimate"].astype(float)


# In[11]:


# # create bubble plot
# fig, ax = plt.subplots(figsize = (5,5))
# scatter = ax.scatter(
#         range(len(pattern_data)),
#         -np.log10(pattern_data["prop_mediated_pval_adj"]),
#         s = pattern_data["prop_mediated_estimate"] * 100,
#         c = colors["AHEI"],
#         alpha = 0.6,
#         edgecolors = "black",
#         linewidth = 0.5
#     )


# In[12]:


results_test_copy.columns


# # Plot the bubble figure

# In[13]:


# !pip install --quiet adjustText


# In[14]:


# from adjustText import adjust_text


# In[15]:


results_test_copy

results_test_copy["Species"] = results_test_copy["Species"].str.replace("s__", "", regex = False)


# In[16]:


results_test_copy


# In[17]:


results_test_copy["prop_mediated_pval_adj"].describe()


# In[18]:


results_test_copy[results_test_copy["prop_mediated_pval_adj"] == 0]


# In[19]:


results_test_copy[results_test_copy["prop_mediated_pval_adj"] <= 0.05].Exposure.value_counts()


# In[20]:


get_ipython().system('pip install --quiet seaborn')


# In[21]:


import re
def format_index(idx):
    # extract the genus
    genus_match = re.search(r'g__(.+?)(?:_s__)', idx)
    genus = genus_match.group(1) if genus_match else "Unknown"
    # extract the species
    species_match = re.search(r's__(.+)', idx)
    species = species_match.group(1) if species_match else "Unknown"
    return f"s_{species} (g_{genus})"
    
results_test_copy = results_test_copy.copy()
results_test_copy["Mediator"] = results_test_copy["Mediator"].map(format_index)


# In[22]:


results_test_copy


# In[23]:


dietary_patterns = ["AHEI", "AMED", "rDII", "hPDI"]

sig_microbiomes = []
for pattern in dietary_patterns:
    pattern_data_temp = results_test_copy[results_test_copy["Exposure"] == pattern]
    sig = pattern_data_temp[pattern_data_temp["prop_mediated_pval_adj"] <= 0.05]["Mediator"].unique()
    sig_microbiomes.extend(sig)

sig_microbiomes = list(set(sig_microbiomes))
len(sig_microbiomes)


# In[30]:


from matplotlib.cm import get_cmap
import seaborn as sns
from matplotlib.lines import Line2D

# get unique microbiome names 
# all_microbiomes = results_test_copy["Mediator"].unique()
sig_microbiomes = results_test_copy[
    results_test_copy["prop_mediated_pval_adj"] <= 0.05
]["Mediator"].unique()

# cmap = get_cmap("tab20", len(all_microbiomes))
# husl_colors = sns.color_palette("husl", n_colors = len(all_microbiomes))

# sig_data = results_test_copy[results_test_copy["prop_mediated_pval_adj"] <= 0.05]

# top5_species = (sig_data.groupby("Mediator")["prop_mediated_estimate"]
#                 .max()
#                 .nlargest(8)
#                 .index.tolist())
# top5_species


# In[84]:


get_ipython().run_line_magic('config', 'inlineBackend.figure_format = "retina"')
plt.rcParams["figure.dpi"] = 600

colors_nature = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4F9",
    "#4477AA",
    "#EE6677",
    "#228833", 
    "#AA3377",
    "#BBCC33", 
    "#332288",
    "#88CCEE",
    "#F0E442"
]

microbiome_color_map = dict(zip(sig_microbiomes, colors_nature))

# plot layout
fig, axes = plt.subplots(2, 2, sharey = True, figsize = (14, 10), 
                         constrained_layout = False)

fig.subplots_adjust(right = 0.75, hspace = 0.1, wspace = 0.15)

# fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
axes = axes.flatten()

for idx, pattern in enumerate(dietary_patterns):
    
    ax = axes[idx]
    pattern_data = results_test_copy[results_test_copy["Exposure"] == pattern]
    p_val_adj = np.maximum(pattern_data["prop_mediated_pval_adj"], 1e-3)

    # separate significant and non-significant points
    sig_mask = pattern_data["prop_mediated_pval_adj"] <= 0.05
    
    non_sig_mask = ~sig_mask

    sig_subset = pattern_data[sig_mask]

    # top 5 species
    top5_species = sig_subset.nlargest(5, "prop_mediated_estimate")["Mediator"].tolist()
    
    edge_color = ["red" if m in top5_species else "black" for m in sig_subset["Mediator"]]
    edge_width = [2.5 if m in top5_species else 0.5 for m in sig_subset['Mediator']]
    
    # create bubble plot (significant)
    scatter = ax.scatter(
        range(len(pattern_data[sig_mask])),
        -np.log10(p_val_adj[sig_mask]),
        s = pattern_data[sig_mask]["prop_mediated_estimate"] * 100 * 55,
        # c = colors[pattern],
        c = [microbiome_color_map[m] for m in pattern_data[sig_mask]["Mediator"]],
        # alpha = 0.8,
        # edgecolors = "black",
        edgecolors = edge_color,
        linewidth = edge_width
    )

    # Add proportion text for all significant bubles
    for xi, yi, val in zip(range(len(sig_subset)),
                           -np.log10(p_val_adj[sig_mask]),
                          sig_subset["prop_mediated_estimate"] * 100
                        ):
        ax.text(xi, yi, f"{val:.0f}%", ha = "center", va = "center", fontsize = 6,
               fontweight = "bold", c = "white")
                                                           

    scatter_nosig = ax.scatter(
        range(len(pattern_data[sig_mask]), len(pattern_data)),
        -np.log10(p_val_adj[non_sig_mask]),
        s = pattern_data[non_sig_mask]["prop_mediated_estimate"] * 100 * 55,
        # grey
        c = "#B0B0B0B0",
        # alpha = 0.8,
        # edgecolors = "black",
        linewidth = 0.5
    )

    # formatting for each subplot
    ax.set_title(pattern, fontsize = 12, fontweight = "bold")
    ax.set_xlabel(None)
    ax.set_ylabel("-log10 (p_value_adj)", fontsize = 12)
    ax.grid(False)
    ax.axhline(y = -np.log10(0.05), color = "black", linestyle = "--", linewidth = 1, alpha = 0.8)
    ax.set_ylim(0, 3.5)
    ax.set_xticks([]) # hide the x-axis ticks

# Legend for the buble size
sizes = [5, 10, 15, 20]
labels_size = ["5%", "10%", "15%", "20%"]
    
legend_bubbles = [plt.scatter([], [], 
                              s = val * 55, 
                              c = "black", 
                              alpha = 0.9, 
                              edgecolors = "black") for val in sizes]

# Add legend
legend_elements = [
    Line2D([0], [0], marker = "o", color = "w", markerfacecolor = color, 
          markersize = 8, label = name, markeredgecolor = "black", markeredgewidth = 0.5)
    for name, color in microbiome_color_map.items()
]

fig.legend(
    handles = legend_elements,
    title = "Microbiome Species",
    # loc = "lower left",
    loc = "center right",
    bbox_to_anchor = (1.1, 0.6),
    # bbox_to_anchor = (0.2, -0.25),
    # ncol = 3,
    ncol = 1,
    fontsize = 9, 
    title_fontsize = 10, frameon = False,
    handletextpad = 0.3,
    # columnspacing = 1,
)

legend_size = fig.legend(
    legend_bubbles, 
    labels_size,
    scatterpoints = 1, 
    title = "Mediating proportion (%)\n",
    # loc = "lower right",
    # ncol = 4,
    frameon = False,
    # bbox_to_anchor = (0.95, -0.15),
    ncol = 4,
    loc = "center right",
    bbox_to_anchor = (1.05, 0.3),
    fontsize = 9,
    title_fontsize = 10,
    handletextpad = 1.2,
    columnspacing = 2,
    #frameon = False
)

# Adjust the title padding 
# legend_size.get_title().set_position((0, -15))
# legend_size.get_title().set_y(0.4)

plt.savefig("./Results/Mediation_Microbes_to_Speed_of_Sound_v5.pdf", bbox_inches = "tight", dpi = 600)
# plt.savefig("./Results/Mediation_Microbes_to_Speed_of_Soud_2.png", dpi = 300, bbox_inches = "tight")
plt.show()


# In[25]:


dietary_patterns = ["AHEI", "AMED", "rDII", "hPDI"]

# colors = {
#     "AHEI" : "#F8B195",
#     "hPDI" : "#3C5488",
#     "rDII" : "#FF6B6B",
#     "AMED" : "#00A087"
# }

# Replace zero or very small p-value with minimum threshold
# create figure
fig, axes = plt.subplots(1, 4, sharey = True, figsize = (20, 5))

fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
axes = axes.flatten()

for idx, pattern in enumerate(dietary_patterns):
    
    ax = axes[idx]
    pattern_data = results_test_copy[results_test_copy["Exposure"] == pattern]
    p_val_adj = np.maximum(pattern_data["prop_mediated_pval_adj"], 1e-3)

    # separate significant and non-significant points
    sig_mask = pattern_data["prop_mediated_pval_adj"] <= 0.05
    non_sig_mask = ~sig_mask
    
    # create bubble plot (significant)
    scatter = ax.scatter(
        range(len(pattern_data[sig_mask])),
        -np.log10(p_val_adj[sig_mask]),
        s = pattern_data[sig_mask]["prop_mediated_estimate"] * 100 * 20,
        c = colors[pattern],
        alpha = 0.8,
        edgecolors = "black",
        linewidth = 0.5
    )

    scatter_nosig = ax.scatter(
        range(len(pattern_data[sig_mask]), len(pattern_data)),
        -np.log10(p_val_adj[non_sig_mask]),
        s = pattern_data[non_sig_mask]["prop_mediated_estimate"] * 100 * 20,
        c = "grey",
        alpha = 0.8,
        edgecolors = "black",
        linewidth = 0.5
    )

    # Add labels for significant only (top 5)
    
    texts = []
    sig_data = pattern_data[sig_mask].reset_index(drop = True)
    top_5 = sig_data["prop_mediated_estimate"].nlargest(5).index

    # Add specific specific species
    specific = ["Ruminococcus_gnavus"]
    
    for i, (idx_row, row) in enumerate(sig_data.iterrows()):
        p_val_adj2 = np.maximum(row["prop_mediated_pval_adj"], 1e-3)

        # only top 5
        if idx_row in top_5 or row["Species"] in specific:     
            txt = ax.text(i, -np.log10(p_val_adj2)+0.05,
                      row["Species"],
                      fontsize = 10,
                      va = "bottom")
            texts.append(txt)

    # adjust_text(texts, 
    #             arrowprops = dict(arrowstyle = "-", 
    #                               connectionstyle = "arc3",
    #                               color = "black", lw = 0.2),
    #             expand_points = (1.6, 1.6), expand_text = (1.5, 1.5),
    #             force_points = (1.3, 1.3), 
    #             force_text = (1.0, 1.0),
    #             # jitter = 0.5,
    #             ax = ax
    #            )

    # formatting for each subplot
    ax.set_title(pattern, fontsize = 10, fontweight = "bold")
    ax.set_xlabel(None)
    ax.set_ylabel("-log10(p_value_adj)", fontsize = 10)
    ax.grid(False)
    ax.axhline(y = -np.log10(0.05), color = "black", linestyle = "--", linewidth = 1, alpha = 0.8)
    ax.set_ylim(0, 3.5)
    ax.set_xticks([]) # hide the x-axis ticks

# Legend for the buble size
sizes = [0.1, 0.2, 0.3]
labels_size = ["10%", "20%", "30%"]
legend_bubbles = [plt.scatter([], [], s = s * 100 * 20, c = "grey", alpha = 0.6, edgecolors = "black") for s in sizes]

# Add legend
fig.legend(
    legend_bubbles, 
    labels_size,
    scatterpoints = 1, 
    title = "Mediating proportion (%)",
    loc = "lower center",
    ncol = 4,
    frameon = True,
    bbox_to_anchor = (0.5, -0.05)
)

# plt.tight_layout(rect = [0, 0.02, 1, 0.98])
plt.tight_layout()
plt.savefig("./Results/Mediation_Microbes_to_Speed_of_Sound_v3.pdf", bbox_inches = "tight", dpi = 300)
# plt.savefig("./Results/Mediation_Microbes_to_Speed_of_Soud_2.png", dpi = 300, bbox_inches = "tight")
plt.show()


# In[ ]:


# Plot the heatmap using python code
results_test_copy_select =  results_test_copy[['Exposure', "Species", "prop_mediated_estimate", "prop_mediated_CI",
                                               "prop_mediated_pval",  "prop_mediated_pval_adj"]]
results_test_copy_select


# In[ ]:


results_test_copy_select_sig = results_test_copy_select[results_test_copy_select["prop_mediated_pval_adj"] < 0.05]
results_test_copy_select_sig.sort_values(["Exposure" ,"prop_mediated_estimate"], ascending = False)


# In[ ]:


results_test_copy_select_sig.to_csv("./Results/Mediation_sig_SoS.csv")


# # Read the results for the mediationa analysis for the binary outcome

# In[ ]:


AHEI_mediation_MASLD = pd.read_csv("./Results/Archive/Mediation_Results_AHEI_MASLD.csv")
AMED_mediation_MASLD = pd.read_csv("./Results/Archive/Mediation_Results_AMED_MASLD.csv")
hPDI_mediation_MASLD = pd.read_csv("./Results/Archive/Mediation_Results_hPDI_MASLD.csv")
rDII_mediation_MASLD = pd.read_csv("./Results/Archive/Mediation_Results_rDII_MASLD.csv")


# In[ ]:


def add_adjust_p(df, p_col = "prop_mediation_pvalue", method = "fdr_bh"):
    reject, prop_mediated_pval_adj, _, _ = multipletests(
        df[p_col],
        alpha = 0.05,
        method = method 
    )
    df["prop_mediated_pval_adj"] = prop_mediated_pval_adj
    return df


# In[ ]:


AHEI_mediation_MASLD = add_adjust_p(AHEI_mediation_MASLD)
AMED_mediation_MASLD = add_adjust_p(AMED_mediation_MASLD)
hPDI_mediation_MASLD = add_adjust_p(hPDI_mediation_MASLD)
rDII_mediation_MASLD = add_adjust_p(rDII_mediation_MASLD)

# combine them into one
results_Med_MASLD = pd.concat([AHEI_mediation_MASLD, AMED_mediation_MASLD, hPDI_mediation_MASLD, rDII_mediation_MASLD], ignore_index=True)
results_Med_MASLD.head()


# In[ ]:


results_Med_MASLD = results_Med_MASLD[["Exposure", "Mediator", "Proportion_Mediated", "prop_mediation_ci_Lower", "prop_mediation_ci_Uppper"]]

results_Med_MASLD[["Genus", "Species"]] = results_Med_MASLD["Mediator"].str.split("_s_", expand = True)
results_Med_MASLD["Species"] = 's_' + results_Med_MASLD['Species']

results_Med_MASLD = results_Med_MASLD.sort_values(["Exposure", "Proportion_Mediated"])
results_Med_MASLD


# In[ ]:


results_Med_MASLD["Exposure"].unique()


# In[ ]:


results_Med_MASLD_select = results_Med_MASLD[["Exposure", "Species", "Proportion_Mediated", "prop_mediation_ci_Lower", "prop_mediation_ci_Uppper"]]

results_Med_MASLD_select = results_Med_MASLD_select.copy()

results_Med_MASLD_select["Exposure"] = results_Med_MASLD_select["Exposure"].replace(
    {"AHEI_2010_score_eadj_scaled" : "AHEI to prevalent MASLD",
    "AMED_score_eadj_scaled" : "AMED to prevalent MASLD",
    "hPDI_score_eadj_scaled" : "hPDI to prevalent MASLD",
    "rDII_score_eadj_scaled" :"rDII to prevalent MASLD"}
)
results_Med_MASLD_select


# In[ ]:


results_Med_MASLD_select["Exposure"] = results_Med_MASLD_select["Exposure"].astype(str)
results_Med_MASLD_select["Species"] = results_Med_MASLD_select["Species"].astype(str)

# Get all outcomes and variables
Exposure = results_Med_MASLD_select["Exposure"].unique().tolist()
Species = results_Med_MASLD_select["Species"].unique().tolist()

first_Exposure = Exposure[0]
base_order = results_Med_MASLD_select.loc[results_Med_MASLD_select["Exposure"] == first_Exposure, "Species"].tolist()
extras = [v for v in Species if v not in base_order]
global_order = base_order + sorted(extras)
global_order_rev = global_order[::-1] # reverse the order

# map variables to y-axis
pos = {v: i for i, v in enumerate(global_order_rev)}

# Expand to full template
template = pd.MultiIndex.from_product(
    [Exposure, global_order], names=["Exposure", "Species"]
).to_frame(index = False)
template


# In[ ]:


exposure_colors = {
    "AHEI to prevalent MASLD" : "#F8B195",
    "hPDI to prevalent MASLD" : "#3C5488",
    "rDII to prevalent MASLD" : "#FF6B6B",
    "AMED to prevalent MASLD" : "#00A087"
}


# In[ ]:


# Plot the forest plot
import seaborn as sns

color_pos = "black"

df_select_full = template.merge(results_Med_MASLD_select, on = ["Exposure", "Species"], how = "left")
df_select_full["y"] = df_select_full["Species"].map(pos)
# df_select_full["present"] = ~ df_select_full["Coefficient"].isna()

AHEI_sub = df_select_full[df_select_full["Exposure"] == "AHEI to prevalent MASLD"].copy()
AHEI_sub = AHEI_sub.sort_values("Proportion_Mediated", ascending = False)
global_order = AHEI_sub["Species"].tolist()

# create facetGrid
n_vars = len(global_order)
# height = max(4, 0.4 *n_vars)

g = sns.FacetGrid(
    df_select_full,
    col = "Exposure",
    sharey = True,
    sharex = False,
    height= 12 ,
    aspect = 0.85
)

for idx, (ax, Exposure) in enumerate(zip(g.axes.flat, Exposure)):

    xlabel = "Proportion_Mediated with 95%CI"
   
    # subset for the outcome
    sub = results_Med_MASLD_select[results_Med_MASLD_select["Exposure"] == Exposure].copy()
    # sub = sub.reset_index(drop = True)
    # sub["y"] = range(len(sub))
    sub["y"] = sub["Species"].map({v: i for i, v in enumerate(global_order)})
    # vertical line at ref
    ax.axvline(0, color = "grey", lw = 1, ls = "--", zorder = 0)

    color = exposure_colors[Exposure]

    for _, row in sub.iterrows():
        if pd.isna(row["Proportion_Mediated"]):
            continue

        # color = color_pos if row["Proportion_Mediated"] >= 0 else color_neg
        
        y = row["y"]
        est = row["Proportion_Mediated"]
        # y = sub.loc[present, "y"].to_numpy()
        # est = sub.loc[present, "Coefficient"].to_numpy()
        # lo = sub.loc[present, "CI_lower"].to_numpy()
        # hi = sub.loc[present, "CI_upper"].to_numpy()
        # xerr = np.vstack([est-lo, hi-est])
        xerr = [[est - row["prop_mediation_ci_Lower"]], [row["prop_mediation_ci_Uppper"] - est]]

        ax.errorbar(
            est, y, 
            xerr = xerr,
            fmt = "o", 
            color = color, 
            ecolor = color,
            elinewidth = 1, 
            # set the capsize value to remove the end caps on the error bars
            capsize = 0, 
            markersize = 8, 
            # zorder, controls the drawing order of plot elements, zorder = 2, (errorbars _ points)
            zorder = 2
        )

        ax.set_xlabel(xlabel, fontsize = 14, 
                      # spacing from the axis
                      labelpad = 10
                     )
    
    # set the y-ticks to full list (blank rows where missing)
    ax.set_yticks(range(len(global_order)))
    ax.set_yticklabels(global_order, fontsize = 14)
    ax.set_ylim(-0.5, len(global_order) - 0.5)
    ax.invert_yaxis()
    # ax.grid(axis = "y", linestyle= ":", alpha = 0.3)
     # ax.set_yticks(range(len(sub)))
     # ax.set_yticklabels(sub["Variable_new"].tolist(), fontsize = 12)
     # ax.set_ylim(-0.5, len(sub) - 0.5)

    # x-axis tick labels
    ax.tick_params(axis = "x", labelsize = 14)

    # Facet title
    ax.set_title(Exposure, fontsize = 14, fontweight = "bold")

    # Add panel annotation only the first and last facet
    # if idx == 0:
    #     panel_label = "A"
    # elif idx == len(outcomes) - 1:
    #     panel_label = "B"
    # else:
    #     panel_label = None

    # if panel_label:
    #     ax.text(
    #         -0.1, 1.05,
    #         panel_label,
    #         transform = ax.transAxes,
    #         fontsize = 14,
    #         fontweight = "bold",
    #         va = "bottom",
    #         ha = "right"
    #     )

# Axis labels 
# g.set_axis_labels("Coefficient (95%CI)", "", fontsize = 12)
# g.fig.suptitle("Forest plot by outcome", y = 1.03, fontsize = 12, fontweight = "bold")

g.fig.set_size_inches(20, 8)
plt.tight_layout()
plt.show
g.fig.savefig("./Results/Mediation_to_MASLD_v1.pdf", bbox_inches = "tight")


# In[ ]:


# # Plot the heatmap using python code
# results_test_copy_select =  results_test_copy[['Exposure', "Species", "prop_mediated_estimate", "prop_mediated_CI",
#                                                "prop_mediated_pval",  "prop_mediated_pval_adj"]]
# results_test_copy_select


# In[ ]:


# results_test_copy_select_sig = results_test_copy_select[results_test_copy_select["prop_mediated_pval_adj"] < 0.05]
# results_test_copy_select_sig.sort_values(["Exposure" ,"prop_mediated_estimate"])


# In[ ]:


# results_test_copy_select_sig["Exposure"].value_counts()


# ## Make the circular barplot results

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment

def add_labels(angles, values, labels, offset, ax):
    
    # This is the space between the end of the bar and the label
    padding = 0.05
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x= angle, 
            y= value + padding, 
            s= label, 
            ha= alignment, 
            va= "center", 
            rotation= rotation, 
            rotation_mode= "anchor"
        ) 


# In[ ]:


AHEI_Mediation = AHEI_Mediation[AHEI_Mediation["prop_mediated_pval"] < 0.05]
AHEI_Mediation.shape


# In[ ]:


# Create Angle
ANGLES = np.linspace(0, 2 * np.pi, len(AHEI_Mediation), endpoint = False)
VALUES = AHEI_Mediation["prop_mediated_estimate"].values
LABELS = AHEI_Mediation["Species"].values + " (" + AHEI_Mediation["prop_mediated_estimate"].values + ")"

# Create the width of each bar
Width = 2 * np.pi / len(VALUES)
Offset = np.pi/4
VALUES = VALUES.astype(float)


# In[ ]:


# Group = AHEI_Mediation["Genus"].values
# Group_size = [len(i[1]) for i in AHEI_Mediation.groupby("Genus")]
# Group_size

# get the right indexes 
# offset = 0
# IDXS = []
# for size in Group_size:
#     IDXS += list(range(offset + PAD, offset + size + PAD)


# In[ ]:


# Initialize Figure and Axis
fig, ax = plt.subplots(figsize = (10,10), subplot_kw = {"projection" : "polar"})

# specify offset
ax.set_theta_offset(Offset)

# Set limits for radial (y) axis. the negative lower bound creates the whole in the middle
# ax.set_ylim(0, 1)

# remove all spines
ax.set_frame_on(False)

# remove grid and tick marks
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Add bars and labels
ax.bar(ANGLES, VALUES, width = Width, linewidth = 2, color = "#f3868c", edgecolor = "white")
add_labels(
    ANGLES, VALUES, LABELS, Offset, ax
)


# In[ ]:


AMED_Mediation[["Genus", "Species"]] = AMED_Mediation["Mediator"].str.split("_s_", expand = True)
AMED_Mediation["Species"] = 's_' + AMED_Mediation['Species']

AMED_Mediation[["prop_mediated_estimate", "prop_mediated_CI"]] = AMED_Mediation["prop_mediated"].str.split("[", expand = True)
AMED_Mediation["prop_mediated_CI"] = "[" + AHEI_Mediation["prop_mediated_CI"]
AMED_Mediation.head()


# In[ ]:


AMED_Mediation.sort_values(by = ["prop_mediated_estimate"], ascending=False)


# In[ ]:


ANGLES = np.linspace(0, 2 * np.pi, len(AMED_Mediation), endpoint = False)
VALUES = AMED_Mediation["prop_mediated_estimate"].values
LABELS = AMED_Mediation["Mediator"].values

# Create the width of each bar
Width = 2 * np.pi / len(VALUES)
Offset = np.pi/2

VALUES = VALUES.astype(float)
VALUES

# Initialize Figure and Axis
fig, ax = plt.subplots(figsize = (20,10), subplot_kw = {"projection" : "polar"})

# specify offset
ax.set_theta_offset(Offset)

# Set limits for radial (y) axis. the negative lower bound creates the whole in the middle
# ax.set_ylim(0, 1)

# remove all spines
ax.set_frame_on(False)

# remove grid and tick marks
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Add bars and labels
ax.bar(ANGLES, VALUES, width = Width, linewidth = 2, color = "#61a4b2", edgecolor = "white")
add_labels(
    ANGLES, VALUES, LABELS, Offset, ax
)


# In[ ]:


rDII_Mediation[["Genus", "Species"]] = rDII_Mediation["Mediator"].str.split("_s_", expand = True)
rDII_Mediation["Species"] = 's_' + rDII_Mediation['Species']

rDII_Mediation[["prop_mediated_estimate", "prop_mediated_CI"]] = rDII_Mediation["prop_mediated"].str.split("[", expand = True)
rDII_Mediation["prop_mediated_CI"] = "[" + rDII_Mediation["prop_mediated_CI"]
rDII_Mediation.head()

ANGLES = np.linspace(0, 2 * np.pi, len(rDII_Mediation), endpoint = False)
VALUES = rDII_Mediation["prop_mediated_estimate"].values
LABELS = rDII_Mediation["Mediator"].values

# Create the width of each bar
Width = 2 * np.pi / len(VALUES)
Offset = np.pi/2

VALUES = VALUES.astype(float)
VALUES

# Initialize Figure and Axis
fig, ax = plt.subplots(figsize = (20,10), subplot_kw = {"projection" : "polar"})

# specify offset
ax.set_theta_offset(Offset)

# Set limits for radial (y) axis. the negative lower bound creates the whole in the middle
# ax.set_ylim(0, 1)

# remove all spines
ax.set_frame_on(False)

# remove grid and tick marks
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Add bars and labels
ax.bar(ANGLES, VALUES, width = Width, linewidth = 2, color = "#61a4b2", edgecolor = "white")
add_labels(
    ANGLES, VALUES, LABELS, Offset, ax
)


# In[ ]:


import numpy as np

hPDI_Mediation[["Genus", "Species"]] = hPDI_Mediation["Mediator"].str.split("_s_", expand = True)
hPDI_Mediation["Species"] = 's_' + hPDI_Mediation['Species']

hPDI_Mediation[["prop_mediated_estimate", "prop_mediated_CI"]] = hPDI_Mediation["prop_mediated"].str.split("[", expand = True)
hPDI_Mediation["prop_mediated_CI"] = "[" + hPDI_Mediation["prop_mediated_CI"]
hPDI_Mediation.head()

ANGLES = np.linspace(0, 2 * np.pi, len(hPDI_Mediation), endpoint = False)
VALUES = hPDI_Mediation["prop_mediated_estimate"].values
LABELS = hPDI_Mediation["Mediator"].values

# Create the width of each bar
Width = 2 * np.pi / len(VALUES)
Offset = np.pi/2

VALUES = VALUES.astype(float)
VALUES

# Initialize Figure and Axis
fig, ax = plt.subplots(figsize = (20,10), subplot_kw = {"projection" : "polar"})

# specify offset
ax.set_theta_offset(Offset)

# Set limits for radial (y) axis. the negative lower bound creates the whole in the middle
# ax.set_ylim(0, 1)

# remove all spines
ax.set_frame_on(False)

# remove grid and tick marks
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Add bars and labels
ax.bar(ANGLES, VALUES, width = Width, linewidth = 2, color = "#f3868c", edgecolor = "white")
add_labels(
    ANGLES, VALUES, LABELS, Offset, ax
)


# In[ ]:


# Adjust for the multiple test
hPDI_Mediation
rDII_Mediation
AMED_Mediation
AHEI_Mediation.head()


# In[ ]:


import statsmodels
from statsmodels.stats.multitest import multipletests

AHEI_Mediation["prop_mediated_pval_adj"] = multipletests(AHEI_Mediation["prop_mediated_pval"], method = "fdr_bh")[1]
AMED_Mediation["prop_mediated_pval_adj"] = multipletests(AMED_Mediation["prop_mediated_pval"], method = "fdr_bh")[1]
rDII_Mediation["prop_mediated_pval_adj"] = multipletests(rDII_Mediation["prop_mediated_pval"], method = "fdr_bh")[1]
hPDI_Mediation["prop_mediated_pval_adj"] = multipletests(hPDI_Mediation["prop_mediated_pval"], method = "fdr_bh")[1]


# In[ ]:


AHEI_Mediation.head()


# In[ ]:


AHEI_Mediation.loc[AHEI_Mediation["prop_mediated_pval_adj"] < 0.05].shape # 28 significant
AHEI_Mediation.loc[AHEI_Mediation["prop_mediated_pval_adj"] < 0.05]


# In[ ]:


print(AMED_Mediation.loc[AMED_Mediation["prop_mediated_pval_adj"] < 0.05].shape) # 29 significant
AMED_Mediation_sig = AMED_Mediation.loc[AMED_Mediation["prop_mediated_pval_adj"] < 0.05]
AMED_Mediation_sig = AMED_Mediation_sig.sort_values('prop_mediated_estimate')
AMED_Mediation_sig


# In[ ]:


print(rDII_Mediation.loc[rDII_Mediation["prop_mediated_pval_adj"] < 0.05].shape) # 23 significant
rDII_Mediation_sig = rDII_Mediation.loc[rDII_Mediation["prop_mediated_pval_adj"] < 0.05]
rDII_Mediation_sig = rDII_Mediation_sig.sort_values('prop_mediated_estimate')
rDII_Mediation_sig


# In[ ]:


print(hPDI_Mediation.loc[hPDI_Mediation["prop_mediated_pval_adj"] < 0.05].shape) # 1 significant
hPDI_Mediation_sig = hPDI_Mediation.loc[hPDI_Mediation["prop_mediated_pval_adj"] < 0.05]
hPDI_Mediation_sig = hPDI_Mediation_sig.sort_values('prop_mediated_estimate')
hPDI_Mediation_sig


# In[ ]:


print(hPDI_Mediation.loc[hPDI_Mediation["prop_mediated_pval_adj"] < 0.05].shape)


# In[ ]:


hPDI_Mediation


# In[ ]:






# ============================================================
Source: 11_Plot_GLM_Results-v2.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Plot the results from GLM
# ### Circular heatmap
# 
# Author: Keyong Deng, LUMC

# In[3]:


get_ipython().system('pip install --quiet mne')


# In[4]:


import warnings
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.stats.multi_comp import fdr_correction
from numpy import asarray, compress, not_equal, sqrt
from pandas import Series, DataFrame, concat
# from pandas.core.common import isnull
from scipy.stats import distributions, find_repeats, rankdata, tiecorrect
# rankdata: rank; tiecorrect:correction index


# In[5]:


from matplotlib import colors as mcolors
from matplotlib.colors import Normalize
def _get_scale_colors(cmaps, data, zero_is_middle = True, base_n=300, boundries=None, return_cmap=False):
    if boundries is None:
        data_plus_min = data - min(0, data.min())
        # standardization to [0,1]
        data_plus_min /= data_plus_min.max() 
        min_max_ratio = abs(data.min() / float(data.max()))
    else:
        data_plus_min = data + abs(boundries[0])
        data_plus_min /= (abs(boundries[0]) + boundries[1])
        min_max_ratio = abs(boundries[0] / float(boundries[1]))
    if len(cmaps) == 1:
        return [cmaps[0](i) for i in data_plus_min]

    colors1 = cmaps[0](np.linspace(0., 1, int(base_n*min_max_ratio)))
    colors2 = cmaps[1](np.linspace(0., 1, base_n))
    # combine them and build a new colormap
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    if return_cmap:
        return mymap
    return [mymap(i) for i in data_plus_min]


# In[6]:


import seaborn as sns
sns.palplot(_get_scale_colors([plt.cm.Blues_r, plt.cm.Reds], np.sort(np.random.uniform(-0.5, 0.5, 100)), boundries=[-1, 0.5] ))


# In[7]:


sns.palplot(_get_scale_colors([plt.cm.Blues_r, plt.cm.Reds], np.sort(np.random.uniform(-10, 30, 100)), boundries=[-10, 40] ))


# In[8]:


from matplotlib.colors import LinearSegmentedColormap

positive_color, negative_color = sns.color_palette('colorblind')[3], sns.color_palette('colorblind')[0]
cm_positive = LinearSegmentedColormap.from_list('positive', ['white', positive_color], N=1000)
cm_negative = LinearSegmentedColormap.from_list('negative', [negative_color, 'white'], N=1000)


# In[9]:


df_species = pd.read_csv("./Results/DPs_to_Species_cont_1207.csv")
df_species


# In[10]:


df_species_model1 = df_species[(df_species["Model"] == "Model1") & (df_species["p_adj"] < 0.05)]
df_species_model1.shape


# In[11]:


df_species_model2 = df_species[(df_species["Model"] == "Model2") & (df_species["p_adj"] < 0.05)]
df_species_model2.shape


# In[12]:


df_species_model3 = df_species[(df_species["Model"] == "Model3") & (df_species["p_adj"] < 0.05)]
df_species_model3.shape


# In[13]:


df_species_model4 = df_species[(df_species["Model"] == "Model4") & (df_species["p_adj"] < 0.05)]
df_species_model4.shape


# In[14]:


df_species_model4


# In[15]:


df_species_model1.groupby("Variable")["Outcome"].nunique()


# In[16]:


df_species_model2.groupby("Variable")["Outcome"].nunique()


# In[17]:


# check the largest and smallest coefficient of the dataframe
top_20 = pd.concat([
    df_species_model2.nsmallest(20, "Coefficient"),
    df_species_model2.nlargest(20, "Coefficient"),
])

top_20["Outcome_clean"] = top_20["Outcome"].str.split("s__", expand = True)[1]
top_20


# In[18]:


df_species_model2.groupby("Variable")["Outcome"].nunique()


# In[19]:


df_species_model2 = df_species[df_species["Model"] == "Model2"]
df_species_model2_copy = df_species_model2.copy()  
df_species_model2_copy.drop(columns=["N", "P_value", "Model"], inplace=True)
df_species_model2_copy


# In[20]:


df_species_model2_sig = df_species_model2_copy[df_species_model2_copy["p_adj"] < 0.05]

test_tbl = df_species_model2_sig.groupby("Outcome").count()
test_tbl

test_tbl_filter = test_tbl.loc[test_tbl["Variable"] == 5]
test_tbl_filter

# Extract the index 
keep_model2 = test_tbl_filter.index
print(len(keep_model2))

df_species_model2_sig = df_species_model2_sig[df_species_model2_sig["Outcome"].isin(keep_model2)]
df_species_model2_sig


# In[21]:


def same_direction(group):
    coeffs = group['Coefficient']
    # Check if all positive or all negative (excluding zeros)
    non_zero_coeffs = coeffs[coeffs != 0]
    if len(non_zero_coeffs) == 0:
        return False
    return (non_zero_coeffs > 0).all() or (non_zero_coeffs < 0).all()


# In[22]:


consistent_outcomes = df_species_model2_sig.groupby(['Outcome'])[["Coefficient"]].apply(same_direction)


# In[23]:


outcomes_to_keep = consistent_outcomes[consistent_outcomes].index
outcomes_to_keep # 138 species


# In[24]:


# Need to create the mask
mask = df_species_model2_sig.set_index(['Outcome']).index.isin(outcomes_to_keep)

df_species_model2_sig_all = df_species_model2_sig[mask]
df_species_model2_sig_all.shape


# In[25]:


# pandas dataframe, long to wide
# df_species_model4 = df_species_model3.set_index(["Outcome"]).unstack("Variable")
# df_species_model4
df_species_model2_sig_all_wide = df_species_model2_sig_all.pivot_table(
                          index='Outcome', 
                          columns='Variable', 
                          values=['Coefficient'])
df_species_model2_sig_all_wide


# In[26]:


df_species_model2_sig_all_wide.columns = ["AHEI", "AMED", "hPDI", "rDII", "rEDIH"]
df_species_model2_sig_all_wide.reset_index(inplace = True)
df_species_model2_sig_all_wide


# In[27]:


df_species_model2_sig_all_wide.loc[:,"Family"] = df_species_model2_sig_all_wide["Outcome"].str.split("_g__", expand=True)[0].str.split("__", expand = True)[1]
df_species_model2_sig_all_wide

df_species_model2_sig_all_wide.loc[:,"Species"] = df_species_model2_sig_all_wide["Outcome"].str.split("_s__", expand=True)[1]
df_species_model2_sig_all_wide

df_species_model2_sig_all_wide.loc[:,"Genus"] = df_species_model2_sig_all_wide["Outcome"].str.split("_g__", expand=True)[1].str.split("_s__", expand = True)[0]
df_species_model2_sig_all_wide


# In[28]:


# df_species_model2_sig_all_wide = df_species_model2_sig_all_wide.drop(columns = ["Outcome"])
df_species_model2_sig_all_wide_sorted = df_species_model2_sig_all_wide.sort_values("AHEI", ascending = False)
df_species_model2_sig_all_wide_sorted


# In[29]:


df_species_model2_sig_all_wide_sorted.drop(columns = ["Outcome"]).to_csv("./Results/Species_in_all_DPs.csv", index = False)


# In[30]:


df_species.shape


# In[31]:


# Read in the Diet-specific species
specific_species = pd.read_csv("./Results/DPs_specific_gut_1207.csv")
specific_species


# In[32]:


species_interested = specific_species[specific_species["Diet"] == "AHEI"].Specific_gut

pd.set_option("display.max_colwidth", 100)
df_species[(df_species['Outcome'].isin(species_interested)) & (df_species["Variable"] == "AHEI_2010_score_eadj_scaled")]


# In[33]:


species_interested = specific_species[specific_species["Diet"] == "hPDI"].Specific_gut

pd.set_option("display.max_colwidth", 100)
df_species[(df_species['Outcome'].isin(species_interested)) & (df_species["Variable"] == "hPDI_score_eadj_scaled")]


# In[34]:


species_interested = specific_species[specific_species["Diet"] == "rDII"].Specific_gut

pd.set_option("display.max_colwidth", 100)
df_species[(df_species['Outcome'].isin(species_interested)) & (df_species["Variable"] == "rDII_score_eadj_scaled")]


# In[35]:


species_interested = specific_species[specific_species["Diet"] == "rEDIH"].Specific_gut

pd.set_option("display.max_colwidth", 100)
df_species[(df_species['Outcome'].isin(species_interested)) & (df_species["Variable"] == "rEDIH_score_all_eadj_scaled") & (df_species["Model"] == "Model2")]


# #### Part2

# In[36]:


# Extract the results for 138 species but in other models
df_species_all = df_species[df_species['Outcome'].isin(df_species_model2_sig_all_wide["Outcome"])]
df_species_all


# In[37]:


df_species_all_sig = df_species_all[df_species_all["p_adj"] < 0.05]


# In[38]:


df_species_all_sig.shape


# In[39]:


df_species_all_sig["Model"].value_counts()


# In[40]:


print(len(set(df_species_all_sig[df_species_all_sig["Model"] == "Model3"].Outcome)))
print(len(set(df_species_all_sig[df_species_all_sig["Model"] == "Model4"].Outcome)))


# In[41]:


df_species_all_model3_sig = df_species_all_sig[df_species_all_sig["Model"] == "Model3"]
df_species_all_model3_sig

test_tbl = df_species_all_model3_sig.groupby("Outcome").count()
test_tbl

test_tbl_filter = test_tbl.loc[test_tbl["Variable"] == 5]
test_tbl_filter

# Extract the index 
keep_model3 = test_tbl_filter.index
print(len(keep_model3))

df_species_all_model3_sig = df_species_all_model3_sig[df_species_all_model3_sig["Outcome"].isin(keep_model3)]

consistent_outcomes = df_species_all_model3_sig.groupby(['Outcome'])[["Coefficient"]].apply(same_direction)
consistent_outcomes

outcomes_to_keep = consistent_outcomes[consistent_outcomes].index
outcomes_to_keep 


# In[42]:


df_species_all_model4_sig = df_species_all_sig[df_species_all_sig["Model"] == "Model4"]
df_species_all_model4_sig

test_tbl = df_species_all_model4_sig.groupby("Outcome").count()
test_tbl

test_tbl_filter = test_tbl.loc[test_tbl["Variable"] == 5]
test_tbl_filter

# Extract the index 
keep_model4 = test_tbl_filter.index
print(len(keep_model4))
df_species_all_model4_sig = df_species_all_model4_sig[df_species_all_model4_sig["Outcome"].isin(keep_model4)]

consistent_outcomes = df_species_all_model4_sig.groupby(['Outcome'])[["Coefficient"]].apply(same_direction)
consistent_outcomes

outcomes_to_keep = consistent_outcomes[consistent_outcomes].index
outcomes_to_keep 


# In[43]:


# for each model
# species_model1 = set(df_species_all[df_species_all["Model"] == "Model1"].Outcome)
species_model2 = set(df_species_all_sig[df_species_all_sig["Model"] == "Model2"].Outcome)
species_model3 = set(df_species_all_model3_sig[df_species_all_model3_sig["Model"] == "Model3"].Outcome)
species_model4 = set(df_species_all_model4_sig[df_species_all_model4_sig["Model"] == "Model4"].Outcome)

# store it in a dict
species_sets = {
    "Model2": species_model2,
    "Model3" : species_model3,
    "Model4" : species_model4
}

for model, species_sets in species_sets.items():
    print(f'{model}: {len(species_sets)} species') 


# In[44]:


# Common to all the models
common_all = species_model2 & species_model3 & species_model4
len(common_all)


# In[45]:


species_model2 - species_model3


# In[46]:


species_model2 - species_model4


# In[47]:


len(common_all)

common_species_across_models = pd.DataFrame(
    list(common_all), columns = ["Common_species"]
)

common_species_across_models.to_csv("./Results/Common_species.csv", index = False)
common_species_across_models


# In[48]:


# # Pairwise overlaped
# overlap_dict = {}

# for i, model1 in enumerate(species_sets.keys()):
#     for j, model2 in enumerate(species_sets.keys()):
#         if i < j:
#             overlap = len(species_sets[model1] & species_sets[model2])
#             overlap_dict[f"{model1}_&_{model2}"] = overlap

# print(overlap_dict)

# # Common to all the models
# common_all = species_model2 & species_model3 & species_model4
# common_all


# In[49]:


# # add expand = TRUE,generate a new columm
# Family_names = df_species_model4["Family"].unique().tolist()
# print(len(Family_names)) 
# print(Family_names)
# group_size=[df_species_model4['Family'].value_counts().loc[i] for i in Family_names]
# group_size

# rename the columns
# df_species_model4.rename(columns = {
#     'AHEI_2010_score_eadj_scaled':"AHEI",
#     'AMED_score_eadj_scaled': "AMED",
#     "hPDI_score_eadj_scaled": "hPDI",
#     "rDII_score_eadj_scaled" : "rDII",
#     "rEDIH_score_all_eadj_scaled" : "rEDIH"
# }, inplace = True)

# df_species_model4


# In[ ]:





# # Plot the Figure

# In[50]:


df_species_model5 = df_species_model4.copy()
df_species_model5 = df_species_model5.drop(columns = ["Outcome"]).set_index("Species")
df_species_model5


# In[ ]:


df_species_model5 = pd.concat([df_species_model5, 
                               pd.DataFrame(0, 
                                            index=['']*int(df_species_model5.shape[0]*0.25), 
                                            columns=df_species_model5.columns)], 
                              axis=0, 
                              sort=False)

df_species_model5


# In[ ]:


Family_names = df_species_model5["Family"].unique().tolist()
print(len(Family_names)) 
print(Family_names)
group_size=[df_species_model5['Family'].value_counts().loc[i] for i in Family_names]
group_size


# In[ ]:


cmaps = {
    # "AHEI" : [plt.cm.RdGy],
    # "AMED" : [plt.cm.Greens],
    # "hPDI" : [plt.cm.Reds],
    # "rDII" : [plt.cm.Blues],
    # "rEDIH" : [plt.cm.Purples]
    "AHEI" : [cm_negative, cm_positive],
    "AMED" : [cm_negative, cm_positive],
    "hPDI" : [cm_negative, cm_positive],
    "rDII" : [cm_negative, cm_positive],
    "rEDIH" : [cm_negative, cm_positive]
}

boundries = {k:(-0.25,0.25) for k in cmaps}

label_fontsize=20
tick_fontsize=15


# In[ ]:


cmaps["AHEI"]


# In[ ]:


boundries


# In[ ]:


df_species_model5.columns


# In[ ]:


fig, circ_ax = plt.subplots(figsize=(18, 18))
ax = circ_ax

radius = 1.2
radius_step = 0.1

for layer in ['AHEI', 'AMED', 'hPDI', 'rDII', 'rEDIH']:
    if layer == 'AHEI':
        labels = df_species_model5[layer].index
    else:
        labels = ['' for i in df_species_model5.index]
        
    mypie2, texts = ax.pie([1 for i in range(df_species_model5.shape[0])], 
                           radius=radius, 
                           labels=labels, rotatelabels=True,
                           labeldistance=1., 
                           textprops={'fontsize': 9},
                           colors=_get_scale_colors(cmaps[layer], df_species_model5[layer], 
                                                    boundries=boundries[layer]), startangle=90)
    plt.setp(mypie2, width=radius_step, edgecolor='white')
    radius -= radius_step

mypie, _ = ax.pie(group_size, radius=radius,
                  colors=sns.color_palette('Paired', len(Family_names))[:-1] + [(1, 1, 1)],
                  textprops=dict(color='black', fontsize=50), startangle=90)
plt.setp(mypie, width=0.05, edgecolor='white')

legend = ax.legend(mypie, [s for s in Family_names][:-1], #super_pathway_names,
                                  title="Gut Family",
                                  loc="center",
                                  bbox_to_anchor=(0., 0, 1, 1.),
                                  fontsize=tick_fontsize, frameon=False)
legend.get_title().set_fontsize(str(label_fontsize))


# In[ ]:


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# In[ ]:


tick_fontsize = 15
label_fontsize = 15

import matplotlib as mpl
# fig, axes = plt.subplots(1, 1, figsize=(2, 5))
fig, ax = plt.subplots(1, 1, figsize=(4, 0.5))

# ax = axes[0]
norm1 = MidpointNormalize(vmin=df_species_model5['AHEI'].min(), 
                          vmax=df_species_model5['AHEI'].max(), 
                          midpoint=0)
cmap1 = _get_scale_colors([cm_negative, cm_positive], 
                          df_species_model5['AHEI'], return_cmap=True, boundries=[-0.25, 0.25])
cb1 = mpl.colorbar.ColorbarBase(ax, cmap = cmap1,
                                norm = norm1, 
                                ticks=[-0.25, 0, 0.25], 
                                orientation='horizontal')
# cb1.set_label('Healthy-ACS ($log_{10}$ pvalue)', fontsize=label_fontsize)
cb1.ax.tick_params(labelsize=tick_fontsize)
cb1.ax.yaxis.label.set_size(label_fontsize)

cb1.ax.set_xticklabels([-0.25, 0, 0.25]) 
# ax.text(df['Healthy-ACS'].max() * 1.1, 0.1, 'Non ACS vs ACS -$log_{10}$(p)', fontsize=label_fontsize+5)


# ## Unified Color scaele

# In[ ]:


# add color scale
from matplotlib.cm import ScalarMappable

fig, circ_ax = plt.subplots(figsize=(18, 18))
ax = circ_ax

radius = 1.2
radius_step = 0.1

all_values = []
for layer in ['AHEI', 'AMED', 'hPDI', 'rDII', 'rEDIH']:
    all_values.extend(df_species_model5[layer].dropna().values)

# Define common scale
common_vmin = min(all_values)
common_vmax = max(all_values)
# common_cmap = cmaps["AHEI"]
common_cmap = cmaps["AHEI"]
# common_boundaries =  boundries["AHEI"]

for layer in ['AHEI', 'AMED', 'hPDI', 'rDII', 'rEDIH']:
    if layer == 'AHEI':
        labels = df_species_model5[layer].index
    else:
        labels = ['' for i in df_species_model5.index]

    # Normalize values to common range
    norm = mcolors.Normalize(vmin=common_vmin, vmax=common_vmax)
    colors = [common_cmap(norm(val)) for val in df_species_model5[layer]]
        
    mypie2, texts = ax.pie([1 for i in range(df_species_model5.shape[0])], 
                           radius=radius, 
                           labels=labels, 
                           rotatelabels=True,
                           labeldistance=1., 
                           textprops={'fontsize': 9},
                           colors=_get_scale_colors(cmaps[layer], df_species_model5[layer], 
                                                    boundries=boundries[layer]), startangle=90)
    plt.setp(mypie2, width=radius_step, edgecolor='white')
    radius -= radius_step

mypie, _ = ax.pie(group_size, radius=radius,
                  colors=sns.color_palette('Paired', len(Family_names))[:-1] + [(1, 1, 1)],
                  textprops=dict(color='black', fontsize=50), startangle=90)
plt.setp(mypie, width=0.05, edgecolor='white')

legend = ax.legend(mypie, [s for s in Family_names][:-1], #super_pathway_names,
                                  title="Gut Family",
                                  loc="center",
                                  bbox_to_anchor=(0., 0, 1, 1.),
                                  fontsize=tick_fontsize, frameon=False)
legend.get_title().set_fontsize(str(label_fontsize))

# Add single colorbar for all layers
fig = ax.get_figure()

# Create a ScalarMappable object with the common scale
norm = mcolors.Normalize(vmin=common_vmin, vmax=common_vmax)
sm = ScalarMappable(norm=norm, cmap=common_cmap)

# Position for the single colorbar
colorbar_height = 0.03
bottom_margin = 0.05

# Create colorbar axes (horizontal at bottom)
cbar_ax = fig.add_axes([0.1, bottom_margin, 0.1, colorbar_height])

# Create the colorbar
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
# Customize this label
cbar.set_label('Value Scale', fontsize=label_fontsize)  
cbar.ax.tick_params(labelsize=tick_fontsize)



# ============================================================
Source: 13_Correlation_Functions.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# # Correlation functions
# 
# Author: Keyong Deng, Leiden University Medical Center

# In[22]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
# from utils.data_processing import process_string

def target_col_corr(
    data, target, ignore_cols = [], corr_threshold = 0.1, method = "spearman"
):
    """
    paramters: data, pandas.DataFrame; target: target column name, ignore_cols: list, columns to exclude from analysis
    corr_threshold: float, minimum correlation
    """
    cor_res= (
        data[[i for i in data.columns if i not in ignore_cols] + [target]]
        .corrwith(data[target], method = method, numeric_only = True)
        .sort_values(ascending = False)
    )
    cor_res = cor_res[cor_res.abs() > corr_threshold].drop(target)
    return cor_res

# Gut diversity correlation heatmap
def process_correlation(
    meta, target_col, ignore_cols, corr_threshold
):
    """Format the result"""
    return(
        target_col_corr(
            meta, target_col, ignore_cols = ignore_cols, corr_threshold=corr_threshold
        )
            .to_frame(name = target_col).reset_index(names = ["Features"])   
    )


# In[23]:


# # Define columns for spearman correlation analysis
# cols_for_corr_spearman = []


# In[24]:


# # Calculate the correlations for each targeted column
# all_corr = pd.DataFrame()

# for col in cols_for_corr_spearman:
#     corr_res = target_col_corr()
#     corr_res = corr_res.to_frame(name = "spearman").reset_index(names = ["Features"])
#     corr_res["target"] = col
#     all_corr = pd.concat([all_corr, corr_res])


# # Explained variance

# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams["font.family"] = "Arial"


# In[26]:


def func_explained_var(data, factor_cols, target, label = None):
    
    data = data[data[factor_cols[0]].notna()]
    
    # Initialize and fit the model
    model = LinearRegression()
    model.fit(data[factor_cols], data[target])

    # Predict the outcome
    y_pred = model.predict(data[factor_cols])

    # Calculate the variance explained (R2)
    r_squared = round(r2_score(data[target], y_pred), 4)

    if label:
        print(f"{label} Explained Variance (R2):{r_squared}")
    else:
        print(f"Explained Variance (R2):{r_squared}")
    return r_squared


# In[27]:


# def func_explained_var(data, factor_cols, target, label = None):
    
#     data = data[data[factor_cols].notna().any(axis = 1)]
#     # encode the categorical variable
#     data_encoded = pd.get_dummies(data[factor_cols], drop_first = True)
#     # fit the linear regression model
#     model = LinearRegression()
#     model.fit(data_encoded, data[target])
#     # predict the outcome
#     y_pred = model.predict(data_encoded)
#     # calculate the varianece explained
#     r_squared = round(r2_score(data[target], y_pred), 2)

#     if label:
#         print(f"{label} Explained Variance (R2):{r_squared}")
#     else:
#         print(f"Explained Variance (R2):{r_squared}")
#     return r_squared


# In[28]:


# read in the data
df = pd.read_csv("./Data/Data_species_analysis_20251023_scaled.csv")


# In[29]:


df["shannon_skbio"].isnull().sum()


# In[30]:


df["shannon_skbio"].head()


# In[31]:


Person_var = [
    "age", "sex_all", "bmi", "smoking_status", 'edu_status', 'sleep_hours_daily', 'MET_hour'
]

Diet_var = [
    'AHEI_2010_score_eadj', 
    'hPDI_score_eadj', 
    'rDII_score_eadj', 
    'AMED_score_eadj', 
    'rEDIH_score_all_eadj'
]

target = ['shannon_skbio']


# In[32]:


df_select = df[Person_var + Diet_var + target]
df_select = df_select.dropna()
df_select.shape


# In[33]:


# Create dunnies variables
df_select_dummy = pd.get_dummies(df_select, columns = ["smoking_status", "edu_status"],
                                drop_first= True,
                                dtype = int)

df_select_dummy


# In[34]:


df_select_dummy["edu_status_Low"].value_counts()


# In[35]:


df_select_dummy.describe()


# In[36]:


Person_var = [
    "age", "sex_all", "bmi",
    "smoking_status_Former", "smoking_status_Never",
    'edu_status_Low', 'sleep_hours_daily', 'MET_hour'
]

Diet_var = [
    'AHEI_2010_score_eadj', 
    'hPDI_score_eadj', 
    'rDII_score_eadj', 
    'AMED_score_eadj', 
    'rEDIH_score_all_eadj'
]

target = ['shannon_skbio']


# In[37]:


person_r2 = func_explained_var(df_select_dummy, Person_var, target="shannon_skbio", label = "Person:")
person_r2


# In[38]:


AHEI_r2 = func_explained_var(df_select_dummy, [Diet_var[0]], target="shannon_skbio", label = "Person:")
print(AHEI_r2)

hPDI_r2 = func_explained_var(df_select_dummy, [Diet_var[1]], target="shannon_skbio", label = "Person:")
print(hPDI_r2)

rDII_r2 = func_explained_var(df_select_dummy, [Diet_var[2]], target="shannon_skbio", label = "Person:")
print(rDII_r2)

AMED_r2 = func_explained_var(df_select_dummy, [Diet_var[3]], target="shannon_skbio", label = "Person:")
print(AMED_r2)

rEDIH_r2 = func_explained_var(df_select_dummy, [Diet_var[4]], target="shannon_skbio", label = "Person:")
print(rEDIH_r2)


# In[39]:


All_r2 = func_explained_var(df_select_dummy, Diet_var + Person_var, target="shannon_skbio", label = "Person:")
All_r2


# In[40]:


# Combine all the results into one dataframe
results_dict = {
    "Variable": ["AHEI", "hPDI", "rDII", "AMED", "rEDIH", "ALL"],
    "R2": [AHEI_r2, hPDI_r2, rDII_r2, AMED_r2, rEDIH_r2, All_r2]
}

df_results = pd.DataFrame(results_dict)
df_results


# In[41]:


# Plot the results

fig, ax = plt.subplots(figsize = (5,5))

# create the bar plot
bars = ax.bar(df_results["Variable"], df_results["R2"],
             color = "cornflowerblue", edgecolor = "black", linewidth = 1.2, alpha = 0.8)

# Highlight All bar in a different color
bars[-1].set_color("coral")

# Customize plot
ax.set_xlabel("Dietary Pattern", fontsize = 10, fontweight = "bold")
ax.set_ylabel("R2 value", fontsize = 10, fontweight = "bold")
ax.set_title("Model Fit R2 by dietary patterns", fontsize = 10, fontweight = "bold")
ax.grid(False)
ax.set_axisbelow(True)

# Add value to the bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height,
            f'{height:.4f}',
            ha = "center", va = "bottom", fontsize = 10, fontweight = "bold"
           )

plt.tight_layout()


# In[ ]:






# ============================================================
Source: 14_Logistic_Mediation_Analysis.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# # Logistic regression model (mediation analysis)
# Author: Keyong Deng
# LUMC, The netherlands

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# In[11]:


from Logistic_Mediation import BinaryOutcomeMediation


# In[63]:


# # read my own data
# df_mediation = pd.read_csv("./Data/data_for_mediation_exclude_reportMASLD.csv")
# df_mediation.columns


# In[64]:


# df_mediation["MAFLD_baseline_diagnosis"].value_counts()


# In[12]:


df = pd.read_csv("./Data/Data_species_analysis_20251023_scaled.csv")
df


# In[13]:


df.columns


# In[14]:


# include the MAFLD diagnosis
MASLD_incidence = pd.read_csv("../05_Medication_status/MAFLD_baseline.csv")
MASLD_incidence = MASLD_incidence[["participant_id", "mafld_diagnosed"]].dropna()
MASLD_incidence.shape


# In[15]:


df_merge = pd.merge(df, MASLD_incidence, on = "participant_id", how = "left")
df_merge


# In[68]:


# print(df_merge.columns.tolist())


# In[16]:


# mediation analysis (DPs -> Species -> Prevalent MASLD
df_merge["mafld_diagnosed"].value_counts(dropna= False)


# In[17]:


# include the age
df_merge["age"].isnull().sum()


# In[71]:


# lifestyles = pd.read_csv("../02_Lifestyle factors/lifestyle_factor_all_disease_V2.csv")
# lifestyles_age = lifestyles[["participant_id", "age_all"]]
# lifestyles_age


# In[18]:


# # merge with merge_df_MAFLD
# df_merge = pd.merge(df_merge,lifestyles_age, on="participant_id", how = "left")
df_merge["age_all"].isnull().sum()


# In[19]:


variable_select = ["sex_all", "age_all", "bmi", "edu_status", 
                   "smoking_status",  "sleep_hours_daily", 
                   "MET_hour", "Hormone_use", "Vit_use", 
                   "Antibio_use", "PPI_use", "AHEI_2010_score_eadj_scaled", "mafld_diagnosed"]


# In[20]:


df_merge_select = df_merge.dropna(subset=variable_select)
df_merge_select.shape


# In[22]:


# recode the variable
df_merge_select["Antibio_use"].value_counts()
df_merge_select["PPI_use"].value_counts()


# In[23]:


df_merge_select["edu_status"] = pd.Categorical(df_merge_select["edu_status"],
                                              categories= ["High", "Low"], ordered = True)

df_merge_select["smoking_status"] = pd.Categorical(df_merge_select["smoking_status"],
                                              categories= ["Never", "Former", "Current"], ordered = True)

df_merge_select["Hormone_use"] = pd.Categorical(df_merge_select["Hormone_use"],
                                              categories= ["Use", "not use"], ordered = True)

df_merge_select["Vit_use"] = pd.Categorical(df_merge_select["Vit_use"],
                                              categories= ["Use", "not use"], ordered = True)

df_merge_select["Antibio_use"] = pd.Categorical(df_merge_select["Antibio_use"],
                                              categories= ["Use", "not use"], ordered = True)

df_merge_select["PPI_use"] = pd.Categorical(df_merge_select["PPI_use"],
                                              categories= ["Use", "not use"], ordered = True)


# In[24]:


df_merge_select["mafld_diagnosed"] = df_merge_select["mafld_diagnosed"].map({"Yes": 1, "No": 0})
df_merge_select["mafld_diagnosed"].value_counts()


# In[25]:


# Read in the mediators
species_MASLD = pd.read_csv("./Results/Species_Prevalent_MASLD.csv")
species_MASLD

species_MASLD_sig = species_MASLD[species_MASLD["p_adj"] < 0.05]
species_MASLD_sig.shape


# In[26]:


species_MASLD_sig["Variable"][0]


# In[27]:


# Mediation: for binary outcome, Continuous Exposure and Mediator

class BinaryOutcomeMediation:
    def __init__(self, data, exposure, mediator, outcome, covariates = None):
        """
        data: dataframe
        exposure: name of exposure variables
        mediator: str: names of mediator variables
        outcoem: str: name of outcome variables
        covariate: list: additional adjusted covariates
        """
        self.data = data.copy()
        self.exposure = exposure
        self.mediator = mediator
        self.outcome = outcome
        self.covariates = covariates if covariates else []

        # verify outcome is binary
        if self.data[outcome].nunique() != 2:
            raise ValueError(f"{outcome} must be binary")

        # remove missing values
        vars_need = [exposure, mediator, outcome] + self.covariates
        self.data = self.data[vars_need].dropna()

    def fit(self):
        """
        fit two logistic regression model
        1. total effect: Y ~ X + covariates
        2. Direct effect: Y ~ X + M + covariates
        3. Mediator model: M ~ X + covaraites
        """
        # build model
        covariates_str = " + ".join(self.covariates) if self.covariates else ""

        # model1
        formula_total = f"{self.outcome} ~ {self.exposure}"
        if covariates_str:
            formula_total += f" + {covariates_str}"

        # direct model
        formula_direct = f"{self.outcome} ~ {self.exposure} + {self.mediator}"
        if covariates_str:
            formula_direct += f" + {covariates_str}"

        # mediator model
        formula_mediator = f"{self.mediator} ~ {self.exposure}"
        if covariates_str:
            formula_mediator += f" + {covariates_str}"

        # fit the model
        # disp = 0 indicate not print the converage information 
        # use smf.logit or smf.ols , need to dropna first
        self.model_total = smf.logit(formula_total, data = self.data).fit(disp = 0)
        self.model_direct = smf.logit(formula_direct, data = self.data).fit(disp = 0)
        self.model_mediator = smf.ols(formula_mediator, data = self.data).fit()

        return self

    def calculate_effect(self):
        """
        - Natural indirect effect (NIE)
        - Natural direct effect
        - Total effect = indirect + direct
        - Proportion mediated = NIE / TE
        """
        # Extract coefficients
        a = self.model_mediator.params[self.exposure] # x -> M
        b = self.model_direct.params[self.mediator] # M -> Y (direct model)
        c = self.model_total.params[self.exposure] # x -> Y (total model)
        c_prime = self.model_direct.params[self.exposure] # x -> Y | M

        # for binary outcome, use odds ratio
        nde = np.exp(c_prime)

        # indirect effect = exp(a*b) 
        nie = np.exp(a * b)

        # total effect
        te = np.exp(c)

        # proportion mediated
        if c != 0:
            prop_mediated = (c - c_prime)/c
        else:
            prop_mediated =  np.nan

        Indirect_effect = c - c_prime

        # Calculate the pvalue
        se_c = self.model_total.bse[self.exposure]
        se_c_prime = self.model_direct.bse[self.exposure]

        # For independent effects
        se_indirect = np.sqrt(se_c**2 + se_c_prime**2)

        # Test if indirect effect is different from 0
        if se_indirect > 0:
            z_indirect = Indirect_effect / se_indirect
            p_indirect = 2 * (1-stats.norm.cdf(np.abs(z_indirect)))
        else:
            p_indirect = np.nan

        p_prop_mediated = p_indirect

        # results 
        results = {
            "a_coefficient" : a,
            "b_coefficient" : b,
            "c_coefficient" : c,
            "c_prime_coefficient" : c_prime,
            "Natural_Direct_Effect_OR" : nde,
            "Natural_Indirect_Effect_OR" : nie,
            "Total_Effect_OR": te,
            "Indirect_Effect" : Indirect_effect,
            "SE_indirect_Effect" : se_indirect,
            "Z_indirect_Effect" : z_indirect,
            "p_indirect_Effect" : p_indirect,
            "Proportion_Mediated": prop_mediated,
            "Proportion_Mediated_pct": prop_mediated * 100 if not np.isnan(prop_mediated) else np.nan,
            "p_Proportion_Mediated": p_prop_mediated
        }

        self.effects = results

        return results

    # bootstrapping ci
    def bootstrap_ci(self, n_bootstrap = 1000, ci = 95):
        """
        calculate the bootstrap confidence interval
        """
        indirect_effect = []
        nde_effect = []
        nie_effect = []
        prop_mediated_list = []

        np.random.seed(42)

        for i in range(n_bootstrap):
            if (i + 1) % 200 == 0:
                print(f' Complete {i + 1}/{n_bootstrap}')
            # resampling without replacement
            boot_data = self.data.sample(n = len(self.data), replace = True)
            try:
                covariates_str = " + ".join(self.covariates) if self.covariates else ""
                formula_total = f"{self.outcome} ~ {self.exposure}"
                if covariates_str:
                    formula_total += f" + {covariates_str}"
                # direct model
                formula_direct = f"{self.outcome} ~ {self.exposure} + {self.mediator}"
                if covariates_str:
                    formula_direct += f" + {covariates_str}"
                # mediator model
                formula_mediator = f"{self.mediator} ~ {self.exposure}"
                if covariates_str:
                    formula_mediator += f" + {covariates_str}"
                # fit the model
                model_boot_total = smf.logit(formula_total, data = boot_data).fit(disp = 0)
                model_boot_direct = smf.logit(formula_direct, data = boot_data).fit(disp = 0)
                model_boot_mediator = smf.ols(formula_mediator, data = boot_data).fit()

                a_boot = model_boot_mediator.params[self.exposure]
                
                b_boot = model_boot_direct.params[self.mediator]
                
                c_boot = model_boot_total.params[self.exposure]
                
                c_prime_boot = model_boot_direct.params[self.exposure]

                # store effects
                indirect_effect.append(c_boot - c_prime_boot)
                nde_effect.append(np.exp(c_prime_boot))
                nie_effect.append(np.exp(a_boot * b_boot))

                # mediation proportion
                if c_boot != 0:
                    prop_med = (c_boot - c_prime_boot)/c_boot
                    prop_mediated_list.append(prop_med)
            except:
                continue

        # Calculate the CI
        alpha = (100 - ci) /2

        # ci
        indirect_ci = np.percentile(indirect_effect, [alpha, 100-alpha])
        nde_ci = np.percentile(nde_effect, [alpha, 100-alpha])
        nie_ci = np.percentile(nie_effect, [alpha, 100-alpha])

        indirect_pval = 2 * min(
            np.mean(np.array(indirect_effect) <= 0),
            np.mean(np.array(indirect_effect) >= 0)
        )

        indirect_pval = round(indirect_pval, 8)
        
        prop_med_ci = np.percentile(prop_mediated_list, [alpha, 100-alpha])

        # self_boot results 
        self.boot_results = {
            "indirect_effect_ci": indirect_ci,
            "nde_ci" : nde_ci,
            "nie_ci" : nie_ci,
            "indirect_pval": indirect_pval,
            "prop_mediation_ci" : prop_med_ci,
            "n_successful": len(indirect_ci)
        }

        return self.boot_results

# # Example

# if __name__ == "__main__":

#     # column names
#     exposure_vars = ""
#     mediator_vars = ""
#     outcome_vars = ""
#     covariates = []

#     # Initiate and run the analysis
#     med_analysis = BinaryOutcomeMediation(
#         data = df,
#         exposure = exposure_vars,
#         mediator = mediator_vars,
#         outcome = outcome_vars,
#         covariates = covariates
#     )

#     # fit model
#     med_analysis.fit()
#     med_analysis.calculate_effect()
#     med_analysis.bootstrap_ci(n_bootstrap=1000, ci=95)


# In[28]:


df_merge_select.columns = df_merge_select.columns.str.replace("|", "_", regex = False)


# In[29]:


df_merge_select.shape


# In[42]:


med = BinaryOutcomeMediation(
    data = df_merge_select,
    exposure = "rEDIH_score_all_eadj_scaled",
    mediator = species_MASLD_sig["Variable"][0],
    outcome = "mafld_diagnosed",
    covariates = ["sex_all", "age_all", "edu_status", 
                   "smoking_status",  "sleep_hours_daily", 
                   "MET_hour", "Hormone_use", "Vit_use"]
)


# In[43]:


df_merge_select["mafld_diagnosed"].value_counts()


# In[44]:


med.fit()


# In[45]:


med.calculate_effect()


# In[35]:


import statsmodels.api as sm

def multiple_glm_model (data, dependent_var, 
                        covariate_sets, 
                        family = sm.families.Binomial()):
    models = {}
    # covariate_sets (is a dictionary)

    for model_name, covariates in covariate_sets.items():
        
        formula = f"{dependent_var} ~ {'+'.join(covariates)}"
        
        print(f"Fitting {model_name} with formula: {formula}")

        # fit the model (remove missing)
        model = sm.GLM.from_formula(formula, family = family, data = data, missing = "drop")

        # results
        result = model.fit()

        # store the results
        # models[model_name] = result.summary()
        models[model_name] = result
        
    return models


# In[41]:


temp2 =  multiple_glm_model(data = df_merge_select, 
                   dependent_var = "mafld_diagnosed",
                    covariate_sets = {"model2": ["AHEI_2010_score_eadj_scaled","sex_all", "age_all", "bmi", "edu_status", 
                   "smoking_status",  "sleep_hours_daily", 
                   "MET_hour", "Hormone_use", "Vit_use"]})
print(temp2["model2"].params)
temp2["model2"].pvalues


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[86]:


# med.bootstrap_ci(n_bootstrap= 1000, ci=95)


# In[87]:


from tqdm import tqdm
from itertools import product


# In[88]:


Exposures = ["AHEI_2010_score_eadj_scaled" #, 
             #"hPDI_score_eadj_scaled", 
             #"rDII_score_eadj_scaled", 
             #"AMED_score_eadj_scaled", 
             #"rEDIH_score_all_eadj_scaled"
            ]
             

# list all the mediators
mediators = species_MASLD_sig["Variable"].tolist()

# Select only first two mediators
# mediators = mediators[:2]

# combinations
combinations = list(product(Exposures, mediators))
combinations


# In[89]:


results_list = []

# loop through each combination
for exposure, mediator in tqdm(combinations, desc = "Running mediation analysis"):
    try:
        med = BinaryOutcomeMediation(
            data = df_merge_select,
            exposure = exposure,
            mediator = mediator,
            outcome = "mafld_diagnosed",
            covariates = ["sex_all", "age_all", "bmi", "edu_status", 
                          "smoking_status",  "sleep_hours_daily", 
                         "MET_hour", "Hormone_use", "Vit_use", 
                         "Antibio_use", "PPI_use"])
        # fit the model
        med.fit()
        # calculate the effects
        effects = med.calculate_effect()
        # Bootstrap cis
        ci_results = med.bootstrap_ci(n_bootstrap=1000, ci = 95)
        
        # Combine the results
        results = {
            "Exposure": exposure,
            "Mediator": mediator,
            # Effect estimates
            "a_coefficient" : effects["a_coefficient"],
            "b_coefficient" : effects["b_coefficient"],
            "c_coefficient" : effects["c_coefficient"],
            "c_prime_coefficient" : effects["c_prime_coefficient"],
            "Natural_Direct_Effect_OR" : effects["Natural_Direct_Effect_OR"],
            "Natural_Indirect_Effect_OR" : effects["Natural_Indirect_Effect_OR"],
            "Total_Effect_OR": effects["Total_Effect_OR"],
            "Indirect_Effect" : effects["Indirect_Effect"],
            "Proportion_Mediated": effects["Proportion_Mediated"],
            # Confidence interval
            "indirect_effect_ci_Lower": ci_results["indirect_effect_ci"][0],
            "indirect_effect_ci_Uppper": ci_results["indirect_effect_ci"][1],
            "nde_ci_Lower": ci_results["nde_ci"][0],
            "nde_ci_Uppper": ci_results["nde_ci"][1],
            "nie_ci_Lower": ci_results["nie_ci"][0],
            "nie_ci_Upper": ci_results["nie_ci"][1],
            "prop_mediation_ci_Lower":ci_results["prop_mediation_ci"][0],
            "prop_mediation_ci_Uppper": ci_results["prop_mediation_ci"][1],
            "prop_mediation_pvalue" : ci_results["indirect_pval"],
            "status" : "success"
        }

        results_list.append(results)

    except Exception as e:
        # Log failed 
        results_list.append({
            "Exposure": exposure,
            "Mediator": mediator,
            "status": "Failed"
        })


# In[90]:


results_df_AHEI = pd.DataFrame(results_list)
results_df_AHEI


# In[92]:


results_df_AHEI.to_csv("./Results/Mediation_Results_AHEI_MASLD.csv", index = False)


# In[99]:


Nutrients = pd.read_csv("Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])

df_merge_select = pd.merge(df_merge_select, Nutrients_2, on = "participant_id", how = 'left')
df_merge_select


# In[101]:


# df_merge_select["alcohol_g"].describe()


# In[93]:


Exposures = [
            "hPDI_score_eadj_scaled"# , 
             #"rDII_score_eadj_scaled", 
             #"AMED_score_eadj_scaled", 
             #"rEDIH_score_all_eadj_scaled"
            ]
             

# list all the mediators
mediators = species_MASLD_sig["Variable"].tolist()

# Select only first two mediators
# mediators = mediators[:2]

# combinations
combinations = list(product(Exposures, mediators))
combinations


# In[102]:


results_list = []

# loop through each combination
for exposure, mediator in tqdm(combinations, desc = "Running mediation analysis"):
    try:
        med = BinaryOutcomeMediation(
            data = df_merge_select,
            exposure = exposure,
            mediator = mediator,
            outcome = "mafld_diagnosed",
            covariates = ["sex_all", "age_all", "bmi", "edu_status", 
                          "smoking_status",  "sleep_hours_daily", 
                         "MET_hour", "Hormone_use", "Vit_use", 
                         "Antibio_use", "PPI_use", "alcohol_g"])
        # fit the model
        med.fit()
        # calculate the effects
        effects = med.calculate_effect()
        # Bootstrap cis
        ci_results = med.bootstrap_ci(n_bootstrap=1000, ci = 95)
        
        # Combine the results
        results = {
            "Exposure": exposure,
            "Mediator": mediator,
            # Effect estimates
            "a_coefficient" : effects["a_coefficient"],
            "b_coefficient" : effects["b_coefficient"],
            "c_coefficient" : effects["c_coefficient"],
            "c_prime_coefficient" : effects["c_prime_coefficient"],
            "Natural_Direct_Effect_OR" : effects["Natural_Direct_Effect_OR"],
            "Natural_Indirect_Effect_OR" : effects["Natural_Indirect_Effect_OR"],
            "Total_Effect_OR": effects["Total_Effect_OR"],
            "Indirect_Effect" : effects["Indirect_Effect"],
            "Proportion_Mediated": effects["Proportion_Mediated"],
            # Confidence interval
            "indirect_effect_ci_Lower": ci_results["indirect_effect_ci"][0],
            "indirect_effect_ci_Uppper": ci_results["indirect_effect_ci"][1],
            "nde_ci_Lower": ci_results["nde_ci"][0],
            "nde_ci_Uppper": ci_results["nde_ci"][1],
            "nie_ci_Lower": ci_results["nie_ci"][0],
            "nie_ci_Upper": ci_results["nie_ci"][1],
            "prop_mediation_ci_Lower":ci_results["prop_mediation_ci"][0],
            "prop_mediation_ci_Uppper": ci_results["prop_mediation_ci"][1],
            "prop_mediation_pvalue" : ci_results["indirect_pval"],
            "status" : "success"
        }

        results_list.append(results)

    except Exception as e:
        # Log failed 
        results_list.append({
            "Exposure": exposure,
            "Mediator": mediator,
            "status": "Failed"
        })


# In[103]:


results_df_hPDI = pd.DataFrame(results_list)
results_df_hPDI


# In[104]:


results_df_hPDI.to_csv("./Results/Mediation_Results_hPDI_MASLD.csv", index = False)


# # rDII

# In[ ]:


Exposures = [
             "rDII_score_eadj_scaled"# , 
             #"AMED_score_eadj_scaled", 
             #"rEDIH_score_all_eadj_scaled"
            ]
             

# list all the mediators
mediators = species_MASLD_sig["Variable"].tolist()

# Select only first two mediators
# mediators = mediators[:2]

# combinations
combinations = list(product(Exposures, mediators))

results_list = []

# loop through each combination
for exposure, mediator in tqdm(combinations, desc = "Running mediation analysis"):
    try:
        med = BinaryOutcomeMediation(
            data = df_merge_select,
            exposure = exposure,
            mediator = mediator,
            outcome = "mafld_diagnosed",
            covariates = ["sex_all", "age_all", "bmi", "edu_status", 
                          "smoking_status",  "sleep_hours_daily", 
                         "MET_hour", "Hormone_use", "Vit_use", 
                         "Antibio_use", "PPI_use"])
        # fit the model
        med.fit()
        # calculate the effects
        effects = med.calculate_effect()
        # Bootstrap cis
        ci_results = med.bootstrap_ci(n_bootstrap=1000, ci = 95)
        
        # Combine the results
        results = {
            "Exposure": exposure,
            "Mediator": mediator,
            # Effect estimates
            "a_coefficient" : effects["a_coefficient"],
            "b_coefficient" : effects["b_coefficient"],
            "c_coefficient" : effects["c_coefficient"],
            "c_prime_coefficient" : effects["c_prime_coefficient"],
            "Natural_Direct_Effect_OR" : effects["Natural_Direct_Effect_OR"],
            "Natural_Indirect_Effect_OR" : effects["Natural_Indirect_Effect_OR"],
            "Total_Effect_OR": effects["Total_Effect_OR"],
            "Indirect_Effect" : effects["Indirect_Effect"],
            "Proportion_Mediated": effects["Proportion_Mediated"],
            # Confidence interval
            "indirect_effect_ci_Lower": ci_results["indirect_effect_ci"][0],
            "indirect_effect_ci_Uppper": ci_results["indirect_effect_ci"][1],
            "nde_ci_Lower": ci_results["nde_ci"][0],
            "nde_ci_Uppper": ci_results["nde_ci"][1],
            "nie_ci_Lower": ci_results["nie_ci"][0],
            "nie_ci_Upper": ci_results["nie_ci"][1],
            "prop_mediation_ci_Lower":ci_results["prop_mediation_ci"][0],
            "prop_mediation_ci_Uppper": ci_results["prop_mediation_ci"][1],
            "prop_mediation_pvalue" : ci_results["indirect_pval"],
            "status" : "success"
        }

        results_list.append(results)

    except Exception as e:
        # Log failed 
        results_list.append({
            "Exposure": exposure,
            "Mediator": mediator,
            "status": "Failed"
        })

results_df_rDII = pd.DataFrame(results_list)
results_df_rDII

# In[91]:


# Convert to pandas dataframe
# results_df = pd.DataFrame(results_list)

# # For each exposure, save separately
# for exposure in exposures:
#     exposure_df = results_df[results_df["Exposure"] == exposure]
#     # clean filename 
#     clean_name = exposure.replace("/", "_").replace(" ", "_")
#     exposure_df.to_csv(f'./Results/mediation_results_{clean_name}.csv', index = False)


# # Read in the data

# In[2]:


import pandas as pd


# In[3]:


AMED_mediation = pd.read_csv("./Results/Mediation_Results_AMED_MASLD.csv")
AMED_mediation.columns


# In[5]:


AMED_mediation= AMED_mediation[['Exposure', 'Mediator', 'a_coefficient', 'b_coefficient', 'c_coefficient', "Proportion_Mediated", 
                                "prop_mediation_ci_Lower", "prop_mediation_ci_Uppper"]]
AMED_mediation.sort_values(by = "Proportion_Mediated")


# In[6]:


AHEI_mediation = pd.read_csv("./Results/Mediation_Results_AHEI_MASLD.csv")
AHEI_mediation= AHEI_mediation[['Exposure', 'Mediator', 'a_coefficient', 'b_coefficient', 'c_coefficient', "Proportion_Mediated", 
                                "prop_mediation_ci_Lower", "prop_mediation_ci_Uppper"]]
AHEI_mediation.sort_values(by = "Proportion_Mediated")


# In[7]:


hPDI_mediation = pd.read_csv("./Results/Mediation_Results_hPDI_MASLD.csv")
hPDI_mediation= hPDI_mediation[['Exposure', 'Mediator', 'a_coefficient', 'b_coefficient', 'c_coefficient', "Proportion_Mediated", 
                                "prop_mediation_ci_Lower", "prop_mediation_ci_Uppper"]]
hPDI_mediation.sort_values(by = "Proportion_Mediated")


# In[8]:


rDII_mediation = pd.read_csv("./Results/Mediation_Results_rDII_MASLD.csv")
rDII_mediation= rDII_mediation[['Exposure', 'Mediator', 'a_coefficient', 'b_coefficient', 'c_coefficient', "Proportion_Mediated", 
                                "prop_mediation_ci_Lower", "prop_mediation_ci_Uppper"]]
rDII_mediation.sort_values(by = "Proportion_Mediated")


# In[9]:


rEDIH_mediation = pd.read_csv("./Results/Mediation_Results_rEDIH_MASLD.csv")
rEDIH_mediation= rEDIH_mediation[['Exposure', 'Mediator', 'a_coefficient', 'b_coefficient', 'c_coefficient', "Proportion_Mediated", 
                                "prop_mediation_ci_Lower", "prop_mediation_ci_Uppper"]]
rEDIH_mediation.sort_values(by = "Proportion_Mediated")


# In[ ]:






# ============================================================
Source: 15_Combine_results.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# # Combine data frame to export
# 
# Author: keyong deng

# In[1]:


import pandas as pd


# In[4]:


# The mediation analysis for overlapped microbes
AHEI_mediation = pd.read_csv("./Results/AHEI_mediation_results.csv")
AMED_mediation = pd.read_csv("./Results/AMED_mediation_results_overlappedGut.csv")
rDII_mediation = pd.read_csv("./Results/rDII_mediation_results_overlappedGut.csv")
rEDIH_mediation = pd.read_csv("./Results/rEDIH_mediation_results_overlappedGut.csv")
hPDI_mediation = pd.read_csv("./Results/hPDI_mediation_1.csv")


# In[5]:


# combine
DP_mediation = pd.concat((AHEI_mediation,AMED_mediation,
                          rDII_mediation, rEDIH_mediation,
                          hPDI_mediation
                         ), axis = 0)
DP_mediation.shape


# In[6]:


DP_mediation.head()


# In[7]:


DP_mediation.to_csv("./Results/DP_mediation_results.csv", index = False)



# ============================================================
Source: A0_helpfunction.py
# ============================================================

#### I will write several function for using in this .py files
#### Author: Keyong Deng, LUMC
#### Leiden University Medical Center


# Make the MLR model for DPs separately, adjusted for age, bmi, and sex
## Write a function to run the multiple linear regression analysis\

import statsmodels.api as sm
import numpy as np
import pandas as pd

def run_multiple_regressions(df, predictors, outcomes, control_vars = None):
    
    results_list = []
    for var in predictors:
        
            # Generate X and Y
        if control_vars is not None:
            
            df1 = df[[var] + control_vars + outcomes]
            df1 = df1.dropna()
            X = df1[[var] + control_vars]
            X = sm.add_constant(X)
            y = df1[outcomes]
            # fit the model
            model = sm.OLS(y, X).fit()
            
            # extract the results for the variable of interest
            coef = model.params[var] 
            ci_lower, ci_upper = model.conf_int().loc[var]
            pvalue = model.pvalues[var]
            
            # Store results
            results_list.append({
                'N': df1.shape[0],
                'Variable': var,
                'Outcome': ''.join(outcomes),
                'Coefficient': coef,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'P_Value': pvalue
        })
        else:
            df1 = df[[var] + outcomes]
            df1 = df1.dropna()
            X = df1[[var]]
            X = sm.add_constant(X)
            y = df1[outcomes]
            # fit the model
            model = sm.OLS(y, X).fit()
            
            # extract the results for the variable of interest
            coef = model.params[var] 
            ci_lower, ci_upper = model.conf_int().loc[var]
            pvalue = model.pvalues[var]
            
            # Store results
            results_list.append({
                'N': df1.shape[0],
                'Variable': var,
                'Outcome': ''.join(outcomes),
                'Coefficient': coef,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'P_Value': pvalue
        })
       
    results_df = pd.DataFrame(results_list)
    
    # set three decimals
    results_df['CI'] = results_df.apply(lambda row: f"({row['CI_Lower']:.3f}, {row['CI_Upper']:.3f})", axis=1)
    return results_df

## Function to run the GLM based on the formula ----------
def run_regression_formula(df, predictors, outcomes, control_vars = None):

    import statsmodels.formula.api as smf
    
    results_list = []

    # check predictors if is a list
    if isinstance(predictors, str):
        predictors = [predictors]
    
    for var in predictors:
        
        # generate the formula string for the current predictor
        if control_vars is not None:
            formula = f'{outcomes} ~ {var} + {" + ".join(control_vars)}'
        else: 
            formula = f"{outcomes} ~ {var}"

        # fit the model using the formula
        model = smf.ols(formula, data = df, missing='drop').fit()
        
        N = model.nobs
        # Extract the results for the variables of interest
        coef = model.params[var]
        ci_lower, ci_upper = model.conf_int().loc[var]
        pvalue = model.pvalues[var]

        # store the results
        results_list.append({
            "N" : N,
            "Variable" : var,
            "Outcome" : outcomes,
            "Coefficient" : coef,
            "CI_lower" : ci_lower,
            "CI_upper" : ci_upper,
            "P_value" : pvalue      
        })
        return results_list


# Function to Rank-based inverse normal transformation on pandas series. 
def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return ss.norm.ppf(x)

def rank_INT(series, c = 3.0/8, stochastic=True):
    
    """ Perform rank-based inverse normal transformation on pandas series.
        If stochastic is True ties are given rank randomly, otherwise ties will
        share the same value. NaN values are ignored.

        Args:
            param1 (pandas.Series):   Series of values to transform
            param2 (Optional[float]):  Constand parameter (Bloms constant)
            param3 (Optional[bool]):  Whether to randomise rank of ties
        
        Returns:
            pandas.Series
    """

    # Check input
    assert(isinstance(series, pd.Series))
    assert(isinstance(c, float))
    assert(isinstance(stochastic, bool))

    # Set seed
    np.random.seed(123)

    # Take original series indexes
    orig_idx = series.index

    # Drop NaNs
    series = series.loc[~ pd.isnull(series)]

    # Get ranks
    if stochastic == True:
        # Shuffle by index
        series = series.loc[np.random.permutation(series.index)]
        # Get rank, ties are determined by their position in the series (hence
        # why we randomised the series)
        rank = ss.rankdata(series, method="ordinal")
    else:
        # Get rank, ties are averaged
        rank = ss.rankdata(series, method="average")

    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)

    # Convert rank to normal distribution
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
    
    return transformed[orig_idx]


## Function to calculate the missingness of dataframe
def calculate_missingness(df):
    # Total number of rows
    total_rows = len(df)
    
    # Calculate missing values count
    missing_count = df.isna().sum()
    
    # Calculate missing values percentage
    missing_percentage = (missing_count / total_rows) * 100
    
    # Create a summary DataFrame
    missingness_summary = pd.DataFrame({
        'missing_count': missing_count,
        'missing_percentage': missing_percentage
    })
    
    return missingness_summary

# Function to run and export the mediation analysis -------------------
def mediation_gut(data, exposure, controls, dependent_var, mediators):
    
    from statsmodels.stats.mediation import Mediation
    
    result_df = pd.DataFrame(columns = ["Exposure", 
                                        "Mediator", 
                                        "ACME", "ACME_Pval", 
                                        "ADE", "ADE_Pval", 
                                        "total_effect", 
                                        "Total_Pval",
                                        "prop_mediated", 
                                        "prop_mediated_pval"])
    

    controls_joined = "+".join(controls)

    # set the outcome formula
    outcome_formula = f"{dependent_var} ~ {exposure}+{controls_joined}+{mediators}"
    print(outcome_formula)
    outcome_model = sm.OLS.from_formula(outcome_formula, data, missing = "drop")

    # set the mediator formula
    mediator_formula = f"{mediators} ~ {exposure}+{controls_joined}"
    print(mediator_formula)
    mediator_model = sm.OLS.from_formula(mediator_formula, data, missing = "drop")

    # Mediation
    med = Mediation(outcome_model, mediator_model, exposure, mediators).fit(method = "bootstrap", n_rep = 500)

    med_summary = pd.DataFrame(med.summary()).reset_index()

    print(med_summary)
    
    Estimate = med_summary.loc[med_summary["index"] == "ACME (average)"]["Estimate"].values[0]
    Lower_ci = med_summary.loc[med_summary["index"] == "ACME (average)"]["Lower CI bound"].values[0]
    Upper_ci = med_summary.loc[med_summary["index"] == "ACME (average)"]["Upper CI bound"].values[0]    
    
    # comebine them into a string
    Estimate_ci = f"{Estimate:.5f}[{Lower_ci:.5f}, {Upper_ci:.5f}]"
    ACME_Pval = med_summary.loc[med_summary["index"] == "ACME (average)"]["P-value"].values[0]

    Estimate_ADE = med_summary.loc[med_summary["index"] == "ADE (average)"]["Estimate"].values[0]
    Lower_ci_ADE = med_summary.loc[med_summary["index"] == "ADE (average)"]["Lower CI bound"].values[0]
    Upper_ci_ADE = med_summary.loc[med_summary["index"] == "ADE (average)"]["Upper CI bound"].values[0]
    
    # comebine them into a string
    Estimate_ci_ADE = f"{Estimate_ADE:.5f}[{Lower_ci_ADE:.5f}, {Upper_ci_ADE:.5f}]"
    ADE_Pval = med_summary.loc[med_summary["index"] == "ADE (average)"]["P-value"].values[0]

    # Total effect 
    Estimate_total = med_summary.loc[med_summary["index"] == "Total effect"]["Estimate"].values[0]
    Lower_ci_total = med_summary.loc[med_summary["index"] == "Total effect"]["Lower CI bound"].values[0]
    Upper_ci_total = med_summary.loc[med_summary["index"] == "Total effect"]["Upper CI bound"].values[0]
    # comebine them into a string
    Estimate_ci_total = f"{Estimate_total:.5f}[{Lower_ci_total:.5f}, {Upper_ci_total:.5f}]"
    Total_Pval = med_summary.loc[med_summary["index"] == "Total effect"]["P-value"].values[0]

    # proportion mediation (Prop. mediated (average))
    proportion_mediation = med_summary.loc[med_summary["index"] == "Prop. mediated (average)"]["Estimate"].values[0]
    Lower_ci_proportion = med_summary.loc[med_summary["index"] == "Prop. mediated (average)"]["Lower CI bound"].values[0]
    Upper_ci_proportion = med_summary.loc[med_summary["index"] == "Prop. mediated (average)"]["Upper CI bound"].values[0]     
    
    # combine them into a string
    proportion_mediation_all = f"{proportion_mediation:.5f}[{Lower_ci_proportion:.5f}, {Upper_ci_proportion:.5f}]"
    # print(proportion_mediation_all)
            
    # pvalue
    proportion_mediation_Pval = med_summary.loc[med_summary["index"] == "Prop. mediated (average)"]["P-value"].values[0]
    proportion_mediation_Pval = float(f'{proportion_mediation_Pval:.8f}')
    print(proportion_mediation_Pval)

    # extract the results to dataframe
    results_df = pd.concat([
        result_df,
        pd.DataFrame({
                "Exposure": exposure,
                "Mediator": mediators,
                "ACME": Estimate_ci,
                "ACME_Pval": ACME_Pval,
                "ADE": Estimate_ci_ADE,
                "ADE_Pval": ADE_Pval,
                "total_effect": Estimate_ci_total,
                "Total_Pval": Total_Pval,
                "prop_mediated": proportion_mediation_all,
                "prop_mediated_pval": proportion_mediation_Pval
        }, index = [0])],
        ignore_index = True)

    return(results_df)

# Define a function to run the GLM for quintiles 
def run_GLM_Group(df, predictors, outcomes, control_vars = None):
    
    results_list = pd.DataFrame()

    # check predictors if is a list
    if isinstance(predictors, str):
        predictors = [predictors]

    for var in predictors:
        # generate the formula string for the current predictor
        if control_vars is not None:
            formula = f'{outcomes} ~ {var} + {" + ".join(control_vars)}'
        else: 
            formula = f"{outcomes} ~ {var}"

        # fit the model using the formula
        model = smf.ols(formula, data = df, missing='drop').fit()
        
        N = model.nobs
        DPs_run = var.split("_")[0]
        coeffs = model.params
        
        quintile_coeffs = coeffs[coeffs.index.str.contains(var)]
        p_vals = model.pvalues[model.pvalues.index.str.contains(var)]
        conf_int = model.conf_int().loc[model.conf_int().index.str.contains(var)]

        # create a new dataframe
        df_coefficients = pd.DataFrame({
                    "N": N,
                    "DPs": DPs_run,
                    "Categories": quintile_coeffs.index,
                    "Outcome": outcomes,
                    "Coefficient" : quintile_coeffs.values,
                    "P_val" : p_vals.values,
                    "ci_lower" : conf_int.iloc[:, 0].values,
                    "ci_upper" : conf_int.iloc[:, 1].values
                    })

        df_coefficients["Categories"] = ["Q2", "Q3", "Q4", "Q5"]

        # combine the reference group
        reference_row = pd.DataFrame({
                        "N": N,
                        "DPs" : DPs_run,
                        "Categories" : ["Q1"],
                        "Outcome": outcomes,
                        "Coefficient" : [0.0],
                        "P_val" : np.nan,
                        "ci_lower" : np.nan,
                        "ci_upper" : np.nan
                        })
        
        # concat two dataframe
        df_results = pd.concat([reference_row, df_coefficients], ignore_index=True)
        # results_list.append(df_results)
        results_list = pd.concat([results_list, df_results], ignore_index=True)
    return results_list
    # change it into dataframe
    # results_df = pd.DataFrame(results_list)
    # set two decimals
    # results_df['CI'] = results_df.apply(lambda row: f"({row['ci_lower']:.2f}, {row['ci_upper']:.2f})", axis=1)
    # return results_df

# Categories function to run 
def run_GLM_Group(df, predictors, outcomes, control_vars = None):

    import statsmodels.formula.api as smf
    import pandas as pd
    
    results_list = pd.DataFrame()

    # check predictors if is a list
    if isinstance(predictors, str):
        predictors = [predictors]

    for var in predictors:
        # generate the formula string for the current predictor
        if control_vars is not None:
            formula = f'{outcomes} ~ {var} + {" + ".join(control_vars)}'
        else: 
            formula = f"{outcomes} ~ {var}"

        # fit the model using the formula
        model = smf.ols(formula, data = df, missing='drop').fit()
        
        N = model.nobs
        DPs_run = var.split("_")[0]
        coeffs = model.params
        
        quintile_coeffs = coeffs[coeffs.index.str.contains(var)]
        p_vals = model.pvalues[model.pvalues.index.str.contains(var)]
        conf_int = model.conf_int().loc[model.conf_int().index.str.contains(var)]

        # create a new dataframe
        df_coefficients = pd.DataFrame({
                    "N": N,
                    "DPs": DPs_run,
                    "Categories": quintile_coeffs.index,
                    "Outcome": outcomes,
                    "Coefficient" : quintile_coeffs.values,
                    "P_val" : p_vals.values,
                    "ci_lower" : conf_int.iloc[:, 0].values,
                    "ci_upper" : conf_int.iloc[:, 1].values
                    })

        df_coefficients["Categories"] = ["Q2", "Q3", "Q4", "Q5"]

        # combine the reference group
        reference_row = pd.DataFrame({
                        "N": N,
                        "DPs" : DPs_run,
                        "Categories" : ["Q1"],
                        "Outcome": outcomes,
                        "Coefficient" : [0.0],
                        "P_val" : np.nan,
                        "ci_lower" : np.nan,
                        "ci_upper" : np.nan
                        })
        
        # concat two dataframe
        df_results = pd.concat([reference_row, df_coefficients], ignore_index=True)
        # results_list.append(df_results)
        results_list = pd.concat([results_list, df_results], ignore_index=True)
        
    return results_list
    # change it into dataframe
    # results_df = pd.DataFrame(results_list)
    # set two decimals
    # results_df['CI'] = results_df.apply(lambda row: f"({row['ci_lower']:.2f}, {row['ci_upper']:.2f})", axis=1)
    # return results_df


# The function to run the logisitc regression model


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