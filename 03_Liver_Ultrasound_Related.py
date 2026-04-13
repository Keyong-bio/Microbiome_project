
# ============================================================
Source: 01_Liver_ultrasound_V2.py
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

# In[58]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
from pheno_utils import PhenoLoader


# In[59]:


## read in the liver ultrasound data
d2 = PhenoLoader('liver_ultrasound')
liver = d2[d2.fields]


# In[60]:


filenames = '~/studies/hpp_datasets/liver_ultrasound/liver_ultrasound.parquet'
liver_ultrasound = pd.read_parquet(filenames)
liver_ultrasound


# In[61]:


liver_ultrasound.columns


# In[62]:


liver_ultrasound[['elasticity_qbox_median', 'elasticity_qbox_mean', 'velocity_qbox_median', 'velocity_qbox_mean', 
                  'viscosity_qbox_median', 'viscosity_qbox_mean', 'dispersion_qbox_median', 'dispersion_qbox_mean']]


# In[63]:


filenames = '~/studies/hpp_datasets/liver_ultrasound/liver_ultrasound_aggregated.parquet'
liver_ultrasound_aggre = pd.read_parquet(filenames)


# In[64]:


print(liver_ultrasound_aggre.index.get_level_values('participant_id').nunique()) # 8228 individuals
liver_ultrasound_aggre.columns
liver_ultrasound_aggre[['elasticity_mean_mean_of_qboxes', 'elasticity_mean_median_of_qboxes', 
                        'elasticity_median_mean_of_qboxes', 'elasticity_median_median_of_qboxes']]


# In[65]:


# 6758 individuals
liver_ultrasound_aggre[liver_ultrasound_aggre.index.get_level_values('research_stage') == '00_00_visit'].index.get_level_values('participant_id').nunique()


# In[66]:


liver_ultrasound_aggre_base = liver_ultrasound_aggre[liver_ultrasound_aggre.index.get_level_values('research_stage') == '00_00_visit']
liver_ultrasound_aggre_base


# In[67]:


liver_ultrasound_aggre_base.index.get_level_values('research_stage').value_counts()


# In[68]:


# col = ["elasticity_median_median_of_qboxes", 'viscosity_median_median_of_qboxes', 'velocity_median_median_of_qboxes', 
#        'attenuation_coefficient_qbox', 'dispersion_median_median_of_qboxes']
age_sex = d2[["age", "sex"]].loc[:,:,"00_00_visit",:,:]
age_sex.loc[:, 'sex'] = pd.to_numeric(age_sex['sex'])
age_sex.info()
# merge with age and sex
age_sex = age_sex.reset_index().groupby(by = 'participant_id')[['age', 'sex']].mean().reset_index()
age_sex


# In[69]:


liver_ultrasound_aggre_base = liver_ultrasound_aggre_base.droplevel(['cohort', "research_stage", "array_index"]).reset_index()
liver_ultrasound_aggre_base = pd.merge(liver_ultrasound_aggre_base, age_sex, on = "participant_id", how = "left")
liver_ultrasound_aggre_base


# In[70]:


# liver_ultrasound_aggre_base.to_csv('liver_ultrasound_aggre_base.csv', index = False)


# In[71]:


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


# In[72]:


## select variables of interests
liver_ultrasound_aggre_base.columns

# 'elasticity_mean_median_of_qboxes', elasticity_mean_mean_of_qboxes, 'elasticity_median_median_of_qboxes', 'elasticity_median_mean_of_qboxes'
# velocity_mean_median_of_qboxes', velocity_mean_mean_of_qboxes, velocity_median_median_of_qboxes, velocity_median_mean_of_qboxes
# viscosity_mean_median_of_qboxes viscosity_mean_mean_of_qboxes, viscosity_median_median_of_qboxes, viscosity_median_mean_of_median
# dispersion_mean_median_of_qboxes, dispersion_mean_mean_of_qboxes, dispersion_median_median_of_qboxes, dispersion_median_mean_of_median


# In[73]:


cols = ["participant_id", 'elasticity_mean_median_of_qboxes', "elasticity_mean_mean_of_qboxes", 'elasticity_median_median_of_qboxes', 'elasticity_median_mean_of_qboxes',
       "velocity_mean_median_of_qboxes", "velocity_mean_mean_of_qboxes", "velocity_median_median_of_qboxes", "velocity_median_mean_of_qboxes",
       "viscosity_mean_median_of_qboxes", "viscosity_mean_mean_of_qboxes", "viscosity_median_median_of_qboxes", "viscosity_median_mean_of_median",
       "dispersion_mean_median_of_qboxes", "dispersion_mean_mean_of_qboxes", "dispersion_median_median_of_qboxes", "dispersion_median_mean_of_median",
       "age", "sex"]
liver_ultrasound_aggre_base2 = liver_ultrasound_aggre_base[cols]
liver_ultrasound_aggre_base2.to_csv('./Data/liver_ultrasound_aggre_base.csv', index = False)


# In[74]:


liver_ultrasound_aggre_base2


# In[75]:


liver_ultrasound_aggre_base2.columns


# In[76]:


transformed_cols = [f"{col}_rank_INT" for col in liver_ultrasound_aggre_base2.iloc[:, 1:16].columns]
transformed_cols


# In[77]:


liver_ultrasound_aggre_base2.iloc[:, 1:16]


# In[78]:


liver_ultrasound_aggre_base2.dropna(subset = ["elasticity_mean_median_of_qboxes"]).shape


# In[79]:


# after performing the rank_based_inverse_normal
liver_ultrasound_aggre_base2 = liver_ultrasound_aggre_base2.dropna()
# liver_ultrasound_aggre_base2.iloc[:, 1:16] = liver_ultrasound_aggre_base2.iloc[:, 1:16].apply(rank_INT)
liver_ultrasound_aggre_base3 = liver_ultrasound_aggre_base2.copy()
liver_ultrasound_aggre_base3[transformed_cols] = liver_ultrasound_aggre_base3.iloc[:, 1:16].transform(rank_INT)


# In[80]:


liver_ultrasound_aggre_base3.shape


# In[81]:


liver_ultrasound_aggre_base3.columns


# In[82]:


# Create a new columns list with replacements
# new_cols = liver_ultrasound_aggre_base2.columns.tolist()
# new_cols[1:16] = transformed_cols
# liver_ultrasound_aggre_base2.columns = new_cols


# In[83]:


liver_ultrasound_aggre_base3


# In[84]:


filenames = '~/studies/hpp_datasets/liver_ultrasound/liver_ultrasound.parquet'
liver_ultrasound = pd.read_parquet(filenames)
liver_ultrasound_base = liver_ultrasound[liver_ultrasound.index.get_level_values('research_stage') == '00_00_visit']
liver_ultrasound_base_select = liver_ultrasound_base[["elasticity_qbox_mean", "velocity_qbox_mean", "viscosity_qbox_mean", "dispersion_qbox_mean", "attenuation_coefficient_qbox", "speed_of_sound_qbox"]]
liver_ultrasound_base_select = liver_ultrasound_base_select.groupby("participant_id").mean()
liver_ultrasound_base_select = liver_ultrasound_base_select.reset_index()
liver_ultrasound_base_select


# In[85]:


liver_ultrasound_aggre_base3_2 = pd.merge(liver_ultrasound_aggre_base3, 
                                          liver_ultrasound_base_select[["participant_id", "attenuation_coefficient_qbox", "speed_of_sound_qbox"]],
                      on = "participant_id", how = "left")
liver_ultrasound_aggre_base3_2 = liver_ultrasound_aggre_base3_2.dropna(subset=["attenuation_coefficient_qbox", "speed_of_sound_qbox"])
liver_ultrasound_aggre_base3_2[["attenuation_coefficient_qbox_rank_INT", "speed_of_sound_qbox_rank_INT"]] = liver_ultrasound_aggre_base3_2.loc[:, ["attenuation_coefficient_qbox", "speed_of_sound_qbox"]].apply(rank_INT)


# In[86]:


liver_ultrasound_aggre_base3_2.columns


# In[87]:


liver_ultrasound_aggre_base3_2.shape


# ## Read in the dietary scores

# In[88]:


import pandas as pd
five_scores = pd.read_csv("../01_DPs_Diet_patterns/Adj/DPs_Final_score_outlieradj_v2.csv")
five_scores


# In[89]:


five_scores.set_index('participant_id').corr("spearman")


# In[90]:


liver_ultrasound_aggre_base3_dps= pd.merge(liver_ultrasound_aggre_base3_2, five_scores, on = "participant_id", how = "left")
liver_ultrasound_aggre_base3_dps


# In[91]:


# liver_ultrasound_aggre_base2_dps, calculate the missing for the dietary patterns
liver_ultrasound_aggre_base3_dps.columns

liver_ultrasound_aggre_base3_dps[['AHEI_2010_score_eadj', 'hPDI_score_eadj', 'rDII_score_eadj',
       'AMED_score_eadj', 'rEDIH_score_all_eadj']].isnull().sum()


# In[92]:


print(f'{6738-461} participants with dietary pattern score')


# In[93]:


def scaling(df, col_to_scaling):
    df_scaling = df.copy()
    
    for column in col_to_scaling:
        # calculate the mean and std first
        mean_val = df_scaling[column].mean()
        std_val = df_scaling[column].std()
        
        df_scaling[f"{column}_scaled"] = (df_scaling[column] - mean_val)/std_val
        
    return df_scaling  

col_to_scaling = ['AHEI_2010_score_eadj', 'hPDI_score_eadj', 'rDII_score_eadj', 'AMED_score_eadj', 'rEDIH_score_all_eadj']
liver_ultrasound_aggre_base3_dps_scaled = scaling(liver_ultrasound_aggre_base3_dps, col_to_scaling = col_to_scaling)
liver_ultrasound_aggre_base3_dps_scaled.head()


# In[94]:


liver_ultrasound_aggre_base3_dps_scaled.columns


# In[95]:


liver_ultrasound_aggre_base3_dps_scaled["age"].isnull().sum()


# In[96]:


liver_ultrasound_aggre_base3_dps_scaled["sex"].isnull().sum()


# ## Read in the lifestyle factors

# In[97]:


lifestyle_factor = pd.read_csv("../02_Lifestyle factors/lifestyle_factor_all_disease_V2.csv")
display(lifestyle_factor.head())
lifestyle_factor.shape


# In[98]:


lifestyle_factor.columns[lifestyle_factor.columns.str.contains(f'age')]


# In[99]:


lifestyle_factor = lifestyle_factor.drop(columns = ['sex_y', 'age', 'sex_x'])


# In[100]:


liver_ultrasound_aggre_base3_dps_scaled_life = pd.merge(liver_ultrasound_aggre_base3_dps_scaled,
                                                       lifestyle_factor,
                                                       on = "participant_id",
                                                       how = "left")
liver_ultrasound_aggre_base3_dps_scaled_life


# In[101]:


liver_ultrasound_aggre_base3_dps_scaled_life.columns


# In[102]:


col_to_scaling = ['elasticity_mean_median_of_qboxes',
       'elasticity_mean_mean_of_qboxes', 'elasticity_median_median_of_qboxes',
       'elasticity_median_mean_of_qboxes', 'velocity_mean_median_of_qboxes',
       'velocity_mean_mean_of_qboxes', 'velocity_median_median_of_qboxes',
       'velocity_median_mean_of_qboxes', 'viscosity_mean_median_of_qboxes',
       'viscosity_mean_mean_of_qboxes', 'viscosity_median_median_of_qboxes',
       'viscosity_median_mean_of_median', 'dispersion_mean_median_of_qboxes',
       'dispersion_mean_mean_of_qboxes', 'dispersion_median_median_of_qboxes',
       'dispersion_median_mean_of_median']
liver_ultrasound_aggre_base3_dps_scaled_life = scaling(liver_ultrasound_aggre_base3_dps_scaled_life, col_to_scaling = col_to_scaling)
liver_ultrasound_aggre_base3_dps_scaled_life.head()


# In[103]:


liver_ultrasound_aggre_base3_dps_scaled_life.to_csv('./Data/liver_ultrasound_aggre_base3_dps_scaled_life.csv', index = False)


# In[104]:


# containing scaled character for the colnames
liver_ultrasound_aggre_base3_dps_scaled_life.columns[liver_ultrasound_aggre_base3_dps_scaled_life.columns.str.contains("_scaled")]


# In[105]:


import seaborn as sns
sns.displot(liver_ultrasound_aggre_base3_dps_scaled_life.loc[liver_ultrasound_aggre_base3_dps_scaled_life.elasticity_mean_median_of_qboxes<10], 
            x = "elasticity_mean_median_of_qboxes")

sns.displot(liver_ultrasound_aggre_base3_dps_scaled_life, 
            x = "elasticity_mean_median_of_qboxes")


# In[ ]:





# In[ ]:






# ============================================================
Source: 03_MLR_DPs_Liver_V2.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Multiple linear regression analysis v2
# ### DPs to Live measures
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[68]:


# from A0_helpfunction_Copy1 import *


# In[69]:


import pandas as pd

df = pd.read_csv("liver_ultrasound_aggre_base2_dps_scaled_life_clean.csv")
df


# In[70]:


df1 = df.copy()
# df1.columns[df1.columns.str.contains(r'speed')]
df1.columns[df1.columns.str.contains(r'rank_INT')]


# In[71]:


# Exclude na rows for liver measures
df1 = df1.dropna(subset = df1.columns[1:17]) 
df1.shape


# In[72]:


transformed_cols = [f"{col}_rank_INT" for col in df1.iloc[:, 1:17].columns]
transformed_cols


# In[73]:


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


# In[74]:


# create new column names for transformed values
# after performing the rank_based_inverse_normal
df1[transformed_cols] = df1.iloc[:,1:17].apply(rank_INT) 


# In[75]:


#  hPDI should be adjusted also for alcohol intakes
Nutrients = pd.read_csv("~/studies/ruifang/Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
Nutrients_2.info()

# merge with alcohol intake
df_model2 = pd.merge(df1, Nutrients_2, on = "participant_id", how = 'left')
df_model2


# In[76]:


sum(df_model2["elasticity_mean_median_of_qboxes"].isna())


# In[77]:


# sum(df_model2["alcohol_g"].isna())
sum(df_model2["AHEI_2010_score_eadj"].isna()) # 461 without DPs


# In[78]:


df_model2 = df_model2.dropna(subset = ["AHEI_2010_score_eadj"])
df_model2.shape


# In[79]:


df_model2.columns[df_model2.columns.str.contains(r'_eadj')]


# In[80]:


df_model2_2 = df_model2.dropna(subset = ["AHEI_2010_score_eadj", 
                                         'hPDI_score_eadj', 
                                         'rDII_score_eadj',
                                         'AMED_score_eadj', 
                                         'rEDIH_score_all_eadj'])
print(df_model2_2.shape) # 6277

sum(df_model2_2["hPDI_score_eadj"].isna())


# In[81]:


df_MAFLD = pd.read_csv("../05_Medication_status/Medication_baseline.csv")

merge_df_MAFLD = pd.merge(df_model2_2, 
                          df_MAFLD[["participant_id", "NAFLD_baseline_diagnosis"]],
                          on = "participant_id", how = "left")

merge_df_MAFLD["MAFLD_baseline_diagnosis"] = merge_df_MAFLD["NAFLD_baseline_diagnosis"].replace({"Yes":1, "No": 0})
merge_df_MAFLD["MAFLD_baseline_diagnosis"].value_counts(dropna = False) #  with self-reported NAFLD

# merge_df_MAFLD = pd.merge(df_model2_2, df_MAFLD[["participant_id", "mafld_diagnosed"]],
#                           on = "participant_id", how = "left")

# merge_df_MAFLD["MAFLD_baseline_diagnosis"] = merge_df_MAFLD["mafld_diagnosed"].replace({"Yes":1, "No": 0})
# merge_df_MAFLD["MAFLD_baseline_diagnosis"].value_counts(dropna = False)


# # Remove the Self-reported MASLD

# In[82]:


# plot the box plot for the distribution of DPs between the MASLD and without MASLD
merge_df_MAFLD_filter = merge_df_MAFLD.loc[~(merge_df_MAFLD["MAFLD_baseline_diagnosis"] == 1)]


# In[83]:


import seaborn as sns
sns.boxplot(data=merge_df_MAFLD_filter, x="MAFLD_baseline_diagnosis", y="AHEI_2010_score_eadj", 
            hue="MAFLD_baseline_diagnosis", fill=False, gap=.1)

# sns.boxplot(data=merge_df_MAFLD_filter, x="MAFLD_baseline_diagnosis", y="hPDI_score_eadj", 
#             hue="MAFLD_baseline_diagnosis", fill=False, gap=.1)


# In[84]:


# sns.boxplot(data=merge_df_MAFLD, x="MAFLD_baseline_diagnosis", y="rDII_score_eadj", hue="MAFLD_baseline_diagnosis", fill=False, gap=.1)
# sns.boxplot(data=merge_df_MAFLD, x="MAFLD_baseline_diagnosis", y="AMED_score_eadj", hue="MAFLD_baseline_diagnosis", fill=False, gap=.1)
# sns.boxplot(data=merge_df_MAFLD, x="MAFLD_baseline_diagnosis", y="rEDIH_score_all_eadj", hue="MAFLD_baseline_diagnosis", fill=False, gap=.1)


# In[85]:


# Exclude those with MAFLD at baseline
merge_df_MAFLD_filter.shape # 5989 obs


# In[86]:


# df_model2_2_filter = merge_df_MAFLD.loc[merge_df_MAFLD["MAFLD_baseline_diagnosis"] == 0,:]
# df_model2_2_filter.shape # 4748 obs after exclude all potential NAFLD


# In[87]:


merge_df_MAFLD_filter.to_csv("DPs_to_Liver_Exclude_Self_MASLD_Baseline.csv", index = False)


# In[88]:


merge_df_MAFLD_filter.columns[merge_df_MAFLD_filter.columns.str.contains(r'scale')]


# # Update 2025-0530
# ### Clip those with extreme value of liver measures/ or deletecolumns_to_winsorize = ['col3', 'col5', 'col7']  # replace with actual column names
# 
# for col_name in columns_to_winsorize:
#     percentile_99 = df[col_name].quantile(0.99)
#     df[col_name] = df[col_name].clip(upper=percentile_99)

# In[89]:


# columns_to_winsorize = ['col3', 'col5', 'col7']  # replace with actual column names

# for col_name in columns_to_winsorize:
#     percentile_99 = merge_df_MAFLD_filter[col_name].quantile(0.99)
#     merge_df_MAFLD_filter[col_name] = merge_df_MAFLD_filter[col_name].clip(upper=percentile_99)

cols_to_check = ["elasticity_median_median_of_qboxes_rank_INT", 
           'velocity_median_median_of_qboxes_rank_INT',
           'viscosity_median_median_of_qboxes_rank_INT',
           'dispersion_median_median_of_qboxes_rank_INT',
           "attenuation_coefficient_qbox_rank_INT",
            "speed_of_sound_qbox_rank_INT"]

# # calculate 99th percentile for each column
# percentile99 = {}
# for cols in cols_to_check:
#     percentile99[cols] = merge_df_MAFLD_filter[cols].quantile(0.99)

# # create boolean mask
# mask = True
# for col_name in cols_to_check:
#     mask = mask & (merge_df_MAFLD_filter[col_name] <= percentile99[colname]

conditions = [merge_df_MAFLD_filter[col] <= merge_df_MAFLD_filter[col].quantile(0.99) for col in cols_to_check]
mask = pd.concat(conditions, axis=1).any(axis=1)  # any() instead of all()
merge_df_MAFLD_filter = merge_df_MAFLD_filter[mask]
merge_df_MAFLD_filter.shape


# In[90]:


# Calculat the different categories for Dietary pattern score
cols_process = ["AHEI_2010_score_eadj", 
                "hPDI_score_eadj", 
                "rDII_score_eadj", 
                "AMED_score_eadj", 
                "rEDIH_score_all_eadj"]

for col in cols_process:
    col_new = f'{col}_quntile'
    merge_df_MAFLD_filter[col_new] = pd.qcut(merge_df_MAFLD_filter[col], 5, labels = ["Q1","Q2", "Q3", "Q4", "Q5"])


# In[91]:


# DPs_grouops_select = DPs_grouops[["participant_id", 
# "AHEI_2010_score_eadj_quntile", 
# "hPDI_score_eadj_quntile", "rDII_score_eadj_quntile",
# "AMED_score_eadj_quntile", "rEDIH_score_all_eadj_quntile"]]

# df_model2_2_filter.head()


# In[92]:


# df_model2_2_filter_2 = pd.merge(df_model2_2_filter, DPs_grouops_select, how = "left", on = "participant_id")
# df_model2_2_filter_2.shape


# In[93]:


merge_df_MAFLD_filter.columns


# In[94]:


display(merge_df_MAFLD_filter["AHEI_2010_score_eadj_quntile"].value_counts())
display(merge_df_MAFLD_filter["hPDI_score_eadj_quntile"].value_counts())
merge_df_MAFLD_filter["rDII_score_eadj_quntile"].value_counts()


# In[95]:


# check the order of the categorical data
merge_df_MAFLD_filter["hPDI_score_eadj_quntile"].cat.categories


# In[96]:


merge_df_MAFLD_filter.to_csv("DFs_MAFLD_Excluded_Self_Report_MASLD.csv", index = False)


# # Run the GLM regression

# In[97]:


predictors = ["AHEI_2010_score_eadj_scaled", "hPDI_score_eadj_scaled", "rDII_score_eadj_scaled", 
              "AMED_score_eadj_scaled", "rEDIH_score_all_eadj_scaled"]

predictors_2 = ["AHEI_2010_score_eadj_quntile", "hPDI_score_eadj_quntile", "rDII_score_eadj_quntile",
               "AMED_score_eadj_quntile", "rEDIH_score_all_eadj_quntile"]

Outcomes = ["elasticity_median_median_of_qboxes_rank_INT", 
           'velocity_median_median_of_qboxes_rank_INT',
           'viscosity_median_median_of_qboxes_rank_INT',
           'dispersion_median_median_of_qboxes_rank_INT',
           "attenuation_coefficient_qbox_rank_INT",
            "speed_of_sound_qbox_rank_INT"
           ]


# In[98]:


import pandas as pd
import statsmodels.formula.api as smf


# ## Run as the continuous variables

# In[99]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from A0_helpfunction_Copy1 import *


# In[100]:


run_regression_formula(df = merge_df_MAFLD_filter, 
                            predictors = predictors[1], 
                            outcomes = "speed_of_sound_qbox_rank_INT", control_vars = None)


# In[101]:


results_continue_model1 = pd.DataFrame()

for livers in Outcomes:
    # print(metabolites)
    lm_fit = run_regression_formula(df = merge_df_MAFLD_filter, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    results_continue_model1 = pd.concat([results_continue_model1, lm_fit], ignore_index=True)
    
from scipy import stats
results_continue_model1['p_adj'] = stats.false_discovery_control(results_continue_model1['P_value'])
results_continue_model1_sig = results_continue_model1.loc[results_continue_model1['p_adj'] < 0.05] # select based on FDR < 0.05
results_continue_model1_sig


# In[102]:


model2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']

# for hPDI index adjusted for alcohol intake 
model2_2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use',"alcohol_g"]

model3 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use',  "bmi"]

model3_2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "alcohol_g",'bmi']

model4 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "NSAID_use", "CVD_f_history", "T2D_f_history"]

model4_2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "alcohol_g", "NSAID_use", "CVD_f_history", "T2D_f_history"]

# Model2
results_continue_model2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model2)
        results_continue_model2 = pd.concat([results_continue_model2, lm_fit], ignore_index=True)
    
# Model3
results_continue_model3 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model3)
        results_continue_model3 = pd.concat([results_continue_model3, lm_fit], ignore_index=True)

# Model4
results_continue_model4 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model4)
        results_continue_model4 = pd.concat([results_continue_model4, lm_fit], ignore_index=True)


# In[103]:


# Write a function to run p_value adjustment
def fdr_correction_ (df, p_col = "P_value", fdr_threshold = 0.05):
    df_copy = df.copy()
    df_copy["p_adj"] = stats.false_discovery_control(df_copy[p_col])
    df_sig = df_copy.loc[df_copy['p_adj'] < fdr_threshold]
    return df_copy, df_sig


# In[104]:


# adjust for pvalue
df_models = [results_continue_model1, results_continue_model2, results_continue_model3, results_continue_model4]
corrected_dfs = []
significant_dfs = []

for i, df in enumerate(df_models):
    df_corrected, df_sig = fdr_correction_(df)
    corrected_dfs.append(df_corrected)
    significant_dfs.append(df_sig)

# Access to results
results_continue_model1_corrected, results_continue_model2_corrected, results_continue_model3_corrected, results_continue_model4_corrected = corrected_dfs
results_continue_model1_sig, results_continue_model2_sig, results_continue_model3_sig, results_continue_model4_sig = significant_dfs


# In[105]:


display(results_continue_model1_sig)
display(results_continue_model2_sig)
display(results_continue_model3_sig)
display(results_continue_model4_sig)
# display(results_continue_model4_corrected)


# ## Run Categorial DPs
# - the same covariates

# In[106]:


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


# In[107]:


results_cat_model1 = pd.DataFrame()

for livers in Outcomes:
    lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter, 
                          predictors = predictors_2, 
                          outcomes = livers, 
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


# In[108]:


merge_df_MAFLD_filter.columns[merge_df_MAFLD_filter.columns.str.contains(r'sex')]


# In[109]:


merge_df_MAFLD_filter["sex_x"].value_counts()


# In[110]:


# merge_df_MAFLD_filter["sex_y"].value_counts() -> select sex_x


# In[111]:


# model2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
#          'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']

# # for hPDI index adjusted for alcohol intake 
# model2_2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
#          'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use',"alcohol_g"]

results_cat_model2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors_2:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model2)
        results_cat_model2 = pd.concat([results_cat_model2, lm_fit], ignore_index=True)
    
results_cat_model3 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors_2:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model3)
        results_cat_model3 = pd.concat([results_cat_model3, lm_fit], ignore_index=True)

results_cat_model4 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors_2:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter, predictors = predict_var, outcomes = livers, control_vars = model4)
        results_cat_model4 = pd.concat([results_cat_model4, lm_fit], ignore_index=True)


# In[112]:


# results_df_model2.to_csv("result_model2_categocial_exclude_base_MASLD_self_report.csv", index = False)


# In[113]:


results_cat_model4.shape


# # Plot the results

# In[114]:


get_ipython().system('pip install forestplot')


# In[115]:


results_cat_model2.head()


# In[116]:


# Outcome_name
Outcome_name = {
    "elasticity_median_median_of_qboxes_rank_INT": "Liver Elasticity",
    "viscosity_median_median_of_qboxes_rank_INT": "Liver Viscosity",
    "dispersion_median_median_of_qboxes_rank_INT": "Liver Dispersion",
    "attenuation_coefficient_qbox_rank_INT": "Attenuation coefficient",
    "velocity_median_median_of_qboxes_rank_INT": "Liver velocity",
    "speed_of_sound_qbox_rank_INT": "Speed of Sound"
}

results_cat_model2["Outcome"] = results_cat_model2["Outcome"].map(Outcome_name)


# In[117]:


# Based on the AHEI result, plot the forest plots
import matplotlib.pyplot as plt
import random

outcomes = results_cat_model2["Outcome"].unique() # liver outcomes
Categories_values = results_cat_model2["Categories"].unique() # Q1-Q5
dps_values = results_cat_model2["DPs"].unique() # DPs

# plotting for categorical MODEL2
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten() # simplifies the code, using simple index to axes, instead of the 2D coordinates

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = results_cat_model2[results_cat_model2["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["DPs"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Categories"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['DPs']]
        ax.errorbar(x=cat_data['Coefficient'], 
                    y=y_positions,
                    xerr=[cat_data['Coefficient'] - cat_data['ci_lower'], cat_data['ci_upper'] - cat_data['Coefficient']],
                    fmt='s', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.5, 0.5) # set x-axis limits

plt.tight_layout()
# plt.savefig("Categorical_DPs_to_Liver_Model2_Fiver_Cats.pdf", bbox_inches='tight')
plt.show()


# In[118]:


results_continue_model2.head()


# In[119]:


# I would like to combine all four Models into one data frame
results_continue_model1_corrected.loc[:, "Model"] = "Model1"
results_continue_model2_corrected.loc[:, "Model"] = "Model2"
results_continue_model3_corrected.loc[:, "Model"] = "Model3"
results_continue_model4_corrected.loc[:, "Model"] = "Model4"

results_cont_combine = pd.concat([results_continue_model1_corrected,
                                 results_continue_model2_corrected,
                                 results_continue_model3_corrected,
                                 results_continue_model4_corrected], ignore_index=True
                                )
results_cont_combine


# In[120]:


# Rename the Variables 
# Define a mapping dictionary
DPs_names = {
    "AHEI_2010_score_eadj_scaled": "AHEI",
    "hPDI_score_eadj_scaled" : "hPDI",
    "rDII_score_eadj_scaled": "rDII",
    "AMED_score_eadj_scaled": "AMED",
    "rEDIH_score_all_eadj_scaled": "rEDIH"
}


# mapping
results_cont_combine["Variable"] = results_cont_combine["Variable"].map(DPs_names)
results_cont_combine["Outcome"] = results_cont_combine["Outcome"].map(Outcome_name)
results_cont_combine.head()


# In[121]:


outcomes = results_cont_combine["Outcome"].unique()
Categories_values = results_cont_combine["Model"].unique() # Model1-4
dps_values = results_cont_combine["Variable"].unique() # DPs

# Plot for the continuous MODEL2
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = results_cont_combine[results_cont_combine["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Model"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]
        ax.errorbar(x=cat_data['Coefficient'], 
                    y=y_positions,
                    xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
                    fmt='s', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.5, 0.5) # set x-axis limits

plt.tight_layout()
# plt.savefig("Continues_DPs_to_Liver_four_models.pdf", bbox_inches='tight')
plt.show()


# In[122]:


# results_df_model2_clean = results_df_model2.loc[~results_df_model2["P_val"].isnull(),:]
# results_df_model2_clean.shape

# results_df_model2_clean["P_adj"] = multipletests(
#         results_df_model2_clean["P_val"],
#         alpha=0.05, 
#         method='fdr_bh'
# )[1]

# results_df_model2_clean.loc[results_df_model2_clean["P_adj"] < 0.05,:]


# # Analysis For male and Female separately
# - Perform for the continous DPs

# In[123]:


merge_df_MAFLD_filter["sex_x"].value_counts()
df_male = merge_df_MAFLD_filter.loc[merge_df_MAFLD_filter["sex_x"] == 1]
df_male.shape

df_female = merge_df_MAFLD_filter.loc[merge_df_MAFLD_filter["sex_x"] == 0]
df_female.shape


# In[124]:


# Male
# Model1
Male_results_continue_model1 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    lm_fit = run_regression_formula(df = df_male, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    Male_results_continue_model1 = pd.concat([Male_results_continue_model1, lm_fit], ignore_index=True)

# Model2
Male_results_continue_model2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model2)
        Male_results_continue_model2 = pd.concat([Male_results_continue_model2, lm_fit], ignore_index=True)
    
# Model3
Male_results_continue_model3 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model3)
        Male_results_continue_model3 = pd.concat([Male_results_continue_model3, lm_fit], ignore_index=True)

# Model4
Male_results_continue_model4 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model4)
        Male_results_continue_model4 = pd.concat([Male_results_continue_model4, lm_fit], ignore_index=True)

# Adjust for pvalue
Male_df_models = [Male_results_continue_model1, Male_results_continue_model2, 
             Male_results_continue_model3, Male_results_continue_model4]
Male_corrected_dfs = []
Male_significant_dfs = []

for i, df in enumerate(Male_df_models):
    df_corrected, df_sig = fdr_correction_(df)
    Male_corrected_dfs.append(df_corrected)
    Male_significant_dfs.append(df_sig)

# Access to results
Male_results_continue_model1_corrected, Male_results_continue_model2_corrected, Male_results_continue_model3_corrected, Male_results_continue_model4_corrected = Male_corrected_dfs
Male_results_continue_model1_sig, Male_results_continue_model2_sig, Male_results_continue_model3_sig, Male_results_continue_model4_sig = Male_significant_dfs


# In[125]:


Male_results_continue_model1_corrected.loc[:, "Model"] = "Model1"
Male_results_continue_model2_corrected.loc[:, "Model"] = "Model2"
Male_results_continue_model3_corrected.loc[:, "Model"] = "Model3"
Male_results_continue_model4_corrected.loc[:, "Model"] = "Model4"

Male_results_cont_combine = pd.concat([Male_results_continue_model1_corrected,
                                 Male_results_continue_model2_corrected,
                                 Male_results_continue_model3_corrected,
                                 Male_results_continue_model4_corrected], ignore_index=True
                                )
Male_results_cont_combine.shape


# In[126]:


Outcome_name


# In[127]:


Male_results_cont_combine.head()
Male_results_cont_combine["Variable"] = Male_results_cont_combine["Variable"].map(DPs_names)
Male_results_cont_combine["Outcome"] = Male_results_cont_combine["Outcome"].map(Outcome_name)


# In[128]:


Male_results_cont_combine.head()


# In[129]:


Categories_values


# In[130]:


# Plot for the continuous MODEL2
Categories_values = Male_results_cont_combine["Model"].unique() # Model1-4
# dps_values = Male_results_cont_combine["Variable"].unique() # DPs

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = Male_results_cont_combine[Male_results_cont_combine["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Model"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]
        ax.errorbar(x=cat_data['Coefficient'], 
                    y=y_positions,
                    xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
                    fmt='s', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.5, 0.5) # set x-axis limits

plt.tight_layout()
# plt.savefig("Male_Continues_DPs_to_Liver_four_models.pdf", bbox_inches='tight')
plt.show()


# ## Female

# In[131]:


# Female
# Model1
Female_results_continue_model1 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    lm_fit = run_regression_formula(df = df_female, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    Female_results_continue_model1 = pd.concat([Female_results_continue_model1, lm_fit], ignore_index=True)

# Model2
Female_results_continue_model2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model2)
        Female_results_continue_model2 = pd.concat([Female_results_continue_model2, lm_fit], ignore_index=True)
    
# Model3
Female_results_continue_model3 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model3)
        Female_results_continue_model3 = pd.concat([Female_results_continue_model3, lm_fit], ignore_index=True)

# Model4
Female_results_continue_model4 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model4)
        Female_results_continue_model4 = pd.concat([Female_results_continue_model4, lm_fit], ignore_index=True)

# Adjust for pvalue
Female_df_models = [Female_results_continue_model1, Female_results_continue_model2, 
             Female_results_continue_model3, Female_results_continue_model4]
Female_corrected_dfs = []
Female_significant_dfs = []

for i, df in enumerate(Female_df_models):
    df_corrected, df_sig = fdr_correction_(df)
    Female_corrected_dfs.append(df_corrected)
    Female_significant_dfs.append(df_sig)

# Access to results
Female_results_continue_model1_corrected, Female_results_continue_model2_corrected, Female_results_continue_model3_corrected, Female_results_continue_model4_corrected = Female_corrected_dfs
Female_results_continue_model1_sig, Female_results_continue_model2_sig, Female_results_continue_model3_sig, Female_results_continue_model4_sig = Female_significant_dfs

Female_results_continue_model1_corrected.loc[:, "Model"] = "Model1"
Female_results_continue_model2_corrected.loc[:, "Model"] = "Model2"
Female_results_continue_model3_corrected.loc[:, "Model"] = "Model3"
Female_results_continue_model4_corrected.loc[:, "Model"] = "Model4"

Female_results_cont_combine = pd.concat([Female_results_continue_model1_corrected,
                                 Female_results_continue_model2_corrected,
                                 Female_results_continue_model3_corrected,
                                 Female_results_continue_model4_corrected], ignore_index=True
                                )
Female_results_cont_combine.shape


# In[132]:


Female_results_cont_combine["Variable"] = Female_results_cont_combine["Variable"].map(DPs_names)
Female_results_cont_combine["Outcome"] = Female_results_cont_combine["Outcome"].map(Outcome_name)


# In[133]:


# Plot for the continuous MODEL2
Categories_values = Female_results_cont_combine["Model"].unique() # Model1-4
# dps_values = Male_results_cont_combine["Variable"].unique() # DPs

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = Female_results_cont_combine[Female_results_cont_combine["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Model"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]
        ax.errorbar(x=cat_data['Coefficient'], 
                    y=y_positions,
                    xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
                    fmt='s', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.5, 0.5) # set x-axis limits

plt.tight_layout()
# plt.savefig("Female_Continues_DPs_to_Liver_four_models.pdf", bbox_inches='tight')
plt.show()


# ## Export results for Table

# In[134]:


# Export multiple dataframe to separate CSV files

tables = {
    "Male_results_continue_model1_corrected": Male_results_continue_model1_corrected,
    "Male_results_continue_model2_corrected" : Male_results_continue_model2_corrected,
    "Male_results_continue_model3_corrected" : Male_results_continue_model3_corrected,
    "Male_results_continue_model4_corrected" : Male_results_continue_model4_corrected,
    "Male_results_continue_model1_sig" : Male_results_continue_model1_sig,
    "Male_results_continue_model2_sig" : Male_results_continue_model2_sig,
    "Male_results_continue_model3_sig" : Male_results_continue_model3_sig,
    "Male_results_continue_model4_sig" : Male_results_continue_model4_sig,
    "Female_results_continue_model1_corrected": Female_results_continue_model1_corrected,
    "Female_results_continue_model2_corrected" : Female_results_continue_model2_corrected,
    "Female_results_continue_model3_corrected" : Female_results_continue_model3_corrected,
    "Female_results_continue_model4_corrected" : Female_results_continue_model4_corrected,
    "Female_results_continue_model1_sig" : Female_results_continue_model1_sig,
    "Female_results_continue_model2_sig" : Female_results_continue_model2_sig,
    "Female_results_continue_model3_sig" : Female_results_continue_model3_sig,
    "Female_results_continue_model4_sig" : Female_results_continue_model4_sig,
    "results_continue_model1_corrected": results_continue_model1_corrected,
    "results_continue_model2_corrected" : results_continue_model2_corrected,
    "results_continue_model3_corrected" : results_continue_model3_corrected,
    "results_continue_model4_corrected" : results_continue_model4_corrected,
    "results_continue_model1_sig" : results_continue_model1_sig,
    "results_continue_model2_sig" : results_continue_model2_sig,
    "results_continue_model3_sig" : results_continue_model3_sig,
    "results_continue_model4_sig" : results_continue_model4_sig,
    "results_cat_model1" : results_cat_model1,
    "results_cat_model2" : results_cat_model2,
    "results_cat_model3" : results_cat_model3,
    "results_cat_model4" : results_cat_model4
}

# for name, df in tables.items():
#     df.to_csv(f'{name}.csv', index = False)



# ============================================================
Source: 03_MLR_DPs_Liver_V3.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Multiple linear regression analysis v3
# ### DPs to Live measures
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[2]:


from A0_helpfunction_Copy1 import *


# In[3]:


import pandas as pd

df = pd.read_csv("./Data/liver_ultrasound_aggre_base3_dps_scaled_life.csv")
df


# In[4]:


df.columns[df.columns.str.contains("age")]


# In[5]:


print(df["age"].isnull().sum())
df["age_all"].isnull().sum()


# In[6]:


df1 = df.copy()
# df1.columns[df1.columns.str.contains(r'speed')]
df1.columns[df1.columns.str.contains(r'rank_INT')]


# In[7]:


df1.shape


# In[8]:


# Exclude na rows for liver measures
df1 = df1.dropna(subset = df1.columns[1:17]) 
df1.shape


# In[9]:


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


# In[10]:


# create new column names for transformed values
# after performing the rank_based_inverse_normal
# df1[transformed_cols] = df1.iloc[:,1:17].apply(rank_INT) 


# In[11]:


#  hPDI should be adjusted also for alcohol intakes
Nutrients = pd.read_csv("~/studies/ruifang/Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
Nutrients_2.info()

# merge with alcohol intake
df_model2 = pd.merge(df1, Nutrients_2, on = "participant_id", how = 'left')
df_model2


# In[12]:


sum(df_model2["elasticity_mean_median_of_qboxes"].isna())


# In[13]:


# sum(df_model2["alcohol_g"].isna())
sum(df_model2["AHEI_2010_score_eadj"].isna()) # 461 without DPs


# In[14]:


df_model2["alcohol_g"].describe()


# In[15]:


df_model2 = df_model2.dropna(subset = ["AHEI_2010_score_eadj"])
df_model2.shape


# In[16]:


df_model2.columns[df_model2.columns.str.contains(r'_eadj')]


# In[17]:


df_model2_2 = df_model2.dropna(subset = ["AHEI_2010_score_eadj", 
                                         'hPDI_score_eadj', 
                                         'rDII_score_eadj',
                                         'AMED_score_eadj', 
                                         'rEDIH_score_all_eadj'])
print(df_model2_2.shape) # 6277

sum(df_model2_2["hPDI_score_eadj"].isna())


# In[18]:


df_MAFLD = pd.read_csv("../05_Medication_status/Medication_baseline.csv")

merge_df_MAFLD = pd.merge(df_model2_2, 
                          df_MAFLD[["participant_id", "NAFLD_baseline_diagnosis"]],
                          on = "participant_id", how = "left")

merge_df_MAFLD["MAFLD_baseline_diagnosis"] = merge_df_MAFLD["NAFLD_baseline_diagnosis"].map({"Yes":1, "No": 0})
merge_df_MAFLD["MAFLD_baseline_diagnosis"].value_counts(dropna = False) # 288 with self-reported NAFLD

# merge_df_MAFLD = pd.merge(df_model2_2, df_MAFLD[["participant_id", "mafld_diagnosed"]],
#                           on = "participant_id", how = "left")

# merge_df_MAFLD["MAFLD_baseline_diagnosis"] = merge_df_MAFLD["mafld_diagnosed"].replace({"Yes":1, "No": 0})
# merge_df_MAFLD["MAFLD_baseline_diagnosis"].value_counts(dropna = False)


# # Remove the Self-reported MASLD

# In[19]:


# plot the box plot for the distribution of DPs between the MASLD and without MASLD
merge_df_MAFLD_filter = merge_df_MAFLD.loc[~(merge_df_MAFLD["MAFLD_baseline_diagnosis"] == 1)]


# In[20]:


merge_df_MAFLD_filter.shape


# In[21]:


merge_df_MAFLD_filter["smoking_status"].value_counts(dropna = False)


# In[22]:


merge_df_MAFLD_filter["Hormone_use"].value_counts(dropna = False)

pd.crosstab(merge_df_MAFLD_filter["sex_x"], merge_df_MAFLD_filter["Hormone_use"])


# In[23]:


import seaborn as sns

sns.boxplot(data = merge_df_MAFLD_filter, 
            x = "MAFLD_baseline_diagnosis", 
            y = "AHEI_2010_score_eadj", 
            hue = "MAFLD_baseline_diagnosis", fill = False, gap=.1)

# sns.boxplot(data=merge_df_MAFLD_filter, x="MAFLD_baseline_diagnosis", y="hPDI_score_eadj", 
#             hue="MAFLD_baseline_diagnosis", fill=False, gap=.1)


# In[24]:


merge_df_MAFLD_filter2 = merge_df_MAFLD.loc[~(merge_df_MAFLD["MAFLD_baseline_diagnosis"].isna())]
# sns.boxplot(data = merge_df_MAFLD_filter2, x = "MAFLD_baseline_diagnosis", y = "AHEI_2010_score_eadj", 
#             hue = "MAFLD_baseline_diagnosis", fill = False, gap=.1)

sns.boxplot(data = merge_df_MAFLD_filter2, 
            x = "MAFLD_baseline_diagnosis", 
            y = "hPDI_score_eadj", 
            hue = "MAFLD_baseline_diagnosis", fill = False, gap=.1)


# In[25]:


# sns.boxplot(data=merge_df_MAFLD, x="MAFLD_baseline_diagnosis", y="rDII_score_eadj", hue="MAFLD_baseline_diagnosis", fill=False, gap=.1)
# sns.boxplot(data=merge_df_MAFLD, x="MAFLD_baseline_diagnosis", y="AMED_score_eadj", hue="MAFLD_baseline_diagnosis", fill=False, gap=.1)
# sns.boxplot(data=merge_df_MAFLD, x="MAFLD_baseline_diagnosis", y="rEDIH_score_all_eadj", hue="MAFLD_baseline_diagnosis", fill=False, gap=.1)


# In[26]:


merge_df_MAFLD_filter.columns


# In[27]:


# Exclude those with MAFLD at baseline
merge_df_MAFLD_filter.shape # 5989 obs


# In[28]:


# df_model2_2_filter = merge_df_MAFLD.loc[merge_df_MAFLD["MAFLD_baseline_diagnosis"] == 0,:]
# df_model2_2_filter.shape # 4748 obs after exclude all potential NAFLD


# In[29]:


merge_df_MAFLD_filter.to_csv("./Data/DPs_to_Liver_Exclude_Self_MASLD_Baseline.csv", index = False)


# In[30]:


merge_df_MAFLD_filter.columns[merge_df_MAFLD_filter.columns.str.contains(r'INT')]


# In[31]:


merge_df_MAFLD_filter["sex_x"].value_counts()


# # Update 2025-0530
# ### Clip those with extreme value of liver measures/ or delete columns_to_winsorize = ['col3', 'col5', 'col7']  # replace with actual column names
# 
# for col_name in columns_to_winsorize:
#     percentile_99 = df[col_name].quantile(0.99)
#     df[col_name] = df[col_name].clip(upper=percentile_99)

# In[32]:


# columns_to_winsorize = ['col3', 'col5', 'col7']  # replace with actual column names

# for col_name in columns_to_winsorize:
#     percentile_99 = merge_df_MAFLD_filter[col_name].quantile(0.99)
#     merge_df_MAFLD_filter[col_name] = merge_df_MAFLD_filter[col_name].clip(upper=percentile_99)

# cols_to_check = ["elasticity_median_median_of_qboxes_rank_INT", 
#            'velocity_median_median_of_qboxes_rank_INT',
#            'viscosity_median_median_of_qboxes_rank_INT',
#            'dispersion_median_median_of_qboxes_rank_INT',
#            "attenuation_coefficient_qbox_rank_INT",
#             "speed_of_sound_qbox_rank_INT"]


cols_to_check = ["elasticity_median_median_of_qboxes", 
           'velocity_median_median_of_qboxes',
           'viscosity_median_median_of_qboxes',
           'dispersion_median_median_of_qboxes',
           "attenuation_coefficient_qbox",
            "speed_of_sound_qbox"]


for col_name in cols_to_check:
    percentile_99 = merge_df_MAFLD_filter[col_name].quantile(0.99)
    percentile_01 = merge_df_MAFLD_filter[col_name].quantile(0.01)
    merge_df_MAFLD_filter = merge_df_MAFLD_filter.copy()
    merge_df_MAFLD_filter[f'{col_name}_clipped'] = merge_df_MAFLD_filter[col_name].clip(lower = percentile_01, upper = percentile_99)
    
# conditions = [merge_df_MAFLD_filter[col] <= merge_df_MAFLD_filter[col].quantile(0.99) for col in cols_to_check]
# print(conditions)
# # any() instead of all()
# mask = pd.concat(conditions, axis=1).any(axis=1)  
# merge_df_MAFLD_filter = merge_df_MAFLD_filter[mask]
# merge_df_MAFLD_filter.shape


# In[33]:


merge_df_MAFLD_filter


# In[34]:


merge_df_MAFLD_filter["speed_of_sound_qbox"].describe()


# In[35]:


threshold = merge_df_MAFLD_filter["speed_of_sound_qbox"].quantile(0.995)
print(threshold)

filter_df = merge_df_MAFLD_filter[merge_df_MAFLD_filter["speed_of_sound_qbox"] <= threshold]
filter_df.shape


# In[36]:


# merge_df_MAFLD_filter.iloc[np.where(merge_df_MAFLD_filter["speed_of_sound_qbox"] > 1600)[0],:]


# In[37]:


cols_RINT = merge_df_MAFLD_filter.columns[merge_df_MAFLD_filter.columns.str.contains(r'clip')]
display(cols_RINT)

# Trasnformation for the clipped variables
transformed_cols = [f"{col}_rank_INT" for col in cols_RINT]
transformed_cols

merge_df_MAFLD_filter_2 = merge_df_MAFLD_filter.copy()

merge_df_MAFLD_filter_2[transformed_cols] = merge_df_MAFLD_filter_2.filter(regex='clip').transform(rank_INT)


# In[38]:


# Calculat the different categories for Dietary pattern score
cols_process = ["AHEI_2010_score_eadj", 
                "hPDI_score_eadj", 
                "rDII_score_eadj", 
                "AMED_score_eadj", 
                "rEDIH_score_all_eadj"]

merge_df_MAFLD_filter_3 = merge_df_MAFLD_filter_2.copy()
for col in cols_process:
    col_new = f'{col}_quntile'
    merge_df_MAFLD_filter_3[col_new] = pd.qcut(merge_df_MAFLD_filter_3[col], 5, labels = ["Q1","Q2", "Q3", "Q4", "Q5"])


# In[39]:


# merge_df_MAFLD_filter_3.columns


# In[40]:


display(merge_df_MAFLD_filter_3["AHEI_2010_score_eadj_quntile"].value_counts())
display(merge_df_MAFLD_filter_3["hPDI_score_eadj_quntile"].value_counts())
merge_df_MAFLD_filter_3["rDII_score_eadj_quntile"].value_counts()


# In[41]:


# check the order of the categorical data
merge_df_MAFLD_filter_3["hPDI_score_eadj_quntile"].cat.categories


# In[42]:


merge_df_MAFLD_filter_3.to_csv("./Data/DFs_MAFLD_Excluded_Self_Report_MASLD_v2.csv", index = False)


# In[43]:


merge_df_MAFLD_filter_3.columns[merge_df_MAFLD_filter_3.columns.str.contains(r"clipped_rank_INT")]


# # Run the GLM regression

# In[44]:


predictors = ["AHEI_2010_score_eadj_scaled", "hPDI_score_eadj_scaled", "rDII_score_eadj_scaled", 
              "AMED_score_eadj_scaled", "rEDIH_score_all_eadj_scaled"]

predictors_2 = ["AHEI_2010_score_eadj_quntile", "hPDI_score_eadj_quntile", "rDII_score_eadj_quntile",
               "AMED_score_eadj_quntile", "rEDIH_score_all_eadj_quntile"]

Outcomes = ["elasticity_median_median_of_qboxes_rank_INT", 
           'velocity_median_median_of_qboxes_rank_INT',
           'viscosity_median_median_of_qboxes_rank_INT',
           'dispersion_median_median_of_qboxes_rank_INT',
           "attenuation_coefficient_qbox_rank_INT",
            "speed_of_sound_qbox_rank_INT"
           ]

# Clip the liver outcomes
Outcomes2 = merge_df_MAFLD_filter_3.columns[merge_df_MAFLD_filter_3.columns.str.contains(r"clipped_rank_INT")].tolist()


# In[45]:


import pandas as pd
import statsmodels.formula.api as smf


# ## Run as the continuous variables

# In[46]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from A0_helpfunction_Copy1 import *


# In[47]:


display(run_regression_formula(df = merge_df_MAFLD_filter_3, 
                            predictors = predictors[1], 
                            outcomes = "speed_of_sound_qbox_rank_INT", control_vars = None))

display(run_regression_formula(df = merge_df_MAFLD_filter_3, 
                            predictors = predictors[1], 
                            outcomes = "speed_of_sound_qbox_clipped_rank_INT", control_vars = None))

display(run_regression_formula(df = filter_df, 
                            predictors = predictors[1], 
                            outcomes = "speed_of_sound_qbox_rank_INT", control_vars = None))


# In[48]:


results_continue_model1 = pd.DataFrame()

for livers in Outcomes:
    # print(metabolites)
    lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    results_continue_model1 = pd.concat([results_continue_model1, lm_fit], ignore_index=True)
    
from scipy import stats
results_continue_model1['p_adj'] = stats.false_discovery_control(results_continue_model1['P_value'])
results_continue_model1_sig = results_continue_model1.loc[results_continue_model1['p_adj'] < 0.05] # select based on FDR < 0.05
results_continue_model1_sig


# In[49]:


results_continue_model1_sen = pd.DataFrame()

for livers in Outcomes2:
    # print(metabolites)
    lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    results_continue_model1_sen = pd.concat([results_continue_model1_sen, lm_fit], ignore_index=True)
    
from scipy import stats
results_continue_model1_sen['p_adj'] = stats.false_discovery_control(results_continue_model1_sen['P_value'])
results_continue_model1_sen_sig = results_continue_model1_sen.loc[results_continue_model1_sen['p_adj'] < 0.05] # select based on FDR < 0.05
results_continue_model1_sen_sig


# In[50]:


results_continue_model1_sen2 = pd.DataFrame()

for livers in Outcomes:
    # print(metabolites)
    lm_fit = run_regression_formula(df = filter_df, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    results_continue_model1_sen2 = pd.concat([results_continue_model1_sen2, lm_fit], ignore_index=True)
    
from scipy import stats
results_continue_model1_sen2['p_adj'] = stats.false_discovery_control(results_continue_model1_sen2['P_value'])
results_continue_model1_sen2_sig = results_continue_model1_sen2.loc[results_continue_model1_sen2['p_adj'] < 0.05] # select based on FDR < 0.05
results_continue_model1_sen2_sig


# In[51]:


model2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']

# for hPDI index adjusted for alcohol intake 
model2_2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use',"alcohol_g"]

# adjustment for BMI
model3 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "bmi"]

model3_2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "alcohol_g", 'bmi']

# adjustment for Disease history
model4 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "NSAID_use", "CVD_f_history", "T2D_f_history"]

model4_2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "NSAID_use", "alcohol_g", "CVD_f_history", "T2D_f_history"]

# Model2
results_continue_model2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model2)
        results_continue_model2 = pd.concat([results_continue_model2, lm_fit], ignore_index=True)
    
# Model3
results_continue_model3 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model3)
        results_continue_model3 = pd.concat([results_continue_model3, lm_fit], ignore_index=True)

# Model4
results_continue_model4 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model4)
        results_continue_model4 = pd.concat([results_continue_model4, lm_fit], ignore_index=True)


# In[52]:


#### Sensitivity analysis
# Model2
results_continue_model2_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model2)
        results_continue_model2_sen = pd.concat([results_continue_model2_sen, lm_fit], ignore_index=True)
    
# Model3
results_continue_model3_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model3)
        results_continue_model3_sen = pd.concat([results_continue_model3_sen, lm_fit], ignore_index=True)

# Model4
results_continue_model4_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model4)
        results_continue_model4_sen = pd.concat([results_continue_model4_sen, lm_fit], ignore_index=True)


# In[53]:


# Write a function to run p_value adjustment
def fdr_correction_ (df, p_col = "P_value", fdr_threshold = 0.05):
    df_copy = df.copy()
    df_copy["p_adj"] = stats.false_discovery_control(df_copy[p_col])
    df_sig = df_copy.loc[df_copy['p_adj'] < fdr_threshold]
    return df_copy, df_sig


# In[54]:


# adjust for pvalue for primary analysis
df_models = [results_continue_model1, results_continue_model2, results_continue_model3, results_continue_model4]
corrected_dfs = []
significant_dfs = []

for i, df in enumerate(df_models):
    df_corrected, df_sig = fdr_correction_(df)
    corrected_dfs.append(df_corrected)
    significant_dfs.append(df_sig)

# Access to results
results_continue_model1_corrected, results_continue_model2_corrected, results_continue_model3_corrected, results_continue_model4_corrected = corrected_dfs
results_continue_model1_sig, results_continue_model2_sig, results_continue_model3_sig, results_continue_model4_sig = significant_dfs


# In[55]:


# adjust for pvalue for the sensitivity analysis
df_models_sen = [results_continue_model1_sen, results_continue_model2_sen, results_continue_model3_sen, results_continue_model4_sen]
corrected_dfs_sen = []
significant_dfs_sen = []

for i, df in enumerate(df_models_sen):
    df_sen_corrected, df_sen_sig = fdr_correction_(df)
    corrected_dfs_sen.append(df_sen_corrected)
    significant_dfs_sen.append(df_sen_sig)


# In[56]:


# Access to results
results_continue_model1_sen_corrected, results_continue_model2_sen_corrected, results_continue_model3_sen_corrected, results_continue_model4_sen_corrected = corrected_dfs_sen
results_continue_model1_sen_sig, results_continue_model2_sen_sig, results_continue_model3_sen_sig, results_continue_model4_sen_sig = significant_dfs_sen


# In[57]:


display(results_continue_model2_sig)
results_continue_model2_sen_sig


# In[58]:


# Exclude those with extreme large speed of sound
#### Sensitivity analysis
# Model2
results_continue_model2_sen_2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = filter_df, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = filter_df, predictors = predict_var, outcomes = livers, control_vars = model2)
        results_continue_model2_sen_2 = pd.concat([results_continue_model2_sen_2, lm_fit], ignore_index=True)
    
# Model3
results_continue_model3_sen_2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = filter_df, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = filter_df, predictors = predict_var, outcomes = livers, control_vars = model3)
        results_continue_model3_sen_2 = pd.concat([results_continue_model3_sen_2, lm_fit], ignore_index=True)

# Model4
results_continue_model4_sen_2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = filter_df, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = filter_df, predictors = predict_var, outcomes = livers, control_vars = model4)
        results_continue_model4_sen_2 = pd.concat([results_continue_model4_sen_2, lm_fit], ignore_index=True)

# adjust for pvalue for the sensitivity analysis
df_models_sen_2 = [results_continue_model1_sen2, results_continue_model2_sen_2, results_continue_model3_sen_2, results_continue_model4_sen_2]
corrected_dfs_sen_2 = []
significant_dfs_sen_2 = []

for i, df in enumerate(df_models_sen_2):
    df_sen_corrected, df_sen_sig = fdr_correction_(df)
    corrected_dfs_sen_2.append(df_sen_corrected)
    significant_dfs_sen_2.append(df_sen_sig)

# Access to results
results_continue_model1_sen2_corrected, results_continue_model2_sen2_corrected, results_continue_model3_sen2_corrected, results_continue_model4_sen2_corrected = corrected_dfs_sen_2
results_continue_model1_sen2_sig, results_continue_model2_sen2_sig, results_continue_model3_sen2_sig, results_continue_model4_sen2_sig = significant_dfs_sen_2


# In[59]:


DPs_to_Liver_Res_sen2 = pd.concat([results_continue_model1_sen2_corrected, results_continue_model2_sen2_corrected, results_continue_model3_sen2_corrected, 
                                   results_continue_model4_sen2_corrected])
DPs_to_Liver_Res_sen2

DPs_to_Liver_Res_sen2.to_csv("./Results/DPs_to_Liver_Res_sen2_exclude extreme Speed.csv", index = False)


# ## Export the results

# In[60]:


print(results_continue_model1_corrected.shape)

# combine the result into a big dataframe
DPs_to_Liver_Res = pd.concat([results_continue_model1_corrected, results_continue_model2_corrected,
          results_continue_model3_corrected, results_continue_model4_corrected])
DPs_to_Liver_Res


# In[61]:


DPs_to_Liver_Res.to_csv("./Results/DPs_to_Liver_Res_all.csv", index = False)


# In[62]:


# Export the sensitivity analysis
# combine the result into a big dataframe
DPs_to_Liver_Res_sen = pd.concat([results_continue_model1_sen_corrected, results_continue_model2_sen_corrected,
          results_continue_model3_sen_corrected, results_continue_model4_sen_corrected])
DPs_to_Liver_Res_sen.shape

DPs_to_Liver_Res_sen.to_csv("./Results/DPs_to_Liver_Res_all_sens.csv", index = False)


# ## Run Categorial DPs
# - the same covariates for adjustment

# In[63]:


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


# In[64]:


results_cat_model1 = pd.DataFrame()

for livers in Outcomes:
    lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter_3, 
                          predictors = predictors_2, 
                          outcomes = livers, 
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


# In[65]:


merge_df_MAFLD_filter.columns[merge_df_MAFLD_filter.columns.str.contains(r'sex')]


# In[188]:


merge_df_MAFLD_filter["sex_x"].value_counts()


# In[67]:


# merge_df_MAFLD_filter["sex_y"].value_counts() -> select sex_x


# In[68]:


# model2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
#          'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']

# # for hPDI index adjusted for alcohol intake 
# model2_2 = ['sex_x', 'age', 'smoking_status', 'edu_status', 
#          'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use',"alcohol_g"]

results_cat_model2 = pd.DataFrame()

for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors_2:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model2)
        results_cat_model2 = pd.concat([results_cat_model2, lm_fit], ignore_index=True)
    
results_cat_model3 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors_2:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model3)
        results_cat_model3 = pd.concat([results_cat_model3, lm_fit], ignore_index=True)

results_cat_model4 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors_2:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_GLM_Group(df = merge_df_MAFLD_filter_3, predictors = predict_var, outcomes = livers, control_vars = model4)
        results_cat_model4 = pd.concat([results_cat_model4, lm_fit], ignore_index=True)


# In[69]:


# results_df_model2.to_csv("result_model2_categocial_exclude_base_MASLD_self_report.csv", index = False)
results_cat_model1_clean


# In[70]:


results_cat_model2_clean = results_cat_model2.loc[~results_cat_model2["P_val"].isnull(),:]
results_cat_model2_clean.shape

results_cat_model2_clean["P_adj"] = multipletests(
        results_cat_model2_clean["P_val"],
        alpha=0.05, 
        method='fdr_bh'
)[1]

results_cat_model2_clean
display(results_cat_model2_clean[results_cat_model2_clean["P_adj"] < 0.05])

# Model3
results_cat_model3_clean = results_cat_model3.loc[~results_cat_model3["P_val"].isnull(),:]
results_cat_model3_clean.shape

results_cat_model3_clean["P_adj"] = multipletests(
        results_cat_model3_clean["P_val"],
        alpha=0.05, 
        method='fdr_bh'
)[1]

results_cat_model3_clean
display(results_cat_model3_clean[results_cat_model3_clean["P_adj"] < 0.05])

# Model4
results_cat_model4_clean = results_cat_model4.loc[~results_cat_model4["P_val"].isnull(),:]
results_cat_model4_clean.shape

results_cat_model4_clean["P_adj"] = multipletests(
        results_cat_model4_clean["P_val"],
        alpha=0.05, 
        method='fdr_bh'
)[1]

results_cat_model4_clean
display(results_cat_model4_clean[results_cat_model4_clean["P_adj"] < 0.05])


# In[71]:


results_cat_model1.shape
# results_cat_model2.shape
# results_cat_model3.shape
# results_cat_model4.shape

DP_cat_to_Liver = pd.concat([results_cat_model1, results_cat_model2, results_cat_model3, results_cat_model4])
DP_cat_to_Liver.shape

DP_cat_to_Liver.to_csv("./Results/DP_cat_to_liver.csv", index = False)


# In[72]:


DP_cat_to_Liver_adjust = pd.concat([results_cat_model1_clean, results_cat_model2_clean, results_cat_model3_clean, results_cat_model4_clean])
DP_cat_to_Liver_adjust.shape

DP_cat_to_Liver_adjust.to_csv("./Results/DP_cat_to_liver_adjust.csv", index = False)


# In[73]:


# DP_ref = DP_cat_to_Liver[DP_cat_to_Liver["Categories"] == "Q1"]
# DP_ref["P_adj"] = np.nan
# DP_ref

# # Combine two dataframes
# DP_cat_to_Liver_adjust = pd.concat([DP_ref, DP_cat_to_Liver_adjust])
# DP_cat_to_Liver_adjust


# In[74]:


# DP_cat_to_Liver_adjust.to_csv("./Results/DP_cat_to_liver_adjust.csv", index = False)


# # Plot the results

# In[75]:


# !pip install forestplot


# In[76]:


# Outcome_name
Outcome_name = {
    "elasticity_median_median_of_qboxes_rank_INT": "Liver Elasticity",
    "viscosity_median_median_of_qboxes_rank_INT": "Liver Viscosity",
    "dispersion_median_median_of_qboxes_rank_INT": "Liver Dispersion",
    "attenuation_coefficient_qbox_rank_INT": "Attenuation coefficient",
    "velocity_median_median_of_qboxes_rank_INT": "Liver Velocity",
    "speed_of_sound_qbox_rank_INT": "Speed of Sound"
}

DP_cat_to_Liver_adjust["Outcome"] = DP_cat_to_Liver_adjust["Outcome"].map(Outcome_name)


# In[77]:


DP_cat_to_Liver_adjust


# In[78]:


DP_cat_to_Liver_adjust = DP_cat_to_Liver_adjust[DP_cat_to_Liver_adjust["N"] == 4447]
DP_cat_to_Liver_adjust


# In[79]:


# Based on the AHEI result, plot the forest plots
import matplotlib.pyplot as plt
import random

outcomes = DP_cat_to_Liver_adjust["Outcome"].unique() # liver outcomes
Categories_values = DP_cat_to_Liver_adjust["Categories"].unique() # Q2-Q5
dps_values = DP_cat_to_Liver_adjust["DPs"].unique() # DPs

# plotting for categorical MODEL2
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# simplifies the code, using simple index to axes, instead of the 2D coordinates
axes = axes.flatten() 

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = DP_cat_to_Liver_adjust[DP_cat_to_Liver_adjust["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["DPs"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Categories"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['DPs']]
        
        # Assign marker based on the p_adj threshold
        for idx, (y_pos, p_val) in enumerate(zip(y_positions, cat_data["P_adj"])):
            # change the fill for the point
            if p_val < 0.05:
                # filled with color
                markerfacecolor = colors[j % len(colors)]
            else:
                markerfacecolor = "white"

            # Add label only the first iteration
            label = cat if idx == 0 else ""
            
            ax.errorbar(x=cat_data['Coefficient'].iloc[idx], 
                        y=y_pos,
                        xerr=[[cat_data['Coefficient'].iloc[idx] - cat_data['ci_lower'].iloc[idx]], 
                              [cat_data['ci_upper'].iloc[idx] - cat_data['Coefficient'].iloc[idx]]],
                        fmt='o', 
                        capsize=4, 
                        markerfacecolor = markerfacecolor,
                        markersize = 4,
                        markeredgecolor = colors[j % len(colors)],
                        markeredgewidth = 2,
                        label = label, 
                        color = colors[j % len(colors)])

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)

    if i % 3 != 0: # not the first column
        ax.set_yticklabels([])
        
    # ax.set_ylabel("DPs")
    # ax.set_xlabel('Coefficient')
    ax.set_title(f'{outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    # ax.legend()
    # ax.legend(loc = "best", fontsize = 10, frameon = True)
    ax.set_xlim(-0.25, 0.5) # set x-axis limits
    

plt.tight_layout(rect = [0.03, 0.03, 0.95, 0.95])
fig.supxlabel("Coefficient (beta) with Confidence Interval", fontsize = 12)
fig.supylabel("Dietary pattern Scores", fontsize = 12)

# Single shared legend from the first subplot
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc = "upper center", bbox_to_anchor = (0.5, 1.02),
#           ncol = len(Categories_values), fontsize = 12, frameon = True)


# Update the shared legend
from matplotlib.lines import Line2D

# Category color legend
color_handles = [Line2D([0], [0], marker = "o", color = "w", markerfacecolor = colors[j % len(colors)],
                       markeredgecolor = colors[j % len(colors)], markersize = 8, label = cat)
                for j, cat in enumerate(Categories_values)]

# Solid or hollow
sig_handles = [Line2D([0], [0], marker = "o", color = "w", markerfacecolor = "grey",
                     markeredgecolor = "grey", markersize = 8, label = "P_adj < 0.05"),
               Line2D([0], [0], marker = "o", color = "w", markerfacecolor = "white",
                     markeredgecolor = "grey", markersize = 8, label = "P_adj >= 0.05")]

all_handles = color_handles + sig_handles

fig.legend(handles= all_handles, loc = "upper center", bbox_to_anchor = (0.5, 1.02),
          ncol = len(Categories_values) + 2, fontsize = 12, frameon = True)
    
plt.savefig("./Results/Categorical_DPs_to_Liver_Model2_Fiver_Cats_version3.pdf", bbox_inches='tight')
plt.show()


# # Plot the results of continous variable

# In[80]:


# I would like to combine all four Models into one data frame
results_continue_model1_corrected.loc[:, "Model"] = "Model1"
results_continue_model2_corrected.loc[:, "Model"] = "Model2"
results_continue_model3_corrected.loc[:, "Model"] = "Model3"
results_continue_model4_corrected.loc[:, "Model"] = "Model4"

results_cont_combine = pd.concat([results_continue_model1_corrected,
                                 results_continue_model2_corrected,
                                 results_continue_model3_corrected,
                                 results_continue_model4_corrected], 
                                 ignore_index=True)
results_cont_combine


# In[81]:


# Rename the Variables 
# Define a mapping dictionary
DPs_names = {
    "AHEI_2010_score_eadj_scaled": "AHEI",
    "hPDI_score_eadj_scaled" : "hPDI",
    "rDII_score_eadj_scaled": "rDII",
    "AMED_score_eadj_scaled": "AMED",
    "rEDIH_score_all_eadj_scaled": "rEDIH"
}


# mapping
results_cont_combine["Variable"] = results_cont_combine["Variable"].map(DPs_names)
results_cont_combine["Outcome"] = results_cont_combine["Outcome"].map(Outcome_name)
results_cont_combine.head()


# In[82]:


results_cont_combine["ifsig"] = np.where(results_cont_combine.p_adj < 0.05, "full", "none")
results_cont_combine["ifsig"].value_counts()


# In[83]:


DPs_to_Liver_Res_sen.shape
DPs_to_Liver_Res_sen.head()


# In[84]:


# Logistical_Regression_MASLD_v2

LG_MASLD = pd.read_csv("../03_Dylipidemia/Results/Logistical_Regression_MASLD_v2.csv")
LG_MASLD_select = LG_MASLD[~(LG_MASLD["Outcome"] == "Self-reported MASLD")].copy()
LG_MASLD_select

model_order = ["Model4", "Model3", "Model2", "Model1"]
LG_MASLD_select["Model"] = pd.Categorical(
    LG_MASLD_select["Model"],
    categories=model_order,
    ordered = True
)


# In[88]:


# plt.rcParams.update(
#     {
#         "font.size": 10,
#         "axes.titlesize": 12,
#         "axes.labelsize": 10,
#         "xtick.labelsize": 9,
#         "ytick.labelsize": 9,
#         "legend.fontsize": 9
#     }
# )

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

my_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
my_cmap = mpl.colors.ListedColormap(my_colors, name="my_cmap")


fig = plt.figure(figsize = (15, 16))

# top section: 2 plots side by side
# bottom section: 2 * 3
gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[1, 1, 1])

# top left figure A
ax_top_left = fig.add_subplot(gs[0, 0:3])

group_data = LG_MASLD_select
exposure_order = ["AHEI", "hPDI", "rDII", "AMED", "rEDIH"][::-1] # reverse the list 

group_data["Exposure"] = pd.Categorical(group_data["Exposure"],categories=exposure_order,ordered = True)
all_exposure = group_data["Exposure"].cat.categories  
exposure_position = {exp: idx for idx, exp in enumerate(all_exposure)}
models = group_data["Model"].unique()
colors = {model: my_cmap(i) for i, model in enumerate(models)}

# group by model to allow for dodging
for j, (model_name, model_data) in enumerate(group_data.groupby("Model", observed = True)):
        dodge_positions = j * 0.2 - 0.3
        model_color = colors[model_name]

        # convert string exposure to numeric positions
        y_positions = np.array([exposure_position[exp] for exp in model_data["Exposure"]]) + dodge_positions

        for idx, row in model_data.iterrows():
            y_pos = y_positions[model_data.index.get_loc(idx)]

            if row["pval"] <= 0.05:
                facecolor = model_color
            else:
                facecolor = "white"

            # plot each point with error bars
            ax_top_left.errorbar(
                x = row["OR"],
                y = y_pos,
                xerr = [[row["OR"] - row["lower_OR_ci"]],
                   [row["upper_OR_ci"]- row["OR"]]],
                fmt = "o",
                capsize = 4,
                capthick = 1.2,
                linewidth = 1.2,
                color = model_color,
                markerfacecolor = facecolor,
                markeredgecolor = model_color,
                markersize = 8)

            # add the text
            ax_top_left.text(
                x = 1.15, y = y_pos,
                s = f"{row["OR"]:.2f} ({row["lower_OR_ci"]:.2f}, {row["upper_OR_ci"]:.2f})",
                va = "center", ha = "left", fontsize = 8, color = "black"
            )
            
        # Add the text into each point
        # for idx, (or_val, y_pos, lower, upper) in enumerate(zip(
        #         model_data["OR"],
        #         y_positions, 
        #         model_data["lower_OR_ci"],
        #         model_data["upper_OR_ci"]
        #     )):
        #         # format 
        #         label_text = f"{or_val:.2f} ({lower:.2f},{upper:.2f})"

        #         ax_top_left.text(
        #             x = 1.25,
        #             y = y_pos,
        #             s = label_text,
        #             va = "center", # vertical alignment
        #             ha = "left", # horizontal alignment
        #             fontsize = 8,
        #             color = "black"
        #         )
            sig_handles = [Line2D([0], [0], marker = "o", color = "w", markerfacecolor = "grey",
                     markeredgecolor = "grey", markersize = 8, label = "P-value < 0.05"),
                       Line2D([0], [0], marker = "o", color = "w", markerfacecolor = "white",
                     markeredgecolor = "grey", markersize = 8, label = "P-value >= 0.05")]

            # add the legend
            ax_top_left.legend(handles= sig_handles, fontsize = 8, loc = "best")
            ax_top_left.set_xlim(left = 0.6, right = 1.3)
            # add reference line
            ax_top_left.axvline(x = 1, linestyle = "--", color = "grey")
            ax_top_left.set_yticks(np.arange(len(all_exposure)))
            ax_top_left.set_yticklabels(all_exposure)
            ax_top_left.set_xlabel("Odds ratio with 95% CI")


# In[105]:


results_cont_combine


# In[124]:


results_cont_combine["Variable"].unique() # DPs


# In[192]:


get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams["figure.dpi"] = 300

fig = plt.figure(figsize = (20, 20), constrained_layout = False)

gs_main = gridspec.GridSpec(2, 1, figure = fig, height_ratios=[1, 2], hspace = 0.15)

# top left figure A
# ax_top_left = fig.add_subplot(gs[0, 0:3])
gs_top = GridSpecFromSubplotSpec(1, 2, subplot_spec = gs_main[0], width_ratios = [1.4, 1], wspace = 0.6)
ax_top_left = fig.add_subplot(gs_top[0])
ax_top_right = fig.add_subplot(gs_top[1])

# Bottom
gs_bottom = GridSpecFromSubplotSpec(2, 3, subplot_spec = gs_main[1], wspace = 0.15)
axes_bottom = [fig.add_subplot(gs_bottom[i,j]) for i in range(2) for j in range(3)]

group_data = LG_MASLD_select
exposure_order = ["AHEI", "hPDI", "rDII", "AMED", "rEDIH"][::-1] # reverse the list 

group_data["Exposure"] = pd.Categorical(group_data["Exposure"],categories=exposure_order,ordered = True)
all_exposure = group_data["Exposure"].cat.categories  
exposure_position = {exp: idx for idx, exp in enumerate(all_exposure)}
models = group_data["Model"].unique()
colors = {model: my_cmap(i) for i, model in enumerate(models)}

# group by model to allow for dodging
for j, (model_name, model_data) in enumerate(group_data.groupby("Model", observed = True)):
        dodge_positions = j * 0.2 - 0.3
        model_color = colors[model_name]

        # convert string exposure to numeric positions
        y_positions = np.array([exposure_position[exp] for exp in model_data["Exposure"]]) + dodge_positions

        for idx, row in model_data.iterrows():
            y_pos = y_positions[model_data.index.get_loc(idx)]

            if row["pval"] <= 0.05:
                facecolor = model_color
            else:
                facecolor = "white"

            # plot each point with error bars
            ax_top_left.errorbar(
                x = row["OR"],
                y = y_pos,
                xerr = [[row["OR"] - row["lower_OR_ci"]],
                   [row["upper_OR_ci"]- row["OR"]]],
                fmt = "o",
                capsize = 4,
                capthick = 1.2,
                linewidth = 1.2,
                color = model_color,
                markerfacecolor = facecolor,
                markeredgecolor = model_color,
                markersize = 8)

            # add the text
            ax_top_left.text(
                x = 1.15, y = y_pos,
                s = f"{row["OR"]:.2f} ({row["lower_OR_ci"]:.2f}, {row["upper_OR_ci"]:.2f})",
                va = "center", ha = "left", fontsize = 12, color = "black"
            )
            
            sig_handles = [Line2D([0], [0], marker = "o", color = "w", markerfacecolor = "grey",
                     markeredgecolor = "grey", markersize = 12, label = "P-value < 0.05"),
                       Line2D([0], [0], marker = "o", color = "w", markerfacecolor = "white",
                     markeredgecolor = "grey", markersize = 12, label = "P-value >= 0.05")]

            # add the legend
            ax_top_left.legend(handles= sig_handles, fontsize = 12, loc = "best")
            ax_top_left.set_xlim(left = 0.6, right = 1.3)
            # add reference line
            ax_top_left.axvline(x = 1, linestyle = "--", color = "grey")
            ax_top_left.set_yticks(np.arange(len(all_exposure)))
            ax_top_left.set_yticklabels(all_exposure)
            ax_top_left.set_xlabel("Odds ratio with 95% CI")
            # add the annotation
            ax_top_left.text(-0.1, 1.1, "A", transform = ax_top_left.transAxes, fontsize = 16, fontweight = "bold", va = "top")
            
# top right figure
# ax_top_right = fig.add_subplot(gs[0, 3:6])
ax_top_right.axis("off")

df = pd.DataFrame({"Model" : ["Model 1", "Model 2", "Model 3", "Model 4"],
                  "Formulas" : ["Hepatic Health ~ Diet pattern score",
                               "Hepatic Health ~ Diet pattern score + MVs",
                               "Hepatic Health ~ Diet pattern score + MVs + BMI",
                               "Hepatic Health ~ Diet pattern score + MVs + Family history of diabetes + Family history of CVD"]})

col_labels = df.columns
table = ax_top_right.table(cellText = df.values, colLabels = col_labels, loc = "center", cellLoc = "left")

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.0)
table.auto_set_column_width(col = list(range(len(col_labels))))

row_heights = [0.15, 0.1, 0.1, 0.1, 0.1]

for row, height in enumerate(row_heights):
    for col in range(len(col_labels)):
        table[row, col].set_height(height)

# bold the header row
for col_index in range(len(col_labels)):
    table[0, col_index].get_text().set_fontweight("bold")

fig.text(0.80, 0.68,
        "** Multivariables [MVs]: age, sex, education, smoking, \n"
        "sleep duration, physical activity, vitamin use, hormone use",
        fontsize = 12, ha = "center", va = "top")

# bottom figure
# axes_bottom = [fig.add_subplot(gs[i, j*2:(j+1)*2]) for i in range(1,3) for j in range(3)]

outcomes = results_cont_combine["Outcome"].unique()
Categories_values = results_cont_combine["Model"].unique()[::-1] # Model1-4
dps_values = results_cont_combine["Variable"].unique() # DPs


for i, outcome in enumerate(outcomes):
    ax = axes_bottom[i]

    outcome_data = results_cont_combine[results_cont_combine["Outcome"] == outcome] 

    # Positions for each category within the group
    dps = dps_values[::-1]
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488'][::-1]

    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}
    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        
        cat_data = outcome_data[outcome_data["Model"] == cat]
        
        # is_sig = cat_data["ifsig"]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]

        p_values = []
        
        for idx, row in cat_data.iterrows():
            p_value = row.get("p_adj", 0.05)
            p_values.append(p_value)

        sig_threshold = 0.05
        is_significant = [p < sig_threshold for p in p_values]

        for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
            y_positions,
            cat_data["CI_lower"],
            cat_data["CI_upper"],
            cat_data["Coefficient"],
            is_significant
        )):
            ax.errorbar(x = coef,
                        y = y_pos,
                        xerr = [[coef - ci_lower], [ci_upper - coef]],
                        fmt = "none",
                        capsize = 4,
                        color = colors[j % len(colors)],
                        label = cat if n == 0 else None)
            # plot point
            if is_sig:
                # solid for significant results
                ax.scatter(coef, y_pos, s = 30,
                           c = colors[j % len(colors)],
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)
            else:
                ax.scatter(coef, y_pos, s = 30, facecolors = "none",
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)

        # Customize the plot
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
        ax.set_yticks(positions)

        if i == 0 or i == 3:
            ax.set_yticklabels(dps)
        else:
            ax.set_yticklabels([])
                
        # ax.set_yticklabels(dps)
        # ax.set_ylabel("DPs")
        # ax.set_xlabel('Coefficient')
        ax.set_title(f'{outcome}')
        ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
        # ax.legend()
        ax.set_xlim(-0.2, 0.2) # set x-axis limits

        # plt.tight_layout(rect = [0.03, 0.03, 0.95, 0.95])
        
        # fig.supxlabel("Coefficient (beta) with Confidence Interval", fontsize = 12)
        # fig.supylabel("Dietary pattern Scores", fontsize = 12)
        fig.text(0.5, 0.08, "Coefficient (beta) with 95% CI", ha = "center", fontsize = 10, fontweight = "normal")
        # fig.text(0.02, 0.5, "Dietary pattern Scores", rotation = "vertical", fontsize = 10, fontweight = "normal")

# Update the shared legend
from matplotlib.lines import Line2D

        # Category color legend
colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488'] 
color_handles = [Line2D([0], [0], marker = "o", color = "w", markerfacecolor = colors[j % len(colors)],
                       markeredgecolor = colors[j % len(colors)], markersize = 8, label = cat)
                for j, cat in enumerate(Categories_values[::-1])]

# Solid or hollow
sig_handles = [Line2D([0], [0], marker = "o", color = "w", markerfacecolor = "grey",
                     markeredgecolor = "grey", markersize = 12, label = "P_adj < 0.05"),
               Line2D([0], [0], marker = "o", color = "w", markerfacecolor = "white",
                     markeredgecolor = "grey", markersize = 12, label = "P_adj >= 0.05")]

axes_bottom[2].legend(handles= sig_handles, loc = "center left", 
                              bbox_to_anchor = (1.15, 0.94),
                             ncol = 1,fontsize = 12, frameon = True)

axes_bottom[0].text(-0.17, 1.1, "B", transform = axes_bottom[0].transAxes, fontsize = 16, fontweight = "bold",
                           va = "top")

# the legend for the whole figure
fig.legend(handles= color_handles, loc = "upper center", bbox_to_anchor = (0.5, 0.93),
          ncol = len(Categories_values) + 2, fontsize = 12, frameon = True)

# plt.subplots_adjust(wspace = 0.2)
fig.savefig("./Results/Figure1.pdf", bbox_inches = "tight", dpi = 300)
plt.show()


# In[90]:


# Bottom Subplots
# outcomes = results_cont_combine["Outcome"].unique()
# Categories_values = results_cont_combine["Model"].unique() # Model1-4
# dps_values = results_cont_combine["Variable"].unique() # DPs

# Plot for the continuous MODEL2
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()

# for i, outcome in enumerate(outcomes):
#     ax = axes_bottom[i]

#     outcome_data = results_cont_combine[results_cont_combine["Outcome"] == outcome] 
#uHHjht599121adNGzs7OMhgM2rlzp7VjPVIp6RnaceOe/nXiit47HKV/nbiiHTfuKSU944nmKA7Py3fffafAwEB5enqqZMmSql27toYMGaIbN25YO9ojk5SepmXnT6vnjrXy/3qFeu5Yq2XnTyspPe2JZykOz8zu3bvVtWtXVatWTY6OjqpUqZL++te/au/evdaO9sgkp6Tp612nNWH2Oo2YtEITZq/T17tOKznlyT4zxeF5+b3XXnutSOUFAAAAgAfZWTsAAABAYZd0cLfuzQ2RMT5OMthIxgzJYKOkfTsUu+jfch09WY4tn3/iuRwdHbVixQq1atXKrH3Xrl26evWqHBwcnnim/Prxxx81Y8YM1a1bV40aNdL+/futHemROvhLnOaduaaEtAwZJBklGSTt/yVOi368qdHeVeTj7vxEMxXl52Xs2LH69ddf1atXL9WtW1cXLlzQ/PnztWnTJh0/flyVKlWydsQ/ZMPlnzRw72bdTUmSjQzKkFE2Mmjt5XMadWiblrTqpC7V6j7xXEX5mTl37pxsbGz0xhtvqFKlSrp7966++OILtW7dWps3b9Zf//pXa0f8Q77/4SdN+3iz4hOSZWMwKMNolI3BoF0Hz+mD0G2aMLyznmte54lmKsrPy4MOHz6ssLAwOTo6WjsKAAAAADwUZv4CAADkIungbt2d+o6MCfGZDcYMs8/GhHjdnTpGSQd3P/FsHTt21OrVq5WWZj7La8WKFWrWrFmhLog1a9ZMd+7c0blz5/SPf/zD2nEeqYO/xGn6iSu6n/bbM/Jbe9bn+2kZmnbiig7+EvdEcxXl52XOnDn6+eefNWPGDA0ZMkTTp0/Xpk2bdOvWLc2fP9/a8f6QDZd/Uvcda3QvJUmSlPHbk5L1+V5KkrptX6MNl3964tmK8jMzZMgQrV+/XuPHj9fgwYM1ZswY7du3T+7u7po3b5614/0h3//wk96btVYJCZkrT2QYjWafExKSNW7mGn3/w5N9Zory85LFaDRq5MiRGjBggCpWrGjtOAAAAADwUCj+AgAA5MCYkqx7c0N++8aYQ6ffCjRzQ574EtD9+vXTnTt3tHXrVlNbSkqKwsPD9corr1j0T0hI0Ntvv61q1arJwcFB9erV0+zZs2X83b0lJydr9OjRcnd3l7Ozs7p27aqrV69mm+HatWsKDAxUxYoV5eDgIG9vb33++ed5Znd2dla5cuUKeMeFX0p6huaduSbp/4q9v5fVPu/MtSe6BHRRfl5at24tGxsbi7Zy5crp7Nmz+bn9QikpPU0D926WlPfzMnDv5ie+BHRRfmayU7JkSbm7u+vevXsPNb4wSE5J07SPN0vGPJ4ZozT94y1PdAno4vC8LFu2TKdPn9a0adPyPQYAAAAAChuKvwAAADlI/P67zKWecyr8ZjEaZYyPU+Le7U8m2G9q1qwpPz8/ffnll6a2r776SjExMerbt+/vIhrVtWtXzZ07V3/96181Z84c1atXT++8847FzNshQ4Zo3rx5at++vd5//32VKFFCnTp1srj+rVu35Ovrq23btmn48OH64IMPVKdOHQ0ePLjIz6x7WHtvxyohLSPHokwWo6SEtAztvR37JGJJKn7PS3x8vOLj4+Xm5lbgsYXF6qhI3U1JytfzcjclSeFRkU8ilklxeGZiY2MVHR2tyMhIvffeezp9+rTatm1b8BejkNixP1LxCcn5embiEpK088CPTyKWpKL/vMTFxWns2LF67733isQsZQAAAADICcVfAACAHCQf2JW5x29+GGyUvH/nY82TnVdeeUXr169XYmKiJGn58uV64YUX5OHhYdZvw4YN2r59u/75z3/qP//5j958801t2LBBAQEB+uCDD3T+/HlJ0okTJ/TFF19o2LBhWr58ud58802tWbNGDRs2tLj2+PHjlZ6ermPHjik4OFhvvPGG/ve//6lv376aPHmyKdOfyYHbcTLks6/ht/5PUnF6XubNm6eUlBT16dPnIV8N61t/+Zxs8vnE2MigdZfPPeZElor6M9O7d2+5u7urfv36+ve//63XX39dwcHBj+CVsY7vf/hJNoZ8PjMGg/YcerLPTFF+XqZMmSInJyeNHj36Eb0aAAAAAGAdFH8BAABykBEb8397/ObFmKGMuJjHGygbvXv3VmJiojZt2qS4uDht2rQp2+U1t2zZIltbW40cOdKs/e2335bRaNRXX31l6ifJot9bb71l9r3RaNSaNWvUpUsXGY1GRUdHmz7+8pe/KCYmRkePHn2Ed1o0xKWm5zkjL4vxt/5PUnF5Xnbv3q2QkBD17t1bL774Yr7HFTZ3kpOU9zzxTBky6tfkpMecyFJRf2bef/99ffvtt1q8eLF8fX2VkpJisSdtURITl2ja2zcvGUajYuOe7B/hFNXn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hx9+0Pz58/Xmm2+axterV0+BgYHatm2bXn755VyvFRoaqq5du8rR0dGsfcyYMWrcuLG+/fZb2dllvo11cXHR9OnTNWrUKHl5eeV4zg0bNujSpUvatGmTOnXqZGovV66cpkyZohMnTuiZZ56Rvb299u7da/a+8bXXXlPNmjVNBeB27drlmj874eHh2rdvn1avXq2AgABJUu/eveXp6alJkyZpxYoVOY69ceOG5syZo//3//6fli5damr39PTUiBEjtHHjRnXp0sVsTNOmTdW/f/9cM9nY2CggIEBLly5VSEjIQxW1AQAAAAAAULgUq2WfFy9erH79+snLy0uVKlUyfYwcOVLXr1+X0WjM97n69eun0qVLmz6yZhIDgDVsuPyTuu9Yo3spmUuOZsho9vleSpK6bV+jDZd/slpGAIVbeHi4bG1tNXToUFObo6OjBg8erP379+vKlSu5jo+NjVXZsmVNhV9JsrOzk5ubm5ycnMz6SVLFihXNxleuXFmSzPpm5+LFizp58qRFgTUiIkIREREaOnSoqfArScOGDZPRaFR4eHie+fOTy97ePts/GMwqWJ89ezbX6+QkPDxcFStWVI8ePUxt7u7u6t27t/73v/8pOTk5x7H79+9XWlqa+vbta9ae9f3vl3POkpCQoJSUlFxzvfTSS7p06ZKOHz+ezzsBAAAAAABAYVasir9btmxRaGioRbubm5sSExP14osv6t69exbHJ0+erPXr15u+j4qKUmJiouLj400fD+6FZjQa1aRJk8dwBwBgKSk9TQP3bpYk5fQnLFntA/duVlJ6/va+BPDncuzYMXl6esrFxcWs3cfHR5LyLP75+/vrzJkzCg4O1s8//6zz58/rn//8pw4fPqygoCBTv+bNm6tUqVIKDg7W9u3bde3aNe3atUtBQUFq0aJFnrNms2YgN23a1CJ/1vkf5OHhoapVq5qO56R169aysbHRqFGjdODAAV29elVbtmzRtGnT1L1791xnDUsyvRd0c3PLtV9Ojh07pqZNm1rsYezj46P79+/r3LlzOY7NKgz/vnBesmRJSdKRI0csxoSEhKh06dJydHRUixYt9O2332Z77mbNmkmS9u7dm/+bAQAAAAAAQKFVrIq/AFAcrY6K1N2UpBwLv1mMku6mJCk8KvJJxAJQxNy4ccM0y/VBWW3Xr1/PdXxwcLB69+6tadOmqW7duqpTp47ef/99rVmzxmw2q5ubm1auXKmYmBi1bdtWVatWlb+/vzw8PLR9+3azWbvZiYzM/B1Wq1Yti/wP5v39PeSVv0GDBlq0aJEiIiLk5+enatWqqVOnTmrbtq1Wr16d61hJmjlzplxcXNShQ4c8+2bnj7z+9erVk2RZoN2zZ48k6dq1a6Y2GxsbtW/fXrNmzdKGDRs0d+5c3b59Wx06dNDmzZstzl2lShXZ29srIiKi4DcFAAAAAACAQqdY7fkL4M/pfNxdea//zNoxHpsrCbEF6j/swLf616kDjylN4XM+7q6eci5r7RiPRK1atXTq1CnT18CjlJiYaLZkc5asfXUTExNzHe/g4CBPT08FBASoR48eSk9P16JFi9S/f39t3bpVvr6+pr7u7u565plnNHz4cHl7e+v48eOaOXOmBg0alGeh9c6dO7Kzs1Pp0qUt8mflyO4espZ1zk2VKlXk4+Ojjh07qkaNGtqzZ48+/PBDubm5afbs2TmOmz59urZt26YFCxbI1dU1z+tk54+8/k2bNlXLli01Y8YMValSRW3atNHZs2f197//XSVKlDAbW716dX3zzTdm4//f//t/atCggd5++22z/Y6zlC1bVtHR0Q91XwAAAAAAAChcKP4CQCGXbsx4rP1ReDzzzDN65plnrB0DxZSTk1O2+8omJSWZjudm+PDhOnDggI4ePWpaurh3797y9vbWqFGjdPDgQUnShQsX1KZNGy1dulQ9e/aUJHXr1k01a9bUwIED9dVXXz3U7NmsfDndQ1759+7dq86dO+vAgQOmpaO7d+8uFxcXhYSEKDAwUA0aNLAYt3LlSk2YMEGDBw/W3//+9wLnfjD/H3n916xZoz59+igwMFCSZGtrq3/84x/atWuXfvzxx1zHlitXToMGDdL777+vq1evqmrVqmbHjUajDAZDQW4HAAAAAAAAhRTFXwBF3lPOZXWm+xBrx3hseu5Yq/WXf1JGngs/SzYy6K9VamtNmx559i0uivOsb+BRqly5stnywFmyllP28PDIcWxKSooWL16soKAgsz1rS5QooQ4dOmj+/PlKSUmRvb29wsLClJSUpM6dO5udo2vXrpIyi7C5FX/Lly+vtLQ0xcXFydnZ2Sx/Vt5q1apZ3EPW3sU5+fTTT1WxYkWLPYO7du2qyZMna9++fRbF361bt2rAgAHq1KmTPvnkk1zPn5fKlSubXuvfZ5dyf/2lzFnL33//vX766SfdvHlTdevWVaVKleTh4SFPT888r5/1mv36668Wxd979+499F7GAAAAAAAAKFzY8xcACrnu1T3zVfiVpAwZ9XL1vIsAKJwiIyMVEhKikJAQ076nwKPSpEkTnTt3zmJ55KwZu02aNMlx7J07d5SWlqb09HSLY6mpqcrIyDAdu3XrloxGo0Xf1NRUSVJaWlquOb28vCRJFy9etMgvSYcPHzZrv379uq5evZpr/qxcOeXPLtfBgwf18ssvq3nz5lq1alWeexXnpUmTJjp69KgyMsxXZzh48KBKliyZrwKuJNWtW1fPP/+8KlWqpIiICN24cUPt2rXLc9yFCxckZS7J/aBr164pJSVF9evXz+edAAAAAAAAoDCj+AsAhVyvml4qa++ovBbkNEgqa++ogJpeTyIWHoOIiIhsvwYehYCAANM+vVmSk5MVGhqqli1bms2mvXz5stkfIFSoUEGurq5at26dUlJSTO3x8fHauHGjvLy8TMsWe3p6ymg0atWqVWbX//LLLyUpz6XN/fz8JFkWeb29veXl5aVFixaZFXEXLlwog8GggIAAU1tMTIwiIyMVExNjavP09NStW7e0c+fOPHOdPXtWnTp1Us2aNbVp06Y8l2TOj4CAAN26dUtr1641tUVHR2v16tXq0qWL2X7A58+f1/nz53M9X0ZGhoKCglSyZEm98cYbpvZffvnFou+1a9f0+eefq3HjxqYZ1FmOHDkiSXr22Wcf6r4AAAAAAABQuLDsMwAUco62dlrSqpO6bV8jg5TtHOCswvCSVp3kaMuvdgCWWrZsqV69emncuHG6ffu26tSpoyVLligqKkqLFy826ztgwADt2rVLRmPmbxxbW1uNGTNGEyZMkK+vrwYMGKD09HQtXrxYV69e1RdffGEaO3DgQM2ePVuvv/66jh07Jm9vbx09elSfffaZvL299fLLL+eas3bt2mrYsKG2bdtm2t82y6xZs9S1a1e1b99effv21enTpzV//nwNGTLEbObqunXrNGjQIIWGhmrgwIGSMvcsDg0NVZcuXTRixAjVqFFDu3bt0pdffqmXXnpJLVu2lCTFxcXpL3/5i+7evat33nlHmzdvNsvw1FNPmQrUkuTv72/2WuUkICBAvr6+GjRokCIiIuTm5qYFCxYoPT1dISEhZn3btm0rSYqKijK1jRo1SklJSWrSpIlSU1O1YsUKHTp0SEuWLFH16tVN/YKCgnT+/Hm1bdtWHh4eioqK0qeffqqEhAR98MEHFrm2bt2q6tWrs984AAAAAABAMUGFAACKgC7V6mp9m54auHez7qYkyUYGZcho+uxq76glrTqpS7W61o4KoBBbunSpgoODtWzZMt29e1eNGzfWpk2b1Lp16zzHjh8/XrVq1dIHH3ygkJAQJScnq3HjxgoPD1fPnj1N/cqXL6/Dhw9r4sSJ2rhxoz755BOVL19egYGBmj59uuzt7fO8VmBgoCZOnKjExESzWbedO3fW2rVrFRISohEjRsjd3V3vvfeeJk6cmOc569WrpyNHjmjChAn64osvdPPmTXl4eGjMmDFmxdc7d+7oypUrkqR3333X4jx/+9vfzIq/8fHxqlSpUp7Xt7W11ZYtW/TOO+/oww8/VGJiolq0aKGwsDDVq1cvz/HPPPOM5s2bp+XLl8vGxkY+Pj767rvv1KZNG7N+7du31yeffKKPP/5Yd+/elaurq1q3bq0JEyaoadOmZn0zMjK0Zs0aDR48WAZDXutLAAAAAAAAoCgwGPOapoA/LDY2VmXKlFFMTIxcXFysHQdAEZaUnqbwqEitu3xOvyYnqZyDo16u7qmAml7M+C0G1q5dq1OnTkmSGjVqpB49elg5EWAdMTExql27tmbOnKnBgwdbO06O4uLiVK5cOc2bN09vvvmmteMU2Pr16/XKK6/o/PnzFstB54b3tgAAAAAAAIUXlQIAKEIcbe3U/6mG6v9UQ2tHAYDHpkyZMgoKCtKsWbM0aNAg2djYWDtStnbv3q0qVarotddes3aUhzJjxgwNHz68QIVfAAAAAAAAFG7M/H0CmB0BAMgPZv4CKAp4bwsAAAAAAFB4Fc5pFAAAAAAAAAAAAACAAqH4CwBAIVG1atVsvwYAAAAAAAAAID/Y8xcAgELCx8dHPj4+1o4BFAozZ87U559/roiIiEK7529Rlpqaqtq1a2vcuHEaNmyYteMAAAAAAADgEeFf0gAAAP4kkpOTNXbsWHl4eMjJyUktW7bU1q1b8z1+27ZtatOmjdzc3OTq6iofHx8tW7Ys1zHff/+9DAaDDAaDoqOj83Wd2NhYzZgxQ2PHjrUo/G7YsEFNmzaVo6OjqlevrkmTJiktLS3Pc06ePNmUI7uPvXv3mvVftWqVfH195erqqvLly+uFF17Q5s2b85U/J9euXVPv3r3l6uoqFxcXdevWTRcuXMjX2NTUVIWEhKh27dpycHBQ7dq1NXXq1GzvPT8/5xIlSugf//iHpk2bpqSkpD90XwAAAAAAACg8DEaj0WjtEMVdbGysypQpo5iYGLm4uFg7DpBv/2/0Z5KkZXOHWDnJ45eckqYd+yP1/Q8/KSYuUWWcndSqRV218fOSg/2fb5GEP9PPvjCJjIzUypUrJUl9+vSRl5eXlROhuOnXr5/Cw8P11ltvqW7dugoLC9MPP/ygHTt2qFWrVrmO3bBhg7p37y4/Pz/169dPBoNBq1at0u7duzVnzhyNHj3aYkxGRoaaNWumn376SQkJCfrll1/k5uaWZ8558+Zp0qRJunXrlhwdHU3tX331lTp16iR/f3/169dPp06d0scff6yhQ4dq4cKFuZ7z5MmTOnnypEX7e++9p/j4eN28eVP29vaSpI8++kgjR45Up06d1LlzZyUlJSksLEwnTpzQmjVr1KNHjzzv4ffi4+PVtGlTxcTE6O2331aJEiU0d+5cGY1GHT9+XOXLl891fJ8+fbR69WoFBgaqefPmOnDggJYsWaLXXntNixYtMuub35/zvXv3VLFiRS1cuFCBgYH5vhfe2wIAAAAAABReFH+fAP6BDEXVn6UA+P0PP2nax5sVn5AsG4NBGUaj6XPpUg6aMLyznmtex9oxn6g/y8++sFm7dq1OnTolSWrUqNFDFZiAnBw6dEgtW7bUrFmzNGbMGElSUlKSGjZsqAoVKmjfvn25jm/fvr3OnDmjCxcuyMHBQZKUlpYmLy8vlSpVSidOnLAY88knn2jChAnq37+/Pvjgg3wXf59++mk1btzYYlaxt7e3SpQoocOHD8vOLvMPcyZMmKDp06crIiKiwH8wceXKFdWoUUNDhgwxK6B6enrK1dVVBw8elMFgkJT5fq5KlSp68cUX9b///a9A15Eyl7EeO3asDh06pBYtWkjK/IOPhg0bKigoSNOnT89x7A8//CAfHx8FBwdrypQppvYxY8Zozpw5On78uBo3biyp4D/nLl26KCYmRrt37873vfDeFgAAAAAAoPBi2WcAf2rf//CT3pu1VgkJyZKkjN/+Hibrc0JCssbNXKPvf/jJahkB4FEIDw+Xra2thg4dampzdHTU4MGDtX//fl25ciXX8bGxsSpbtqyp8CtJdnZ2cnNzk5OTk0X/X3/9VRMmTNCUKVPk6uqa75wXL17UyZMn1a5dO7P2iIgIRUREaOjQoabCryQNGzZMRqNR4eHh+b5Gli+//FJGo1GvvvqqWXtsbKwqVKhgKvxKkouLi0qXLp3tveZHeHi4WrRoYSr8SpKXl5fatm2rVatW5Tp2z549kqS+ffuatfft21dGo9G0YkDWdQryc37ppZf0/fff69dff32o+wIAAAAAAEDhQvEXwJ9Wckqapn28WTJKOS2BYPztf6Z/vEXJKXnvKQkAhdWxY8fk6elpMVPTx8dHknT8+PFcx/v7++vMmTMKDg7Wzz//rPPnz+uf//ynDh8+rKCgIIv+wcHBqlSpkl5//fUC5cyamdq0aVOL/JLUvHlzs3YPDw9VrVrVdLwgli9frmrVqql169Zm7f7+/vr666/10UcfKSoqSpGRkXrzzTcVExOjUaNGFfg6GRkZOnnypEV2KfP1P3/+vOLi4nIcn5yc+QdKvy88lyxZUpJ05MgRU1tBf87NmjWT0WjMc+Y3AAAAAAAAigaKvwD+tHbsj1R8QnKOhd8sRklxCUnaeeDHJxELAB6LGzduqHLlyhbtWW3Xr1/PdXxwcLB69+6tadOmqW7duqpTp47ef//9bPfAPXnypD799FPNmTNHtra2BcoZGRkpSapVq5ZF/gfz/v4e8sr/e2fOnNHJkydN+xc/6MMPP5S/v79GjhypWrVqqX79+lq1apW+++47+fn5Feg6UuYs6OTk5Id+/evVqydJ2rt3r1l71ozga9eumdoK+nOuXbu2pMyZ1QAAAAAAACj67PLuAuDP7NrNe6b9X4ub23dynmWVnTmffasv1u1/TGkKl2s376lKJVdrxwDwCCUmJpot2ZzF0dHRdDw3Dg4O8vT0VEBAgHr06KH09HQtWrRI/fv319atW+Xr62vqO3LkSHXo0EHt27cvcM47d+7Izs5OpUuXtsiflSO7e4iNjS3QdZYvXy5JFks+S5kzauvVq6eqVauqc+fOiouL09y5c9WjRw/t2bNHdeoUbB/4vLI/2Cc7HTt2VI0aNTRmzBiVLFlSzZo108GDBzV+/HjZ2dmZjS3oz7ls2bKSpOjo6ALdEwAAAAAAAAonir8A/rTS0zMK1j+jYP0BoDBxcnIyLR/8oKSkJNPx3AwfPlwHDhzQ0aNHZWOTuXhM79695e3trVGjRungwYOSpJUrV2rfvn06ffr0I88vKcd7KMhevEajUStWrFDDhg3VuHFji+O9evWSnZ2dNm7caGrr1q2b6tatq/Hjx5vtsfsosj/YJzuOjo7avHmzevfurZ49e0rKLCTPnDlT06ZNMyuUF/TnbPxtj/vfz34GAAAAAABA0UTxF0CuqlRy1bK5Q6wd47GYMHud9hz6SRnGvBZ+lmwMBvk2qa2pY15+Asmsr7jO9gb+zCpXrmy2PHCWrOWUPTw8chybkpKixYsXKygoyFT4laQSJUqoQ4cOmj9/vlJSUmRvb6933nlHvXr1kr29vaKioiRJ9+7dkyRduXJFKSkpuV6rfPnySktLU1xcnJydnc3yZ+WtVq2axT1k7WmbH3v37tWlS5f0r3/9y+LYhQsX9PXXX2vRokVm7eXKlVOrVq0sll7Oj3LlysnBwcH0Wv8+u5T76y9J3t7eOn36tCIiInT37l01aNBATk5OGj16tF544QVTv4L+nO/evStJcnNzK9hNAQAAAAAAoFBiz18Af1qtWtTNV+FXkjKMRj3v4/mYE+HPrmrVqtl+DTwKTZo00blz5yyWR86asdukSZMcx965c0dpaWlKT0+3OJaamqqMjAzTsStXrmjFihWqVauW6eODDz6QJDVt2lQdO3bMNaeXl5ck6eLFixb5Jenw4cNm7devX9fVq1dzzf97y5cvl8Fg0CuvvGJx7NatW5KU472mpaXl+zpZbGxs1KhRI4vsUubrX7t2bbNCd04MBoO8vb3VqlUrlStXTjt27FBGRobatWtn6lPQn3PW61y/fv2C3hYAAAAAAAAKIYq/AP602vh5qXQpB+W10KVBknMpR/n71nsSsfAn5uPjo0mTJmnSpEkFmsUI5EdAQIBpn94sycnJCg0NVcuWLc1m016+fFmRkZGm7ytUqCBXV1etW7dOKSkppvb4+Hht3LhRXl5epuWE161bZ/HRp08fSdLSpUs1d+7cXHP6+flJsizyent7y8vLS4sWLTIrzC5cuFAGg0EBAQGmtpiYGEVGRiomJsbi/KmpqVq9erVatWql6tWrWxyvU6eObGxstHLlStOSyJJ09epV7dmzR88880yu+XMSEBCgH374wey+fvzxR23fvl29evUy6xsZGanLly/ner7ExEQFBwercuXK6tevn9l1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4E4BAPhzqFChglq1amX24evrazp+9OhR9erVS5UqVZK9vb0qVaqkgIAA0x/6ZQkLCzPtrzt58mR98sknqlevnkqUKKFVq1YpKirKdNzf3187duxQs2bN5OTkpKZNm2rnzp2SpIULF6p27dpydHTUc889pxMnTphdp2bNmqbzPGjgwIGm9qxz5eTatWsKDAzU008/LTc3N5UoUULlypXTiy++qPXr11v0//TTT9W8eXOVLl1aDg4OqlKlitq1a2fx/+sAAPBnUCz3/K1Zs6bmzZun7t27WzsKgIfk4+6s956upnlnrikhLUMGZS7xnPW5pF1mgdjHnSWfASC/li5dquDgYC1btkx3795V48aNtWnTJrVu3TrPsePHj1etWrX0wQcfKCQkRMnJyWrcuLHCw8PVs2dPs75r1qwxmzV77NgxHTt2TJJUtWpVNW7cONdrBQYGqmfPnrpy5YrZjOSmTZtq27ZtGjt2rEaPHi1nZ2cNHjw4zz1tH7R8+XJJynUpZCcnJ23fvl1BQUH6/PPPlZCQoCZNmmjz5s36y1/+YtY3Pj5elSpVyvO6tra22rJli9555x19+OGHSkxMVIsWLRQWFqZ69erlOX7hwoV6+umntXjxYlOxt0WLFlq6dGm2P7/ly5erRIkSZjOify8jI0Nr1qzR4MGD81XARtHRs2dPLV++XFevXtXXX3+tjh07mmb99+vXz2L/aEkaO3asxT+OXrx4UR9//LHWrFmjffv2qVatWpKkffv2qUOHDqaVAC5cuKDhw4fn+H/bR48eVdu2bc32Jfzll1+0atUqbdmyRd999518fHwexa0DAPCns2HDBgUEBCg1NdXUduvWLa1Zs0YbNmxQeHi4unbtajFu2bJlunDhQo7n/fnnn9WxY0clJSVJynxP37FjR7355ptmfxC5b98+de/eXT/99JPs7B7dPzVfuXJFoaGhZm13797Vjh07tGPHDi1ZskQDBgww3csbb7xh1vf69eu6fv26IiMjFRQU9MhyAQBQFPzppsvVrFlTTk5OKl26tNzc3NSlSxfTsnphYWGytbVV6dKl5eLioipVqqhnz57avXu32Tn8/f01b948K6QH8u9jv6f0sd9T1o7xh7R0d9aS5z012ttDvu7OauhaUr7uzhrt7aElz3sWq8Jvcfh5ASj8HB0dNWvWLN24cUNJSUk6dOiQRTFT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Vq1axdSpU5XXmt27d3P//n3Gjx+v1N21axcJCQls375dbo4SQgghhA5J/gohsj1HSxsuenbJ6jDeqZbh69hwIwJ1hgs/gwEqGhYuwVqPjPfK/NjltBner+v48eNs374dgEaNGil7sQrxLhQsWDDV5YG1yymnliTSSkhIYMmSJQwZMkTnw24jIyMaNWrEvHnzSEhI0PmQHeDGjRscOHCAbt26KTOUMpIvXz6SkpKIjY3V+bBYuzzynTt39D60u3PnToa/LwsXLsTe3l5vz+DmzZszduxYDh8+/F6TvwULFlSu9csyc/0hZdbywYMHiYiI4O7du5QqVYoCBQpQqFAhnJycMjy/t7c3v/zyC/v379f5kBRSZtFpPyAVHycfHx/69evHli1baNiwIVu2bNGb8au1bds2DA0N+eGHH3TKBw4cSFhYGNu3b6d3795s27YNQK9ev379CA0NVR5rNBrWrl2Lj48PGo1GZ6bn119/zcqVKzl16hRffvnlG4/v/v37tG3bFgcHhwz3GBfZx88//5zq69PAgQN1lpFdt24darUaHx8fnZ+fAgUKUKpUKcLDw3WSv7ly5aJ79+7KY2NjY7p3706PHj04efIk1atXx9DQEENDQyBlVm1UVBRqtZpq1aqlupRqZmzYsAG1Ws3o0aP19m/P7A00a9euxcrKSq/81WTemjVrsLa2pn79+jrXRLskenh4uJLEfDmBFxsbS3x8PLVq1WLhwoVcvnyZzz//PNNjfNX333+v87hWrVosW7Ysw3b37t1j9OjRrFy5kpiYGPLkyUPDhg1p3rw51apVIyEhgXXr1rFr1y7279//xvEBykohsbGxadZ5nev5Jjw9PZXEL4CLiwuurq5s27ZNSf6+mnx9+vSpsmrKqVOnqFWr1hud+01egzt06JDmDTlv8pxrZ+Zv2bKFzz//PNW/+dauXYtKpWLMmDF6x7S/P9r3He2NHVoDBw5k+vTpbN26VSf56+DgoPc3zZo1a6hVq5beqib16tXjp59+Yv/+/emuRvP06VPlxjxAWRb81ZUMjIyMXuvGhbZt29KqVSvu3r3LhQsXuHv3bro/c76+vvTt25f9+/crS4+HhoZibGyss8e19tpv3LiRjh076r02CSGEEOLTJclfIYTIAp5FnVh342qm6qrR0KJoxh/si4/fy/so3rp1S5K/4p2qXLky4eHh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K6bOnTubl/n5+al79+76/vvvMzyPNWvWqH379ipTpox5WcuWLVWpUiWtXLky1W2ioqIUFxeXPScA2Ljo6OhUvwayy4ABAzR9+nQ9/fTTmjFjhuzt7dW2bVv9/vvvGW67du1atWrVSnFxcXrrrbc0ZcoUubq6ql+/fvroo48s2p4/f1716tXTxo0bNWTIEM2ZM0fPPfdcimkC0jJ//nwlJCSoV69eFss3btyoTp06ydvbWzNnzlSnTp00efJkDR8+PMN9du7cWYsWLUrxKl26tHx8fFSvXj1z27Fjx2r48OF66KGHNGPGDHXp0kUzZ860eH5mVWRkpFq0aKFff/1Vr7/+uiZOnKgDBw6oWbNmun79eobb9+nTRxMnTtRjjz2mGTNmqGnTpho/fryGDh1q0e7s2bOKiIhQ//79NWPGDI0fP16S1LFjR33++ecWbZ966il5enpqzpw5931eAAAAAAAAyFsY9tkGJCYmav/+/apRo0aGPX7v5erqqho1amj//v1q0qRJpgvH2eHUqVOSpCJFiqS6vm7duipfvrxWrlyp/v37W6xbsWKFfHx81Lp1a0nSnj179Oeff6pnz54qVaqUQkJC9Omnn6p58+Y6evRotgwrnZSUpI4dO+r333/XoEGDVLVqVf3111/66KOPdPLkSX333Xfpbt+7d29dvXpVy5Yt02effaagoCANHDhQffr0SfMaSNLt27cz1WPY3t5ePj4+6bY5cOCAateunaIXVf369fX555/r5MmTCgoKSnXbixcv6urVq6pbt26KdfXr19eGDRtSLJ84caLGjBkjk8mkh4Jq6oP331OrVq0yPBdbERuXoK07juv3PcEKi4iWl4erGtcLVIuGVeTsxK/fvqO+lCQt+ug5K0cCFAy7d+/W8uXL9f7772v06NGSpH79+umhhx7SK6+8oj///DPd7WfNmiV/f3/98ssvcnZ2liQNHjxYVapU0cKFCzVq1Chz28GDB8vBwUF79uxJ9xmXlgULFqhjx45ycXGxWD569GjVqFFDP/30kxwc7vwe9fT01NSpU/XSSy+pSpUqae6zRo0aqlGjhsWy8+fP68KFC3ruuefMUxdcunRJ06dPV9++ffX111+b21aqVEnDhw/XunXr1KFDhyyf05w5cxQcHKzdu3ebC81t2rTRQw89pA8//FBTp05Nc9s9e/Zo5cqVGj9+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q0iRIrp582a6cRYpUkQJCQmKiIiQh4eHRfzJ8d472salS5fMcxdnxh9//KGzZ8/qnXfeSXV9yZIl9fvvvys4OFiXL19WYGCgihcvrhIlSqT7Qam0JI+Ocb+jb0hS9erV9ffff+vo0aO6efOmqlWrJldXV40aNUrNmjXLMIauXbtq8ODBOnnyZIrhpW/dumXVDwECAAAAAAAg+1D8tQH29vaqXbu2du7cqcceeyxLPV+io6N1+PBhPfrooxZDHOc1PXr00JtvvikXFxcNHDjQ4o1nSWrfvr0WLVokLy8vVatWTTt27NCWLVtUpEiRDPf93HPPafXq1XryySfVvXt3nTp1SosXL7YoekpS3759tXLlSg0ZMkRbt25Vo0aNlJiYqOPHj2vlypXatGlTmsNaS9Kvv/6q+Ph4zZkzR71795aXl1emzj075/zt2rWrHnnkET3zzDM6evSofH19NWfOHCUmJqboXfz4449LkkVv4tdff12rVq1SixYt9NJLLykyMlLvv/++goKC9Mwzz5jbvfLKKzp16pQef/xxlShRQgcOH9XyJV8rOvq2Pp0zW+VLpz08dUxMjBJiwjTttW7moTDzkzc++FbbdwcryUhrxt//sTOZ9Eit8po8+j+5EFne03fUl9YOAShQatWqpa1btyo8PFyenp7m5ck9dtMbweH69etKSEhI9UNV8fHxSkpKMq+rU6eOpJTz0MbFxSk0NDTDKQqSe6ueOXPGolducnx79+61KPT++++/unDhQooPZqVnyZIlMplM6t27d7rtAgMDzR+OO3r0qC5duqQBAwZk+jjJ7OzsFBQUlOroGbt27VL58uUtCt1pMZlMql79f1MFbNiwQUlJSWrZsmWG2yYPDR0WFmax/OLFi4qLi1PVqlUz3AcAAAAAAADyPruMmyA/qFu3rkwmk1avXp2p3q7SnTltV69eLXt7e/MbtXlVjx49lJSUpNu3b6tHjx4p1s+YMUP9+vXTkiVL9N///leXLl3Sli1bUh0G8V6tW7fWhx9+qJMnT2rkyJHasWOH1q9fn2JOWjs7O3333Xd699139ddff2n06NGaOHGi9uzZo5deeinDnkCjR4/Wvn379MILL2S68Jvd7O3ttWHDBvXo0UOffPKJxowZI19fX/3yyy8pegGlpnTp0vr1119VoUIFvfrqq5o2bZratm2rzZs3W/Qa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SVJ8fLzefvttHTp0SN99951CQkLMBd7SpUtrzZo1kqQTJ07o0qVLmjFjRqr7HjBggPbu3au1a9dqx44dMgxDbdu2VXx8fKrtmzVrpt9//12JiYmSpF9//VW+vr7atm2bJOnixYs6deqUmjdvnmLbGTNmqGHDhnr++efNvXBLly5tXj9u3Dh9+OGH2rt3rxwcHPTss8+meU1iY2Pl5OQkO7v/PV6T7+Hvv/9u0fbFF1+Ur6+v6tevr/nz55vvpyTVrFlTmzdvVkJCgn7++WfVqFFDkjRkyBBNmzYtU72kN23apNDQUL3yyiuprs+Jns8AAAAAAAAAAOtysHYAAABLwcHBMgxDVatWTXV91apVdfPmTV27dk23bt3Sxo0btXv3btWrV0+SNG/ePIttt2zZouPHj2vTpk0qUaKEJGnq1Klq06aNuc25c+cUGBioxo0by2QyqWzZshnGWahQIX355ZdycnIyL7u7MFq+fHl98sknqlevniIjI+Xu7q7ChQtLkooWLZpm8TE4OFhr167VH3/8oUcffVSStGTJEpUuXVrfffedunXrlmKbJk2aKCIiQgcOHFCdOnX022+/acyYMfruu+8k3ZlruGTJkqpYsWKKbb28vOTk5CQ3NzcVL148xfopU6aoWbNmkqRXX31V7dq1U0xMjEVP3WSPPfaYXn75Zb3//vt66aWXFBUVpVdffVWSdOnSJXO7SZMm6bHHHpObm5t++uknDR06VJGRkRoxYoQk6csvv9TQoUP1wQcfqFGjRnrttde0aNEiubm5qV69emrdurVOnTqlnj17avLkyWleR0mqUqVKqusBAAAAAAAAALaHnr8AkEfd3RM0LceOHZODg4Pq1KljXlalShWLwuqxY8dUunRpc+FXkho2bGixnwEDBujgwYOqXLmyRowYoZ9++inDYwcFBVkUfqU7QyR36NBBZcqUkYeHh7loeu7cuQz3d+85NWjQwLysSJEiqly5snn443t5e3urZs2a2rZtm/766y85OTlp0KBBOnDggCIjI/Xrr7+aY8mq5F63kuTv7y9JFj2r71a9enV99dVX+vDDD83F5HLlyqlYsWIWvYHHjx+vRo0a6eGHH9bYsWP1yiuv6P3337fYz6+//qqzZ89q6dKlio+P14QJEzRr1iwNHz5cjz76qA4dOqRvvvlG69atSzWWzPz7AQAAAAAA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fffd3Agz08gtAAD4n/yeX9jZ2eWLZ/v9vCcCAAAyh2GfATywe+fD+eCDD2QymXT27NkUbV977TU5OTnp5s2b5mW7du3Sk08+KS8vL7m5ualZs2b6448/LLZ76623ZDKZdPToUfXu3Vs+Pj5q3LhxjpzPjRs3NHr0aAUFBcnd3V2enp5q06aNDh06lKLtzJkzVb16dbm5ucnHx0d169bV0qVLzTGPGTNGklSuXDnzkNPJQxilNr/NrVu3NGrUKAUEBMjZ2VmlSpVSv379FBoaKinltR4wYIBmz54tSRbDWhuGoYCAAD311FMpYo6JiZGXl5cGDx6cHZcLAACryYs5yNNPP63jx49r//79KdYtXbpUJpNJvXr1kiTFxsZqwoQJqlixopydnVW6dGm98sorio2NzfDcT58+rW7duqlw4cJyc3PTI488oh9++CFFu5iYGL311luqVKmSXFxc5O/vr86dO+vUqVPmNnfP+Zte/tKsWTPVrFkz1XgqV66s1q1bZxg3AAB5XX7PL1Kb8zc4OFhdunRR8eLF5eLiolKlSqlnz54KCwuz2NfixYtVv35983scTZs21U8//WTRZs6cOapevbqcnZ1VokQJvfjiiymmisjM8e5+T2ThwoXq1q2bJKlFixbm/GPbtm3q37+/fH19FR8fn+LcW7VqpcqVK6d53QAAKKgo/gLIdt27d5fJZNLKlStTrFu5cqVatWolHx8fSdIvv/yipk2bKjw8XBMmTNDUqVN169YtPfbYY9q9e3eK7bt166bbt29r6tSpev755zOMJSIiQqGhoSlehmGkuc3p06f13XffqX379po+fbrGjBmjv/76S82aNdO///5rbvfFF19oxIgRqlatmj7++GNNnDhRtWrV0q5duyRJnTt3Nv/x9dFHH2nRokVatGiR/Pz8Uj1uZGSkmjRpopkzZ6pVq1aaMWOGhgwZouPHj+vChQupbjN48GA98cQTkmTe/6JFi2QymdSnTx9t3LhRN27csNhm3bp1Cg8PV58+fTK8fgAA5Cd5IQd5+umnJcn8YbBkiYmJWrlypZo0aaIyZcooKSlJHTt21AcffKAOHTpo5syZ6tSpkz766CP16NEj3fO8cuWKHn30UW3atElDhw7VlClTFBMTo44dO+rbb7+1OGb79u01ceJE1alTRx9++KFeeuklhYWF6e+//0513+nl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hhINQyOqlZCrg535WV+rSCF1LltEW/69pdCYeHINAECBlVPP5vReoTHx2vzvLXUuW0S1ihQyx+DqYKfh1Uoo0TC04cKNXI+L5zQA3L/ExETt379fNWrUkKura5a2dXV1VY0aNbR//34lJubu8P+PPfaYJOnMmTOZ3mbBggV67LHHVLRoUTk7O6tatWr69NNPU7QzDEOTJ09WqVKl5ObmphYtWujIkSMp2m3btk0mk0nbtm1L97h///23ypQpo6eeekpr165VQkLuT1Hw/fffKz4+XkOHDjUvM5lMeuGFF3ThwgXt2LEj3e1Xr16tYsWKqXPnzuZlfn5+6t69u77//nvFxsam2CYuLk5RUVHZdxIPgI+lAch3Xt0bku37TIqLU8megyVJww5dld2xWynaTD50Ps3t01sHAADyH2vlG8/+Hpzm9umtAwDA1uXEszmzlp8J1fIzoamuW3EmVCvSWGcN7u7uatGihflrAICl4OBgRUZGmnvVZlW9evW0b98+BQcHq0qVKtkcXdpOnTolSSpSpEimt/n0009VvXp1dezYUQ4ODlq3bp2GDh2qpKQkvfjii+Z2b775piZPnqy2bduqbdu22r9/v1q1aqW4uLj7irVWrVoaP368Fi5cqKeeekr+/v7q37+/nn322RTDJd8tPj5eYWFhmTpG4cKFZWeXdv/WAwcOqFChQqpatarF8vr165vXN27cON3ta9euneIY9evX1+eff66TJ08qKCjIvPyXX36Rm5ubEhMTVbZsWY0aNUovvfRSps4lJ1D8BQBJdk5OqtCln7XDAAAANox8AwAA5DR3d3d9++231g4DAPKs0NBQubi4qFixYve1fbFixeTi4qLQ0Jz94E9YWJhCQ0MVExOjP/74Q5MmTZKrq6vat2+f6X38+uuvFr2bhw0bpieffFLTp083F3+vXbumadOmqV27dlq3bp1M/z+14bhx4zR16tT7it3b21tvvvmmxo8fr23btmn+/PmaMWOG3n33XTVt2lQDBw5Ut27dUvS8/uOPP8wfYMrImTNnFBAQkOb6S5cuqVixYubzSZY8FPW///6b7v4vXbqkpk2bplh+9/bJxd8aNWqocePGqly5sq5fv66FCxdq5MiR+vfff/Xee+9l6nyyG8VfAPnOu3UDzPPu5oar0XEavvO0BgQWVZtShS3Wbfn3lj4/cVkfNyin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nXaa/ve//5ldWsA9/PDDmjx5sm644Qa99dZbuvTSS2vct6ysTM8884z69OmjhIQEJSUlqVu3brr22mu1du3aAFYNAADqw2YYhmF2EQDMN3nyZF1xxRVaunSp+vbtW+0+hmGotLRUkZGRioiICHCF1Zs/f75OOukk3XLLLerXr58qKiq0f/9+ff/99/rkk0+UmJioGTNm6OSTT678noqKCpWXl8vpdMpms5lYfcOUl5fL6/XK6XSaXQoAwAIGDBigXbt2acuWLdqwYYPat29fZfvMmTN1/vnna968eRo8eHCVbd27d5fb7db8+fMDV7APuVwujRw58pAGtlUyQ22tXbtWXbp00SuvvKKrr7668vk5c+Zo6NCh+vLLL3XqqadWPl/fLFJaWiq73a7IyEif1f5XN910k55//nnx5y4AhK/vv/9eJ510klq1aqXLL79caWlp2r59uxYvXqzffvtNGzduNLvEw7LZbBo/frzP3szUv39/ORwOfffdd0fcd9iwYfr88881atQoHXfccSovL9fatWv16aef6sEHHwzqO3gcfH1r8+bN3C0NABC2HGYXACB02Gw2RUdHB/y8hYWFR7y65IQTTtDIkSOrPLdq1SqdfvrpOu+887RmzRo1bdpUkhQRERE0zeu/qs1YD/LnC6YAgPCyefNmff/995o1a5auu+46TZ06VePHjze7LNMFc2bwh3379kmSkpKSavV8fbMIb1wDAATCQw89pMTERC1durTGtS2c7Nu3T127dj3ifkuXLtWnn36qhx56SPfee2+Vbc8995xycnL8VCEAAPAVbvsMoNb++pm/TzzxhGw2m7Zu3XrIvvfcc4+ioqKUnZ1d+dySJUs0ZMgQJSYmKjY2VoMGDdLChQurfN+ECRNks9m0Zs0aXXzxxUpOTtbAgQPrVW+vXr309NNPKycnR88991zl89V9/suyZct0xhlnyO12KyYmRm3atNGVV155yNifeOIJPfXUU2rdurViYmI0aNAg/fzzz4ece+3atRo5cqRSUlIUHR2tvn376uOPP66yz8E6FixYoLFjx6px48Zq0aKFJCk/P1+33Xab0tPT5XQ61bhxY5122mlavnx55fdX9zl7hYWFuuOOO9SyZUs5nU516tRJTzzxxCFXvdhsNt1000368MMP/5+9+46Pos7/OP7eTdmUTU8g9N5EBAQEEURQKaIUQaTogWIFRT1Pz/LzRM/OqeidqChFpRcBK2BBFBREEMRCkS49pG56svP7g2OPNQkEyO5seT0fDx6PZWYy85nveXzfmc/OjM4//3zZbDa1bNlSS5cuPeNxBgD4v5kzZyohIUF9+/bV4MGDNXPmTLf106dP13XXXSdJ6t69u+t1C1999ZXq16+vX375RStXrnQtP/nO4MzMTN17772uualx48Z6/vnn5XQ6XducPM9OnjxZjRo1ks1mU4cOHbRu3Tq3Wi677LIydx5L5c+L//rXv9S5c2clJSUpMjJS7dq104IFC9y2sVgsys3N1TvvvOOq/8TdLBW9M27SpElq2bKlbDabatasqbFjx5a5EHrZZZfp/PPP16+//qru3bsrKipKtWrV0gsvvFDB/wplzZgxQxdddJGioqKUkJCgSy+9VMuXLz/jWqTT57BRo0apW7dukqTrrrvO9b/jZZddppEjR0qSOnTo4DY+5Y250+nUK6+8olatWikiIkIpKSnq3bu326tFynvnb1X+dzJq1Ci99tprkuT2ehAAQHDZsWOHWrZsWabxK0nVqlVz+/uJ35FnzpypZs2aKSIiQu3atdPXX39d5mf379+vm2++WdWrV3f9Lj116tQy2xUWFurxxx9X48aNZbPZVKdOHT344IMqLCwss919992nlJQUxcTEqF+/fvrjjz8qfZ5HjhzR6NGjVb16dUVERKh169Z65513XOtPvC5r165d+vjjj13zYkXvxN2xY4ek40+F+bOQkBAlJSW5Lfvxxx/Vp08fxcbGym636/LLL9eaNWvctjlxrefPysta9evX19VXX61Vq1bpoosuUkREhBo2bKh33323zM//8ssv6tGjhyIjI1W7dm099dRTbtkBAIBgxZ2/AM7akCFD9OCDD2revHl64IEH3NbNmzdPPXv2VEJCgiTpyy+/VJ8+fdSuXTs9/vjjslqtmjZtmnr06KFvvvlGF110kdvPX3fddWrSpImeeeaZc3pc3+DBgzV69GgtX75cTz/9dLnbHDlyRD179lRKSooeeughxcfHa/fu3Xr//ffLbPvuu+8qJydHY8eOVUFBgV555RX16NFDmzdvVvXq1SUd/+XjkksuUa1atfTQQw8pOjpa8+bN04ABA7Rw4UINHDjQbZ9jxoxRSkqK/vGPfyg3N1eSdMcdd2jBggW66667dN555+nYsWNatWqVfvvtN1144YXlnodhGOrXr59WrFih0aNHq02bNlq2bJkeeOAB7d+/Xy+//LLb9qtWrdL777+vMWPGKCYmRq+++qoGDRqkvXv3lvllDgAQ2GbOnKlrr71W4eHhGjZsmF5//XWtW7dOHTp0kCRdeumlGjdunF599VU98sgjatGihSSpRYsWmjhxou6++27Z7XY9+uijkuSaE/Py8tStWzft379ft99+u+rWratvv/1WDz/8sA4ePKiJEye61TFr1izl5OTo9ttvl8Vi0QsvvKBrr71WO3fuPKu7TF955RX169dPI0aMUFFRkebMmaPrrrtOH330kfr27StJeu+993TLLbfooosu0m233SZJatSoUYX7HD9+vJ544gldccUVuvPOO7V161bXeK1evdqtzoyMDPXu3VvXXnuthgwZogULFujvf/+7WrVqpT59+pyy9ieeeELjx49X586d9eSTTyo8PFxr167Vl19+qZ49e55RLZXJYbfffrtq1aqlZ555xvU6jRP/OzZr1kyTJ0/Wk08+qQYNGpxyfEaPHq3p06erT58+uuWWW1RSUqJvvvlGa9asqfDVIlX938ntt9+uAwcO6LPPPtN77713ynEGAASuevXq6bvvvtPPP/+s888//7Tbr1y5UnPnztW4ceNks9k0adIk9e7dW99//73r5w8fPqxOnTq5msUpKSn69NNPNXr0aGVnZ+vee++VdPzLUP369dOqVat02223qUWLFtq8ebNefvllbdu2TYsXL3Yd95ZbbtGMGTM0fPhwde7cWV9++aUrp5xOfn6+LrvsMv3++++666671KBBA82fP1+jRo1SZmam7rnnHrVo0ULvvfee7rvvPtWuXVv333+/JCklJaXCcZOO58NLLrlEoaEVXz7+5Zdf1LVrV8XGxurBBx9UWFiY3nzzTV122WVauXKlOnbsWKnz+LPff//ddT1n5MiRmjp1qkaNGqV27dqpZcuWkqRDhw6pe/fuKikpcV17mTx5siIjI8/qmAAABBQDAAzDmDZtmiHJWLduXYXb7Nq1y5BkTJs2zbXs4osvNtq1a+e23ffff29IMt59913DMAzD6XQaTZo0MXr16mU4nU7Xdnl5eUaDBg2MK6+80rXs8ccfNyQZw4YNq1TdK1asMCQZ8+fPr3Cb1q1bGwkJCWXOddeuXYZhGMaiRYsqfe6RkZHGH3/84Vq+du1aQ5Jx3333uZZdfvnlRqtWrYyCggLXMqfTaXTu3Nlo0qRJmTq6dOlilJSUuB0vLi7OGDt27CnPfeTIkUa9evVcf1+8eLEhyXjqqafcths8eLBhsViM33//3bVMkhEeHu62bNOmTYYk49///vcpjwsACCw//PCDIcn47LPPDMM4PmfVrl3buOeee9y2mz9/viHJWLFiRZl9tGzZ0ujWrVuZ5f/85z+N6OhoY9u2bW7LH3roISMkJMTYu3evYRj/m2eTkpKM9PR013ZLliwxJBkffviha1m3bt3KPdaf50XDOJ41TlZUVGScf/75Ro8ePdyWR0dHGyNHjiyzzz9nhiNHjhjh4eFGz549jdLSUtd2//nPfwxJxtSpU93qPDkPGYZhFBYWGqmpqcagQYPKHOtk27dvN6xWqzFw4EC34xiG4cpSla3lTHJYRbmqopz45zH/8ssvDUnGuHHjypzTyceuV6+e23h74r+TsWPHGvy6CwDBbfny5UZISIgREhJiXHzxxcaDDz5oLFu2zCgqKiqzrSRDkvHDDz+4lu3Zs8eIiIgwBg4c6Fo2evRoo0aNGkZaWprbzw8dOtSIi4tzZY/33nvPsFqtxjfffOO23RtvvGFIMlavXm0YhmFs3LjRkGSMGTPGbbvhw4cbkozHH3/8lOc4ceJEQ5IxY8YM17KioiLj4osvNux2u5Gdne1aXq9ePaNv376n3J9hHJ+zT+SY6tWrG8OGDTNee+01Y8+ePWW2HTBggBEeHm7s2LHDtezAgQNGTEyMcemll7qWnbjW82d/zlon6pRkfP31165lR44cMWw2m3H//fe7lt17772GJGPt2rVu28XFxZXZJwAAwYbHPgM4J9dff73Wr1/veiyQJM2dO1c2m039+/eXJG3cuFHbt2/X8OHDdezYMaWlpSktLU25ubm6/PLL9fXXX5d5LM8dd9xRZTXa7Xbl5ORUuP7EI6A++ugjFRcXn3JfAwYMUK1atVx/v+iii9SxY0d98sknkqT09HR9+eWXGjJkiHJyclzneuzYMfXq1Uvbt2/X/v373fZ56623lnmfYHx8vNauXasDBw5U+jw/+eQThYSEaNy4cW7L77//fhmGoU8//dRt+RVXXOF2584FF1yg2NhY7dy5s9LHBAD4v5kzZ6p69erq3r27pOOPPbz++us1Z84clZaWntO+58+fr65duyohIcE1J6alpemKK65QaWlpmUcpXn/99a6nhkhS165dJems56aT7/zIyMhQVlaWunbt6vYahTPx+eefq6ioSPfee6+s1v/9KnXrrbcqNjZWH3/8sdv2drtdN9xwg+vv4eHhuuiii057PosXL5bT6dQ//vEPt+NIcj0ysbK1nE0OO1sLFy6UxWIp933Rp3rsstn/nQAAAtOVV16p7777Tv369dOmTZv0wgsvqFevXqpVq1aZ1zJJ0sUXX6x27dq5/l63bl31799fy5YtU2lpqQzD0MKFC3XNNdfIMAy3OatXr17KyspyZYz58+erRYsWat68udt2PXr0kCStWLFCklzXEv78e/yJO4hP55NPPlFqaqqGDRvmWhYWFqZx48bJ4XBo5cqVlR+w/7JYLFq2bJmeeuopJSQkaPbs2Ro7dqzq1aun66+/3vV6idLSUi1fvlwDBgxQw4YNXT9fo0YNDR8+XKtWrVJ2dvYZH1+SzjvvPNf8Lh2/S7lZs2Zuc/0nn3yiTp06uT1JLiUlRSNGjDirYwIAEEh47DOAc3Ldddfpr3/9q+bOnatHHnlEhmFo/vz5rve9SNL27dslyfW+uPJkZWW5XcRr0KBBldXocDgUExNT4fpu3bpp0KBBeuKJJ/Tyyy/rsssu04ABAzR8+HDZbDa3bZs0aVLm55s2bap58+ZJOv5oIsMw9Nhjj+mxxx4r93hHjhxxayCXd64vvPCCRo4cqTp16qhdu3a66qqr9Je//MXtF6o/27Nnj2rWrFnmXE88mvPP72auW7dumX0kJCS4vacZABDYSktLNWfOHHXv3l27du1yLe/YsaNefPFFffHFF65HDJ+N7du366effqrwsYJHjhxx+/uf56YT2eBs56aPPvpITz31lDZu3Oj2fr2zff/ribm0WbNmbsvDw8PVsGHDMnNt7dq1yxwrISFBP/300ymPs2PHDlmtVp133nnnXMvZ5LCztWPHDtWsWVOJiYln9HNm/3cCAAhcHTp00Pvvv6+ioiJt2rRJixYt0ssvv6zBgwdr48aNbnNtRb/v5+Xl6ejRo7JarcrMzNTkyZM1efLkco93Ys7avn27fvvtt9PObXv27JHVai3zSoU/z+8V2bNnj5o0aVLmy2IVXQeoLJvNpkcffVSPPvqoDh48qJUrV+qVV17RvHnzFBYWphkzZujo0aPKy8srt9YWLVrI6XRq3759rsc0n4nKXK/Ys2dPuY+VruzYAQAQyGj+AjgnNWvWVNeuXTVv3jw98sgjWrNmjfbu3avnn3/etc2Ju0kmTJigNm3alLsfu93u9veqekdLcXGxtm3bdsr3+1gsFi1YsEBr1qzRhx9+qGXLlunmm2/Wiy++qDVr1pSp7VROnOvf/vY39erVq9xtGjdu7Pb38s51yJAh6tq1qxYtWqTly5drwoQJev755/X++++f9h2BlfXnu41PMM7hHcsAAP/y5Zdf6uDBg5ozZ47mzJlTZv3MmTPPqfnrdDp15ZVX6sEHHyx3fdOmTd3+Xpm5yWKxlDtX/fku5W+++Ub9+vXTpZdeqkmTJqlGjRoKCwvTtGnTNGvWrDM9lbPiC3Pt2eQwb/PEfycAAJwsPDxcHTp0UIcOHdS0aVPddNNNmj9/frlPq6jIiTn1hhtuqPBLVRdccIFr21atWumll14qd7s6deqc4RmYp0aNGho6dKgGDRqkli1bat68eZo+ffoZ7aOiL95V9JQZ5noAAM4NzV8A5+z666/XmDFjtHXrVs2dO1dRUVG65pprXOtPfIM1NjZWV1xxhVdrW7BggfLz8ytsxJ6sU6dO6tSpk55++mnNmjVLI0aM0Jw5c3TLLbe4tjlx98zJtm3bpvr160uS687csLCwcz7XGjVqaMyYMRozZoyOHDmiCy+8UE8//XSFzd969erp888/V05Ojtvdv1u2bHGtBwDgZDNnzlS1atX02muvlVn3/vvva9GiRXrjjTcUGRl5yrtlK1rXqFEjORyOKp3/ExISyn2875/vbFm4cKEiIiK0bNkytyd5TJs2rczPVvZO4BNz6datW92exlFUVKRdu3ZV2Xk2atRITqdTv/76a4UN28rW4s0c1qhRIy1btkzp6elndPevJ/47Odu7uwEAga99+/aSpIMHD7otr+j3/aioKNcdvDExMSotLT3tnNWoUSNt2rRJl19++SnnpHr16snpdGrHjh1ud6xu3bq1UudSr149/fTTT3I6nW53/3riOkBYWJguuOACbd++XWlpaUpJSVFUVFS5tW7ZskVWq9XV5D7xlI7MzEzXq7eks78zWTp+buX9b1bZsQMAIJDxzl8A52zQoEEKCQnR7NmzNX/+fF199dWKjo52rW/Xrp0aNWqkf/3rX3I4HGV+/ujRox6pa9OmTbr33nuVkJCgsWPHVrhdRkZGmW+PnrjQevIjIqXj7+A7+Z2933//vdauXetqyFarVk2XXXaZ3nzzzTK/SEqVO9fS0lJlZWW5LatWrZpq1qxZpp6TXXXVVSotLdV//vMft+Uvv/yyLBZLld0xDAAIDPn5+Xr//fd19dVXa/DgwWX+3HXXXcrJyXG9E+/E3H7iPW8ni46OLnf5kCFD9N1332nZsmVl1mVmZqqkpOSM627UqJG2bNniNqdu2rRJq1evdtsuJCREFovF7Y6S3bt3a/HixZWu/8+uuOIKhYeH69VXX3XLDlOmTFFWVpb69u17xudTngEDBshqterJJ58s8z7eE8etbC3ezGGDBg2SYRh64oknyqw71Z06nvjv5FT/vQIAgsOKFSvKnX9OvGf3z48H/u6771zv7JWkffv2acmSJerZs6dCQkIUEhKiQYMGaeHChfr555/L7PfkOXXIkCHav3+/3nrrrTLb5efnKzc3V5Jcv6e/+uqrbttMnDixUud41VVX6dChQ5o7d65rWUlJif7973/LbrerW7duldrPybZv3669e/eWWZ6ZmanvvvtOCQkJSklJUUhIiHr27KklS5Zo9+7dru0OHz6sWbNmqUuXLq7XgZ34MtrXX3/t2i43N1fvvPPOGdd3wlVXXaU1a9bo+++/dy07evSoZs6cedb7BAAgUHDnLwA3U6dO1dKlS8ssv+eeeyr8mWrVqql79+566aWXlJOTo+uvv95tvdVq1dtvv60+ffqoZcuWuummm1SrVi3t379fK1asUGxsrD788MNzqvubb75RQUGBSktLdezYMa1evVoffPCB4uLitGjRIqWmplb4s++8844mTZqkgQMHqlGjRsrJydFbb72l2NhYXXXVVW7bNm7cWF26dNGdd96pwsJCTZw4UUlJSW6PKXzttdfUpUsXtWrVSrfeeqsaNmyow4cP67vvvtMff/yhTZs2nfJccnJyVLt2bQ0ePFitW7eW3W7X559/rnXr1unFF1+s8OeuueYade/eXY8++qh2796t1q1ba/ny5VqyZInuvffeMu8QAgAEtw8++EA5OTnq169fues7deqklJQUzZw5U9dff73atGmjkJAQPf/888rKypLNZlOPHj1UrVo1tWvXTq+//rqeeuopNW7cWNWqVVOPHj30wAMP6IMPPtDVV1+tUaNGqV27dsrNzdXmzZu1YMEC7d69W8nJyWdU980336yXXnpJvXr10ujRo3XkyBG98cYbatmypbKzs13b9e3bVy+99JJ69+6t4cOH68iRI3rttdfUuHHjMu/cbdeunT7//HO99NJLqlmzpho0aFDuO+RSUlL08MMP64knnlDv3r3Vr18/bd26VZMmTVKHDh10ww03nNG5VKRx48Z69NFH9c9//lNdu3bVtddeK5vNpnXr1qlmzZp69tlnK12LN3LYCd27d9eNN96oV199Vdu3b1fv3r3ldDr1zTffqHv37rrrrrvK/TlP/HfSrl07SdK4cePUq1cvhYSEaOjQoed8jgAA/3H33XcrLy9PAwcOVPPmzVVUVKRvv/1Wc+fOVf369XXTTTe5bX/++eerV69eGjdunGw2myZNmiRJbl9qeu6557RixQp17NhRt956q8477zylp6drw4YN+vzzz5Weni5JuvHGGzVv3jzdcccdWrFihS655BKVlpZqy5YtmjdvnpYtW6b27durTZs2GjZsmCZNmqSsrCx17txZX3zxhX7//fdKneNtt92mN998U6NGjdL69etVv359LViwQKtXr9bEiRPdngpWWZs2bdLw4cPVp08fde3aVYmJidq/f7/eeecdHThwQBMnTnQ9lvmpp57SZ599pi5dumjMmDEKDQ3Vm2++qcLCQr3wwguuffbs2VN169bV6NGj9cADDygkJERTp05VSkpKuY3mynjwwQf13nvvqXfv3rrnnnsUHR2tyZMnu+6GBgAgqBkAYBjGtGnTDEkV/tm3b5+xa9cuQ5Ixbdq0Mj//1ltvGZKMmJgYIz8/v9xj/Pjjj8a1115rJCUlGTabzahXr54xZMgQ44svvnBt8/jjjxuSjKNHj1aq7hUrVrjVGRYWZqSkpBiXXnqp8fTTTxtHjhyp8Fx37dplGIZhbNiwwRg2bJhRt25dw2azGdWqVTOuvvpq44cffnD9zIlznzBhgvHiiy8aderUMWw2m9G1a1dj06ZNZY6xY8cO4y9/+YuRmppqhIWFGbVq1TKuvvpqY8GCBWXqWLdundvPFhYWGg888IDRunVrIyYmxoiOjjZat25tTJo0yW27kSNHGvXq1XNblpOTY9x3331GzZo1jbCwMKNJkybGhAkTDKfT6badJGPs2LFl6q5Xr54xcuTIcscaABBYrrnmGiMiIsLIzc2tcJtRo0YZYWFhRlpammEYx+f7hg0bGiEhIYYkY8WKFYZhGMahQ4eMvn37GjExMYYko1u3bq595OTkGA8//LDRuHFjIzw83EhOTjY6d+5s/Otf/zKKiooMw3CfZ/9MkvH444+7LZsxY4bRsGFDIzw83GjTpo2xbNmycufFKVOmGE2aNDFsNpvRvHlzY9q0aa6scbItW7YYl156qREZGWlIcs2Ff84MJ/znP/8xmjdvboSFhRnVq1c37rzzTiMjI8Ntm27duhktW7Yscz7l1VmRqVOnGm3btjVsNpuRkJBgdOvWzfjss8/OuBbDqFwOO5Gr5s+f7/azFWWW8s6lpKTEmDBhgtG8eXMjPDzcSElJMfr06WOsX7/etU15eaOq/zspKSkx7r77biMlJcWwWCxl/jcHAAS+Tz/91Lj55puN5s2bG3a73QgPDzcaN25s3H333cbhw4fdtj3xO/KMGTNc2aFt27aurHOyw4cPG2PHjjXq1KljhIWFGampqcbll19uTJ482W27oqIi4/nnnzdatmzpmsvbtWtnPPHEE0ZWVpZru/z8fGPcuHFGUlKSER0dbVxzzTXGvn37ys1A5Tl8+LBx0003GcnJyUZ4eLjRqlWrcq/b1KtXz+jbt2+l9vfcc88Z3bp1M2rUqGGEhoYaCQkJRo8ePdyuaZywYcMGo1evXobdbjeioqKM7t27G99++22Z7davX2907NjRCA8PN+rWrWu89NJL5Watiurs1q2bW8Y0DMP46aefjG7duhkRERFGrVq1jH/+85/GlClTys1vAAAEE4thnOL5WwAASccfE9mgQQNNmDBBf/vb38wuBwAAAAAAVBGLxaKxY8eWeY0SAACAP+KdvwAAAAAAAAAAAAAQAGj+AgAAAAAAAAAAAEAAoPkLAAAAAAAAAAAAAAGAd/4CAAAAAAAAAAAAQADgzl8AAAAAAAAAAAAACAA0fwEAAAAAAAAAAAAgAISaXUAwcDqdOnDggGJiYmSxWMwuBwAAv2YYhnJyclSzZk1ZrXyP7QTyBgAAVYe8UT7yBgAAVYe8AcBTaP56wYEDB1SnTh2zywAAIKDs27dPtWvXNrsMn0HeAACg6pE33JE3AACoeuQNAFWN5q8XxMTESDr+j3hsbKzJ1QAoz+HDh9W+fXtJ0g8//KDq1aubXBGAimRnZ6tOnTqu+RXHkTcA30feAPwHeaN85A3A95E3AP9B3gDgKTR/veDEo5BiY2P55QjwUXl5ea7/r8bExPD/VcAP8KhBd+QNwPeRNwD/Q95wR94AfB95A/A/5A0AVY3mLwBIstvtGjlypOszAABAVSNvAAAATyNvAAAAi2EYhtlFBLrs7GzFxcUpKyuLb9sBAHCOmFfLx7gAAFB1mFfLx7gAAFB1mFcBeIrV7AIAAAAAAAAAAAAAAOeOxz4DgKSCggK9+OKLkqT7779fERERJlcEAAACDXkDAAB4GnkDAADw2Gcv4PENgO87dOiQmjdvLknasmWLUlNTTa4IQEWYV8vHuAC+j7wB+A/m1fIxLoDvI28A/oN5FYCn8NhnAAAAAAAAAAAAAAgANH8BAAAAAAAAAAAAIADQ/AUAAAAAAAAAAACAAEDzFwAAAAAAAAAAAAACAM1fAAAAAAAAAAAAAAgANH8BAAAAAAAAAAAAIACEml0AAHOVpqcp79P3FdXnWoUkJptdjked6lztdruuvfZa12cAAOBZQZlBLu1F3gAAAB7ly9c3gjL/BcG5AgB8D81fIMg509PkmP22IjpeGvBh9FTnarfbNXXqVJMqAwAg+HgzgxjFxcr/8mPlffGxnBnHFFK9pqJ69ldEl8tlsXr+YUgnzjW546XkDQAA4FG+fH2Da1AAAHgHzV8AXuPMz1Pu4tnKW7pIzvQ0WROTFdV7oKIHDJM1Msrs8gAAQAAyCguU/sR9Ktq8Qbb2nRXe4gIVb/1FmS88qojvv1H8X8d7pQEMAAAA7+I6FAAgWNH8BeAVzvw8HXvoDpXs3CYZzuPLjh2RY9ZbKlizUknPvWFq8C4oKNCUKVMkSaNHj1ZERIRptQAAgKrjWPieirb8rMRnX5ft/Atdy/O/+UyZzz+q/LYXKeryq71SS2FhoSa/9pok8gYAAPAMrm8c5+vXoQAA8CSavwAkSc6iQjkL8j22f8eCd1Wyc6tkGO4rDKdKdm6VY8G7sl830mPHl46fY0UyMzP16KOPSpIGDRqk1NRUj9YCAACO82QGMQxDeUsXKfKyXgpr3MLtOLYOXRTe5iLlfbxQEZdc7pHjn3Aig2Q7csgbAADAo/zh+oanr0FJ5l+HOtU1KAAAPI3mLwBJUvqDt5p3cMNQ7typyp3rm++kAQAAnuONDJK//APlL/+gwvWHB3fzeA0AAAA4ztRrUBLXoQAAAY+XWwEAAAAAAAAAAABAAODOXwCSpMQX3lJYw6Ye2//R2wbLmX60wvXWxBSlTF7gseNLUvHObeZ/uxQAALjxdAbJfmOCCn/4VokT3lZIQpJreckfu3XswdtkH3Sjogfd6LHjS2QQAACAk3k6/0nmX4ci/wEAzETzF4AkyRpukzUi0mP7j+pzrRyz3pIMZ9mVFqui+lzr0eNLx88RAAD4Fk9nkJgbblfhj98r/aE7FN3veoXWbajibb8o98N5Cq1RW9EDhpNBAAAAvMjT+U8y/zoU+Q8AYCaavwC8InrAMBWsWamSndvcg7fFqtCGTRU9YJh5xQEAgIAVklxdyRPeVs57ryvnvTekkmJZIqMU2b2PYm68Q9Zou9klAgAAoIpxHQoAEMxo/gLwCmtklJKee0O5i2crb+kiOdPTZE1MVlTvgYoeMEzWyCizSwQAAAEqpFqq4u9/QrFjH5LhyJY1Nl4W7sYAAAAIWFyHAgAEM5q/ALzGGhmlmGGjFTNstNmllGG329WnTx/XZwAAEHisEZGShx8xeCpRkVHkDQAA4FFc3/gfX74OBQCAJ9H8BYKcNTFZ9mG3yJqYbHYpHneqc7Xb7Zo9e7YJVQEAEJyCMYNE1a5L3gAAAB7ly9c3gjH/BcO5AgB8j8UwDMPsIgJddna24uLilJWVpdjYWLPLAQDArzGvlo9xAQCg6jCvlo9xAQCg6jCvAvAU7vwFAEkFBQWaN2+eJGnIkCGKiIgwuSIAABBoyBsAAMDTyBsAAIDmLwBIyszM1Lhx4yRJPXv2VGpqqskVAQCAQEPeAAAAnkbeAAAAVrMLAAAAAAAAAAAAAACcO5q/AAJOemGxZu04ovTCYrNL8TmMDQDgbDGHnBrjAwAAAIlceDqMDwB4Hs1fAAEno7BEc3alKaOwxNQ6DuQVadr2w3py4169/PN+rU/LkdMwTK3JV8YGAOB/mENOjfEBAACARC48HcYHADyPd/4CgAcs35+hSb8dVHRYiJrHRWpnToFWHMpSh2S7HrqgtsKsfPcGAICzkV/i1JK9x7Rsf4bSC0uUaAtVr1oJ6l83SZGhzK8AAACARG4GgGBG8xcAqtjOnAJN+u2getZK0Oim1WULscowDH2f5tALm//QrJ1HNbJxdbPLBADA7+SXOPXI+t3amVOgE8/SOFZYotk7j2rt0Rw9064+F7IAAAAQ9MjNABDcaP4CCFiFTkMFpc5KbVtQ6pQsFtfnyv5ceT7Ye0wJtlCNbFxNxol9S2qdGK3etRK09I8MDaybpPAQ74fsQqe5j50GAPi/M5lfq9rCPWluF7BOMHT8y1cL96RpcP1kM0pjjgUAAIAbcnP5yM0A4Hk0fwEErId+2F3pbZ1FRao19HZJ0l2bjsj6W+Y5H3/Yyq0Vrrvh623nvH8AAMxwJvOrNxmS5u1K07xdaWaXUiG73a7u3bu7PgMAAFQ18obvIDcDAMxC8xcAJFnDw9Vo0F/MLgMAAAQwu92uRYsWmV0GAAAIYOQNAABA8xdAwHqufX01jInw+nE/+SNd7/1+RC9e1FA1o8Jdy52GofE/7lWpYejpdvW9Xpd0/NE+vvrNUwCAfzBrfpWkO779XemFJRWuT7SF6o3Ojb1Y0f8wxwIAAOBk5ObykZsBwPNo/gIIWDarRRGVfK9uSUmJPv74Y0lS3759FRp69v889q6VoM/2Z+rJjXs1rGGKWidE63BBsRbuTtO27HyNb1O30nVVNZvVYspxAQCB40zm16rWu1aCZu88WubdZZJk+e96X55jqzJvAAAAlIe84TvIzeXj2hQAeB6zPwBISktL08iRIyVJW7ZsUWpq6lnvKyo0RM+0q69JWw5q0m8HXUG7dlS4HmtdR22SeOcOAABno3/dJK09mqOdOQVuF7IskhrGRKh/3SSzSquUqswbAAAA5SFvQPL/3AwAODc0fwHAAxJsoXq0dR0dLSjWgbxC2UND1DAmQhYL324EAOBsRYZa9Uy7+lqy95iW7c9QemGJEm2h6lUrQf3rJiky1Jy7FwAAAABfQm4GgOAWsM3fm2++WdOmTdOvv/6qFi1aSJK++uorDRgwQJmZmW7bjh8/Xhs3btTixYslSfXr19fhw4cVEhLi2qZZs2Zav369JMlisejHH39UmzZtvHEqAPxYSkSYUiLCzC4DAICAERlq1dCGKRraMMXsUgAAAACfRW4GgOAVkF/xycnJ0bx585SYmKgpU6ac1T5mz54th8Ph+nOi8QvA9yXYQjW0QbISbAH7/ZazxtgAAM4Wc8ipMT4AAACQyIWnw/gAgOcF5L+wc+fOVXR0tJ5++mk9+uijevbZZxUW5r077woLC1VYWOj6e3Z2tteODUBKtIVpeKNqZpfhkxgbIHCQN+BtzCGnxvgACETkDQA4c+TCU2N8AMDzAvLO3ylTpmjEiBEaOnSocnNz9eGHH3r1+M8++6zi4uJcf+rUqePV4wMAgMBH3gAAAJ5G3gAAAAD8T8A1f3/99VetWbNGI0eOlN1u18CBA90e/ZyVlaX4+Hi3P88991yZ/YwYMcJtm9GjR1e6hocfflhZWVmuP/v27auScwMAADiBvAEAADyNvAEAAAD4n4B77POUKVPUunVrtW7dWpI0cuRI9e7dW/v375ckxcXFKTMz0+1nxo8fr40bN7otmzlzpgYMGHBWNdhsNtlstrP6WQDmsNvt6tixo+szAPg68gbgf8gbAPwNeQPwP+QNAAAQUM3f4uJivffee3I4HEpNTZUkGYah0tJSTZ8+XZdcconJFQLwtoN5Dr257Ufd3rStakRV/EuP3W7XsmXLvFiZb6js+AAA4M98Zb7zxbzhK2MDAACqRlXnDbLCcYwDAMCfBNRjnz/44ANlZ2drw4YN2rhxozZu3KhNmzbpscce09SpU2UYhtklAvCyg/kOPbFptQ7mO8wuRb9lpunJTav04A8rNGvnLyosLTG7JJ8aHwAAPMVX5rvC0hLN2fWrHvxhhZ7YuEq/ZBw1tR7Jd8YGAAD4JrLCcYwDAMCfBNSdv1OmTNGwYcPUvHlzt+Xjxo3ThAkTzqj5O2zYMIWEhLj+brfbdejQoSqrFYBvKSkp0Zo1ayRJnTp1Umho1f3zWOJ06o41SzVl+0+KD7cpyRapCb+s1d9+WKHF3a/VRSk1q+xYAADAN204dkj9vlig/fkO1QqNUJ5VGr9plUY2Ol9vde6jMGvI6XcCAABwGp68vuFtjuIivfzrOk3etlEH8h2qGWnXbU3b6L7zOsgeFm52eQAA+Cz/nf3L8cknn5S7PDk5Wfn5+ZJU5n2/0vF3/p5s9+7dpzwOdxADgSctLU1XX321JGnLli2uR8dXhfEbV2n675v1eqdeuqlxK9lCQrU165huWv2J+nwxT1sH3KbkiKgqOx4AAPAtGYUF6vXZXNW2RSvrsZflOHxMP/36i5Y70nTnmmWqFhGtF9p3N7tMAAAQADx5fcObHMVF6rZ0pjamH5FTx6/F/pGXo/EbV2nx3m1a2XsEDWAAACoQUM1fAKhIfkmxcouLKlyfW1Is46TPp9r2TOSWFOs/W9ZrXPN2urFhS5U4nSpxFql2VIxmdb1GzRe/pde3btBfz7uoSo53pvJLik05LgAAZjhdHvCUN7f+qKziQn3QoY96H/6bDEnFTqeGNWihbdnp/80CHRRjwgVMsgAAAKgMb+eo5zevcWv8nuCUoY3pR/T85jV6qFUnr9VDZgIA+BOavwCCQpelM0+9gSNXCrVIkhovmSzZo6v0+C//9oNe/u2Hctf9Y+Mq/WPjqio9HgAAKOu0ecDDOi99r8K8UWP+f8wqCwAA4LTMzlEnc8rQU5u/1VObvzW7FAAAfJLV7AIAAAAAAAAAAAAAAOeOO38BBIVVvUeoTWL1CtcfOnxYFz76b0nShv63KbV6xdueiaLSUjVf8pauqtVI/+l4pdu63Y4stf5wqia0667bmrapkuOdqY3ph33q27sAAHjS6fKAp0z7/SeN+/5zLb9yqAb9KW/8dd0XWrh3m7YOuFURId7/9YwsAAAAKsPbOarposk6kO+ocH3NSLu2DbzNa/WQmQAA/oTmL4CgEBkapuhTvEcvOjRMlpM+n2rbMxEdJj3c6mLdt+4LpUZG677zOig5IkpfHNytu9Z+ptpRMbqlaesqO96ZigwNM+W4AACY4XR5wFNuanyBXv71B9268SuVNG+g0K27VOAs1XM/r9Hk7Zv0fLvLlBQR5fW6JLIAAACoHG/nqDuatdX4javKvPNXkqyy6I5mbb1aD5kJAOBPaP4CgIfd06K9coqL9Mzm7/Tcz2sUZrWq2OlU28Tq+vjy6xQTZjO7RAAA4EHRYeH6vOdQDfx8vnbfeb1UUqrzV8yRzRqixy7orAdadjS7RAAAAJ9y33kdtHjvNm1MP+LWALbKojaJ1XTfeR1MrA4AAN9G8xcAJEVERKh169auz1XJYrHosdaXaEyzC/XRH7/LUVKkNonV1TmlliwWy+l3AAAA/F6DmHituGywetw6So7kWN03ZqyGNL1AySbd8QsAAAKTJ69veJM9LFwre4/Qy7+u0+RtG3Ug36GakXbd1rSN7juvg+wmPUENAAB/QPMXACTFx8dr5cqVHj1GUkSkRjZu5dFjAAAA35WQkKAfFywxuwwAABDAvHF9w1vsYeF6rPUleqz1JWaXAgCAX7GaXQAAeFKNSLseb32JakTazS7FJzE+AIBgwHxXMcYGAACcClnhOMYBAOBPLIZhGKffDOciOztbcXFxysrKUmxsrNnlAChHSUmJtm7dKklq1qyZQkN5MALgq5hXy8e4AL6PvAH4D+bV8jEugO8jbwD+g3kVgKcw+wOApLS0NF1yyfHHCG3ZskWpqakmVwQAAAINeQMAAHgaeQMAAPDYZwAAAAAAAAAAAAAIADR/AQAAAAAAAAAAACAA0PwF4JfSMhyaOm+V0jIcZpfiFxgvAMCpME+cGcYLAAAg8JDxKo+xAgDfRvMXgF86luHQtPmrdcyHQmZBYbG+XrtNH32xSb9s2y/DMMwuycUXxwsA4DuYJ84M4wUAABB4yHiVx1gBgG8LNbsAAAgEH36xSa+/95Vycgtcy5o0qK7H77lG9WolmVcYAAB+KC+/SPM+XqcPPtukYxkOJSXY1e/K1hrSt4OiIsPNLg8AAADwKPIwAOBccOcvAJyjL1b/phfeWKquFzXR7Fdv04o5D+jF/xui4uIS3fPEHGVm55ldIgAAfiMvv0h3j5+lqfNW62h6jpyGoaPpOZo6b7XuHj9LeflFZpcIAAAAeAx5GABwrrjzF4BfKywqVn7BuYdeQ1Y1bdbM9bmy+zQMQ1PnrVLHNg10z02Xy2KxqLi4RK2a1dKzf79WI/86Ve8v3aBh/S465xrPRWFRsanHBwD4h6qaV8/FzCVrtH3XkTKvTzAMQ9t3HdHMJWt0w4BOJlV33NnOqxEREWrRooXrMwAAQFUjb5w7szNxIOdhAIB3WAxfeillgMrOzlZcXJyysrIUGxtrdjlAQNi685Bu+fs7Zpfhd95+fqSaNUw1uwzgnDCvlo9xwblgXj07zKtA4GJeLR/jAiCQkYnPHHn43DCvAvAUHvsMAAAAAAAAAAAAAAGAxz4D8Guv/XO4mtSvfs77KSkpUVpamiQpOTlZoaGV++expKRUN9w3RRdf2FD33HSF27rCohKNuOctXdHlPN0xots513gutu8+rLGPzTK1BgCA76uqefVcDB/3ltIyHBWuT06wa9art3qxorLOdl4927wBAABQWeSNc2d2Jg7kPAwA8A5mfwB+zRYepsiI8HPez6FD6Wrb5gJJ0pYtW5SaWvlH1gzp215vzlqpJvWrq98VbRQWFqK09By9+NZyFRSW6Lqr2ldJjefCFh5m6vEBAP6hqubVc9G/ZxtNnbe6zDvOJMlqsah/zzam13i282paWpqaN28u6czzBgAAQGWQN86d2Zk4kPMwAMA7aP4CwDka1q+jDh3N1sSpn2va/NVKTrRr9x/HZAsP1VN/G6DaNRLMLhEAAL8xpG8HfbNuu37fdUTOky54WS0WNW5QTUP6djCxOgAAAMCzyMMAgHNF8xcAzpHVatH9t/bUtb0v1Berf1WOo1D9r2yjK7u0lD3aZnZ5AAD4lajIcP17/HDN+3idPvhsk45lOJSUYFe/K1trSN8Oioo09y4HAAAAwJPIwwCAc0XzFwCqSIM6ybpl6KVmlwEAgN+LigzXqMGXaNTgS8wuBQAAAPA68jAA4FxYzS4AAM5GUoJdN113iZIS7GaX4hcYLwDAqTBPnBnGCwAAIPCQ8SqPsQIA38advwD8UnKCXTcP6WJ2GX6D8QIAnArzxJlhvAAAAAIPGa/yGCsA8G3c+QsAAAAAAAAAAAAAAYA7fwFAUkREhBo2bOj6DAAAUNXIGwAAwNPIGwAAgOYvAEiKj4/Xhg0bzC4DAAAEMPIGAADwNPIGAADgsc8AAAAAAAAAAAAAEABo/gIAAAAAAAAAAABAAKD5CwCSDh06pPj4eMXHx+vQoUNmlwMAAAIQeQMAAHgaeQMAAND8BQAAAAAAAAAAAIAAQPMXAAAAAAAAAAAAAAIAzV8AAAAAAAAAAAAACAA0fwEAAAAAAAAAAAAgAND8BQAAAAAAAAAAAIAAQPMXAAAAAAAAAAAAAAJAqNkFAIAviIiIUO3atV2fAQAAqhp5AwAAeBp5AwAA0PwFAEnx8fH6+eefzS4DAAAEMPIGAADwNPIGAADgsc8AAAAAAAAAAAAAEABo/gIAAAAAAAAAAABAAKD5CwCSDh06pKSkJCUlJenQoUNmlwMAAAIQeQMAAHgaeQMAAPDOXwD4r9LSUrNLAAAAAY68AQAAPI28AQBAcOPOXwAAAAAAAAAAAAAIADR/AQAAAAAAAAAAACAA0PwFcMZK09OUM3OyStPTzC7F44LpXAEAMFOwzblG5rGgOl8AAICTBVP2C6ZzBQD4Bpq/AM6YMz1Njtlvy+nF0GoYhpyObBlFhV47pmTOuQIAEIzMmnOdebly5ud59ZiSZGRlkDEAAEDQCqbrLcF0rgAA3xBqdgEAcCpGaYlyl8xR3kfzVXrkoGQNka1jV8UMv1VhDZqYXR4AAPBT+au/VO6Cd1S8/TdJUljzVrJff5MiOnQxuTIAAABUJWd+nnIXz1be0kVypqfJmpisqN4DFT1gmKyRUWaXBwBAlaP5C8BnGYahzJfGq+CbLxTZvbds7TvLmX5MuZ8u1LEHblHiM68rvOl5VXKs0NBQVa9e3fUZAAAErtwlc5T91ksKb9tRcfc9LhmG8j//SBlP/FVx4x5VVM/+HjmuW96whqrUI0cBAADBjOsb7pz5eTr20B0q2blNMpzHlx07Isest1SwZqWSnnuDBjAAIOCQAACcNWdRoZwF+R7bf+HGdSpYuVxx9/5DEV0udy2P6NZT6f+4R9lvvKDEZ16vkmMl2qP126aNrr+fOC+nlx8zDQBAsPN0vnBmZSp72r8V1Xew7KPuksVikSTZLumh7NdfUPbklxTeoYtHLgKenDeKd25TepUfAQAABLvk5GRt3brV7DIqzdPZz7HgXZXs3CoZhvsKw6mSnVvlWPCu7NeN9NjxJa4tAQC8j+YvgLOW/uCtXjlO1sQnlTXxyXLXHR7czSs1AAAA7/BWvsj7eIHyPl5Q7rqjN/bxSg0AAADBzlvZr1yGody5U5U7d6p5NQAA4AFWswsAAAAAAAAAAAAAAJw77vwFcNYSX3hLYQ2bemz/2W++qMIf1yr5tdmyhIS4rcv/4mNlvzFByW/MU0hStXM+1uHDh9WuQwdJ0vp161zvxyneuc3cb6ECABBkPJ0v8ld8quxJzyv5tdkKqVbDbV3Jvl06dt8oxf3tn4rodGmVH/vkvLFu4VxZn3+oyo8BAACC26FDh9SqVStJ0ubNm5WammpyRafm6ex39LbBcqYfrXC9NTFFKZPLfxpMVeHaEgDA22j+Ajhr1nCbrBGRHtt/1FXXKv+zD5T34VzFDB3tWl6adli5778nW7vOCqtVr0qOZbFFKK+o2PX5xHlZw21Vsn8AAFA5ns4Xkd37KGf6a8p593Ul/P0ZWcLCJElGYYFy3n1d1oQkRV7Sw7W8KrnljfDwKt8/AACAJBUXF5tdQqV5/NpSn2vlmPWWZDjLrrRYFdXnWo8eX+LaEgDA+2j+AvBZ4U3Ok334rXLMeFOFa7+Wrf0lKj12VAXffCarPVaxYx40u0QAAOBnrBGRiv/reGU8+5CO3j5YEV2vkJxO5X/9mZyOLCU+9qJHGr8AAADwvugBw1SwZqVKdm5zbwBbrApt2FTRA4aZVxwAAB5C8xeAT4sZfqvCmrZU3kfzlbdssSyRUYruP0xR1wxRSFyC2eUBAAA/FNGxq5JfmqbcJbOV//VyWWSRrX1nRQ8YprA6DcwuDwAAAFXEGhmlpOfeUO7i2cpbukjO9DRZE5MV1XugogcMkzUyyuwSAQCocjR/Afi8iPadFdG+s9llAACAABLWsKni73vc7DIAAADgYdbIKMUMG62YYaNPvzEAAAHAanYBAPyPNTFZ9mG3yJqYbHYpHhdM5woAgJmCbc61xCUE1fkCAACcLJiyXzCdKwDAN3DnL4AzFpKYrJgRt5ldhlcE07kCAGCmYJtzLfFJQXW+AAAAJwum7BdM5woA8A00fwFAUmhoqBISElyfAQAAqhp5AwAAeBp5AwAAkAAAQFJycrJ27dpldhkAACCAkTcAAICnkTcAAADv/AUAAAAAAAAAAACAAEDzFwAAAAAAAAAAAAACAM1fAH4jvbBYs3YcUXphcZXvOy0tTTVq1FCNGjWUlpZW5fv3JE+OCwAA54p56n9OzhvbDxxiXAAAQJXz5+sbvoL8WhZjAgD+hXf+AvAbGYUlmrMrTR1TYpRoC6vSfZeUlCg/P9/1uSJ/5BZq8d5j2pDmkFPS+fFR6l8vSU1iI6u0njPhyXEBAOBcmT1PGYahVYez9ekfGdqXV6iY0BBdmhqnq+skyh4W4tVaTs4b6flFmrPLwfwNAACqVGWvb6BiZudXX8SYAIB/ofkLAJX0c0aunvhxr6LDQtQtNU6hFotWHc7Wg+t26f7za6tL9VizSwQAACcxDENvbD2kT//IUKuEKPWtnajD+UVasDtNXx/K0jPt6ys+nF+JAAAA4FvyS5xasveYlu3PUHphiRJtoepVK0H96yYpMpSHeQIATo0rHQBQCaVOQy/9vF9N4yL1jzZ1ZQs5HrSHN0zRS7/s16u/7lfbpGhFh3r3DiIAAFCx9ccc+vSPDN3VooZ61kpwLb+uQbL+vm633tl+WPe0rGVihQAAAIC7/BKnHlm/WztzCmT8d9mxwhLN3nlUa4/m6Jl29WkAAwBOieYvAL9T6DRUUOqs0n0WlDoli8X1+c/7X5/mUFphiR5oVVvGie3/64ZG1fTtkWx9cSDT7cKytxQ6jdNvBACAyTwxf5/Op39kqIHdpq7VY92OnWgLU986iZq/O01/aVxNkV768tbJeaPIy2MBAACAM2NGfpWkhXvS3Bq/JxiSduYUaOGeNA2un+zVmrj2BAD+heYvAL/z0A+7q3yfJXm5CqnbWJI0Zv1+hUZllrvd309x7Le3Hdbb2w5XeW0AAAQCT8zflXX9V1srXDfym+1eq+PkvPHslqMKjYr22rEBAABwZszMrxUxJM3blaZ5u9LMLgUA4MN4PgQAAAAAAAAAAAAABADu/AXgd55rX18NYyKqdJ9paWnqlHFEkjS5Yz0lJ7s/Pqew1Kk7vv1dbRLtuvu8GrL+95GNhmFoxo6j+uSPdE26uLESbN7/Z3VnToFPfhsVAICTeWL+Pp1vDmXp378d1P+1rqMLEv93l21OcakeWb9bdaJterBVba/Vc3LeePT8Gnp+Z7bXjg0AAIJDaGioYmJiXJ9x9szIr5J0x7e/K72wpML1ibZQvdG5sRcr4toTAPgbEgAAv2OzWhQRUrUPLqhdvZr+2LunwvURIVbd0byGXvp5v9IKi3V5jXiFWS36+lCWfkzP1U1NqqlGVHiV1lRZNqvFlOMCAHAmPDF/n06PGvFafSRbz/30h3rUjFPrhGgdyi/Wp3+kq8hpaHST6l6t6eS8sSM7X6L5CwAAqlhycrL27dtndhkBwYz8Kkm9ayVo9s6jZd75K0mW/673dl1cewIA/0LzFwAqqVtqnGLDQrRgd5ombTkoSWocG6EHW9VWl+qxJlcHAAD+LMRq0aOt6+j93ce0dH+Glu/PVJjVoi7VYjW0YYppX9wCAAAAKtK/bpLWHs3RzpwCtwawRVLDmAj1r5tkVmkAAD8RkM3f+vXra+LEiRowYIDZpQAIMG2T7GqbZFdhqVOGZMo3QAEAQOWFWa26vmGKhjRIVn6pU+FWq0K5cwEAAAA+KjLUqmfa1deSvce0bH+G0gtLlGgLVa9aCepfN0mRoVyLAgCcWtDNFPXr11dkZKTsdruSk5N1zTXXaMeOHZKk6dOnKyQkRHa7XbGxsapVq5YGDRqkr7/+2m0fl112mSZOnGhC9QA8JS0tTXXq1FGdOnWUlpZ22u1tIVYavwAA+BGLxaKo0BBTG78n54309HTT6gAAAIHrTK9vwDdFhlo1tGGKpnVtqiVXnKdpXZtqaMMUGr8AgEoJytli9uzZcjgc2rlzp6KiovSXv/zFta5Vq1ZyOBzKzs7W5s2b1aNHD/Xp00czZ840sWIAkpRgC9XQBslKsFX9QwtKSkqUk5OjnJwclZSUVPn+PcmT4wIAwLlinvqfk/OG3SrGBQAAVDl/vr7hK8ivZTEmAOBfAvZf623btqlTp0765ZdfdOGFF2rGjBmqU6eO2zaxsbG68cYbNWzYsHL3kZiYqLFjxyonJ0d/+9vfNGzYMFmtp++XFxYWqrCw0PX37OzsczsZAJKkRFuYhjeqZnYZPodxAYITeQP+gnmqfAnhIRpel3EB4NvIGwCCEfm1LMYEAPxLwN75O2PGDM2ePVtHjx5VdHS0HnvssTLbZGZm6t1339WFF154yn0NHjxYhw4d0tatWyt17GeffVZxcXGuP39uOgMAAJwr8gYAAPA08gYAAADgfwK2+TtmzBg1aNBAERERGjFihNavX+9aN2LECCUkJKhly5ZyOp169913T7mvWrVqSVKl38v18MMPKysry/Vn3759Z38iAAAA5SBvAAAATyNvAAAAAP4nYB/7nJqa6vocHR2tnJwc199nzpypAQMGVHpf+/fvl3T8MdCVYbPZZLPZKr1/AACAM0XeAAAAnkbeAAAAAPxPwN75W5UWLFig1NRUNWvWzOxSAJzkYJ5D4zd+o4N5DrNL8SmMCwAAZ495tHyMCwAA/ivY5/FgP38AQPAJ2Dt/q0JGRobmzZunp556Sm+++aasVnrlgC85mO/QE5tWq1+dJqoRZT+nfYWGhioyMtL1ubI2ph/Wm1s36tesNCWGR2hEw5YaULepQk3896IqxwUAgGDjyXm0snkjp7hQ7/z+sz7843cVO0vVpVpt3d60rWpFx1RpPWeCfAEAgH8oL2+YNY//knFUr2/9UT9nHlVcuE3DGpynQfWaKcwa4rUaJHIMACD40Pz9k82bN8tut8tqtcput6tTp076+OOP1a1bN7NLA+BBycnJOnjw4Bn9zMRf1+m+dV+oVlSMulWvo52OTF23crEur1FPH/YYrMjQMA9VCwAA/FFl8sZuR6Z6LJutvbnZ6lmzgWJtkXr51x/00q/r9EGPQepRo753igUAAH7pbK5veMKbW3/UnWuWqXpktLqn1tO+3GwN+/oDdalWW59eMUT2sHCzSwQAIGAFZPN39+7dbn8fMGCA6x2/f153slGjRmnUqFGn3f9XX3111rUBCAxrjx7Qfeu+0AMtO+qZC7u57vT94uBuXfPFAj324zf6V4ceJlcJAAD8zYivP5RFFm0beLsaxsRLkrKLCnXdysUa9NUi7R08RjFhvH8TAAD4rp/Sj+jONcs0tvmFeqnD5a47fVcd3qervpivB9ev0KROvUyuEgCAwBWQzV8AwSW/pFi5xUVePeYrv65TQ3ucHrugswpLS1RYenx5p+SaurNZW729fZMeOr+jKXf/5pcUe/2YAAAEGjPyxcb0w/r26H7Nu7S/qkdEuY4fYrHo3xddofOWvK2p23/SLU1ae7UuiXwBAEAg8Fa+eeW3H1Qj0q6n2lyqotJSFZUev2jSNrG67mneXi/9+r0eu6CzYr30hTZyDAAg2ND8BeD3uiydee47yc2XXpx6/PP9N0vRkZX6sdjZL1e4LmXev8+9LgAAYIoqyRd/Vsm8MeTrJRXu4t51X+jedV9UfW0AACAgpKWlqUOHDpKkdevWKTk52bXOI/nmFOLnTKxwXc35r3mvEAAAggzNXwCQJMMpZTn+9xkAAKCqkTcAAICHlZSUKCMjw/UZAAAEH5q/APzeqt4j1Cax+jnt49Dhw7rw0eN36m7of5tSq596f29u26gH16/Qd31u1Hnx//sWbVFpqa74bI6iQsO09Ioh51TT2dqYftjr3+YFACDQVEW++LPT5Y3ckmI1XTRZg+s10ysXXeG2bsbOX3THmmX6ts8NuiChWpXWVRnkCwAA/J8n8k153tvxs8asXa6ve49Q25OOV+J0qs8X81VYWqKve4/weB0nkGMAAMGG5i8AvxcZGqbosPBz2kd0aJgsJ30+3f5ub9pG03dsVt8vF+jh8zupR4162uXI0gs/r9XPmWla0WvYOdd0tsx4zzAAAIGmKvLFn50ub0SHheufbbtq3PefK7ekWLc3ayN7aLjm7d6il39dpxsattTF1WpXaU2VRb4AAMD/eSLflGdUkws05fef1O/LhXq41cXqWbO+9uZm61+/fK91aQe19MohXr1mQo4BAAQbmr8AcBaiw8L1Zc9huuf7z/X3DV+p2Hn80Y0dkmro855D1dmkC7MAAMC/3d2ivaJDw/TkptWas/s3SVJ8uE0Pnt9R49t0Mbk6AACA04sICdVnPYfqvu+/0P/9+LUeXL9CktQmsZo+vWKILq9R39wCAQAIcDR/AeAsJUdEaeal/fRKwRXakZOpRFuEmsQmml0WAADwczc3aa2RjVrp58yjKnI61TI+WVHcsQIAAPxIfHiEpnXpqxc79ND27AzFh9vUNDZRFovl9D8MAADOCc1fADhHyRFRSo6IMrsMAAAQQEKsVrX2wjv5AAAAPCnRFqmOKZFmlwEAQFCh+QvAb9WItOvx1peoRqS9SvYXFhYYd9RU9bgAABBMPD2P+mveIF8AAOA//pw3gn0eD/bzBwAEH4thGIbZRQS67OxsxcXFKSsrS7GxsWaXAwCAX2NeLR/jAgBA1WFeLR/jAgBA1WFeBeApVrMLAAAAAAAAAAAAAACcO5q/AAAAAAAAAAAAABAAaP4CgKS0tDQ1a9ZMzZo1U1pamtnlAACAAETeAAAAnkbeAAAANH8B+I20DIemzlultAxHle+7pKREhw8f1uHDh1VSUlLl+/c2T44VAADMM5V38lgFWt4AAAC+h7xxdsi3lcM4AYB/oPkLwG8cy3Bo2vzVOuYjAfPA4UxNm79a/5q8TDMWrVFaeo7ZJbn42lgBAAKLr80zW3cc0uszvtK/Ji/T4uU/Kjev0OySXHxtrAAAAFAWma1yGCcA8A+hZhcAAP7o3fe/09tzvlZURLhqpiZo2de/6O253+i+0Veq/5VtzC4PAICgUFJSqmcmfaLPvvlVifHRSoyL1kdfbNKbs1bq6b8N1IXn1zO7RAAAAOCM5OUXad7H6/TBZ5t0LMOhpAS7+l3ZWkP6dlBUZLjZ5QEA/ADNXwA4Qyu+26K3Zn+tv1x7sW4Y2EmREeFy5BZq8uyV+tfkZapTI4GLzQAAeMGUeav05bdb9PCYq9Tz0pYKDbHqyLFsPTfpUz30/PuaOfEWpSTFmF0mAAAAUCl5+UW6e/wsbd91RIZhSJKOpudo6rzV+mbddv17/HAawACA06L5C8DvFBYVK7+gqEr3mV9Q5ArV+QVFp9z/zMVrdeH5dXXDwE6u7UNCLLpjRDf9tOUPzf7ge7VoXKNK6ztThUXFph4fABAcPDEnV1ZBYbEWLftRg3pfqO4XN1NxcYmKi6WY6Ag9MvYqDb/nLb2/dL3+MqizKfWdwJwMAADgP8zMt5I0c8kat8bvCYZhaPuuI5q5ZI1uGNDJpOrItgDgL2j+AvA7Yx+bVeX7LCjIkuO/7wcccvcbioiIO+3P9Lzx5XKX79hztMJ1AAAEEk/MyWdq3sc/aN7HP5S7bsbitZqxeK2XKwIAAIC/8oV8WxHDMPTuwu/07sLvzC4FAODjaP4CwH9ZLVazSwAAAAEuJCTE7BIAAECAI28AABDcaP4C8Duv/XO4mtSv7oE9j6/UVv94aYkOHc3SpH+OUGjo/36hyi8o1ui/T1eHC+rrvtFXeqC+ytu++7BPf1sVABAYPDcnn16p06mRf52qlk1r6uExV7mtO3A4Uzc/OF1jbuyufle0NqW+E06ek1NTU3Xs2DFT6wEAAIGNvHFuzMy3kjR83FtKy3BUuD45wa5Zr97qxYrccb0JAPwDzV8AfscWHqbIiHDTjj9qcGfd9Y9ZevLVj3TL0K5qVDdFv+04qDdmrFRubqFGDOhkan3S8TECAMDTzJ6Tbx7SRc+9/qli7ZEa3r+jkhPtWrtxl/7zzpdKTYnTNZdfwJwMAACASjM73/bv2UZT560u885fSbJaLOrfs42p9ZFtAcA/0PwFgDPUsmktPff3QXrhzaW65e/vuJbXTk3Qv/5viOrVSjKxOgAAgkffHheooLBYb8/5Rks+2+ha3qpZLT027hpFRdrMKw4AAAA4Q0P6dtA367br911H5DypAWy1WNS4QTUN6dvBxOoAAP6C5i8ASMrMzFSXLl0kSatWrVJ8fPwpt+/YtqHmvnaHNvy8R2npDtWoFqfWLerIarV4oVoAAHDCoD7tdFX3Vlq3abdy8wvVuF41NWlg3qP6TuVM8wYAAMCZIm/4t6jIcP17/HDN+3idPvhsk45lOJSUYFe/K1trSN8Oioo096k2AAD/QPMXACQVFBTojz/+cH2ujNAQqy5q3cCTZQEAgEqIjAjXpR2bml3GaZ1N3gAAADgT5A3/FxUZrlGDL9GowZeYXQoAwE9ZzS4AACorKcGum667REkJdrNL8XmMFQDAk5hnKo+xAgAA8H1ktsphnADAP3DnLwC/kZxg181Duphdhl9grAAAnsQ8U3knj9WhQw6TqwEAAEB5yLeVwzgBgH/gzl8AAAAAAAAAAAAACAA0fwEAAAAAAAAAAAAgAND8BQAAAAAAAAAAAIAAQPMXAAAAAAAAAAAAAAJAqNkFAIAvSE1NVWZmptllAACAAEbeAAAAnkbeAAAA3PkLAAAAAAAAAAAAAAGA5i8AAAAAAAAAAAAABACavwAgKTMzUxdeeKEuvPBCHo8EAAA8grwBAAA8jbwBAAB45y8ASCooKNDOnTtdnwEAAKoaeQMAAHgaeQMAAHDnLwAAAAAAAAAAAAAEAJq/AAAAAAAAAAAAABAAaP4CAAAAAAAAAAAAQACg+QsAAAAAAAAAAAAAAYDmLwAAAAAAAAAAAAAEAJq/AAAAAAAAAAAAABAAQs0uAAB8QXJysrZs2eL6DAAAUNXIGwAAwNPIGwAAgOYvAEgKDQ1Vamqq2WUAAIAARt4AAACeRt4AAAA89hkAAAAAAAAAAAAAAgDNXwCQlJmZqYsvvlgXX3yxMjMzzS4HAAAEIPIGAADwNPIGAADgsc8AIKmgoEC//fab6zMAAEBVI28AAABPI28AAADu/AVgmtL0NOXMnKzS9DSzS/G4YDpXAAAqI5jmxhPnamQeM7sUAACAgBOMuTIYzhUAcPZo/gIwjTM9TY7Zb8vpxcBamnZEhRvWqGjLZhmlpV47rhnnCgCAL/P23GgUFapw8wYVbvxezpwsrxzzhBPnamRlePW4AAAAwSCYrrkE07kCAM4ej30GEBScWZnKeu05Faz5SnI6JUkh1Wsq5qa7FdnlcnOLAwAAHmMYhvI+mCPH3GlyZmceXxhuU9SV1yh29D2yhNtMrQ8AAAD+xZmfp9zFs5W3dJGc6WmyJiYrqvdARQ8YJmtklNnlAQBA8xdA4DOKCnXs/8bKmZ6m2DsflO3CTnKmHZFj0QxlPvewLI88LzVsYXaZAADAA3IXvquc6a8pqvdARfW5VpaISBWs/kI5c6aq9NgRJTw6QRaLxewyAQAA4Aec+Xk69tAdKtm5TTKO31zgPHZEjllvqWDNSiU99wYNYACA6Wj+AjCds6hQzoJ8j+0//8tPVLL7dyVOeFth9RtLkqxxCYq7/0llPvuQst+dJOOhF1z/IBqFBVVej7OosEr3BwBAoPBkDnDm5coxZ6qirr5OMaPuci2P6jdU1uTqynppvAo3r1d405YeOb6rDnIAAACAx3n6+pIkORa8q5KdWyXDcF9hOFWyc6scC96V/bqRHjs+uRIAUBk0fwGYLv3BW71znL+NrnBd8ZjrNLJatCSp6M7BOmzllegAAHiDN3JA3kfzlffR/HLXZTwyxuPHBwAAgOd56/pShQxDuXOnKnfuVHPrAAAEPZq/ACAp1GLRmIapkqQwHv0IAAA8ICE+QatXr5YkJScnm1wNAAAIRMnJyeQNAACCHM1fAKZLfOEthTVs6rH957wzSfkrPlXKmwtksdnc1jnmT1fu4tmq9tb7skZFe6yG4p3bzP8GKgAAPsiTOaBk704d++tNivvr44ro3MNtXemRg0obO0yxdzygyMv7euT4J5zIAaEhIWrZrLlHjwUAAIJbaGioWrb07CstfJWnry9J0tHbBsuZfrTC9dbEFKVMXuCx43N9CQBQGTR/AZjOGm6TNSLSY/uPvmaI8j5dqOzJLyrurodljYySYRgq+nGt8hbPVtQV1yg00bPfhrWG206/EQAAQciTOSC8aUuFX9BeOe+8rrAGTRXWoIkkqTTjmLL+86yssfGKvLyvR3OIRA4AAADwBk9fX5KkqD7XyjHrLclwll1psSqqz7UerYFcCQCoDJq/AAJeaM06iv/bk8r81+Mq/H6Vws9rrdJjR1Sy+3eFt7lIsTfdrczMTPXv31+StGTJEsXHx5tbNAAAqBLxf3tS6f93l9LuHqGw5q1kiYxS0eYNskREKnH8yx6/QHiy7JxsXdutmyTyBgAA8Ayub3hW9IBhKlizUiU7t7k3gC1WhTZsqugBw8wrDgCA/6L5CyAoRHa5QuHNWilv2WIV796usAZNFDNyrGwXdpIlJEQFmZnatGmTJKmgoMDkagEAQFUJSUxW8ivvqmDVlyr4/msZxcWK+csYRV15tawxcV6tpbCoiLwBAAA8qqCggLzhQdbIKCU994ZyF89W3tJFcqanyZqYrKjeAxU9YJiskVFmlwgAAM1fAMEjJKW6Ym643ewyAACAl1nCwhXZvbciu/c2uxQAAAD4OWtklGKGjVbMsNFmlwIAQLmsZhcAIHhZE5NlH3aLrB5+364vCKZzBQCgMoJpbjxxrpa4BLNLAQAACDjBmCuD4VwBAGePO38BmCYkMVkxI24zuwyvCKZzBQCgMoJpbjxxrrmHDpldCgAAQMAJxlwJAMCpcOcvAAAAAAAAAAAAAAQAmr8AAAAAAAAAAAAAEABo/gIAAAAAAAAAAABAAOCdvwC8Ir2wWEv/yFDv2glKtIWZXU4ZycnJ+uijj1yfvcHXxwQAEJyYn8pXFeNiRt4AAADBJZjzBjm2LMYEAIITzV8AXpFRWKI5u9LUMSXG1LDpKC5VfqlT8eGhCrNaXMtDQ0PVpUsXr9biK2MCAMDJmJ/KVxXjYkbeAAAAwSWY8wY5tizGBACCE81fAEFhV06B3ttxROvTHDIkxYSFqGfNeA1tmCJbCE/ABwDAl+SXOLVk7zEt25+h9MISJdpC1atWgvrXTVJkKPM2AAAAfA8ZFgDgK2j+Agh4O7Lz9cj6PUqOCNOdzWsoJSJMP2Xk6sN96dqena/xbeupIC9XgwYNkiQtXLhQdrvd5KoBAAhO+SVOPbJ+t3bmFMj477JjhSWavfOo1h7N0TPt6vvtxTOHw0HeAAAAHkXeMEcgZ1gAgP+h+QvAqwqdhgpKnV495pRth1UtIkz/bFdPEf+9y7dlQpQuSIjSExv36atDWWps5Gnt999LktKyshUaGeXxugqdxuk3AgDAJGbM2ZK0cE+a20WzEwxJO3MKtHBPmgbX9/7766pi3nY4HFq7dq3rMxdjAQBAVSNvmJNjAznDAgD8D81fAF710A+7TTv2X77eVu7yV389oJK8XIXUbSxJGrN+v0KjMr1YGQAAvsfMObsihqR5u9I0b1ea2aUAAADAR/lajiXDAgC8jWdNAAAAAAAAAAAAAEAA4M5fAF71XPv6ahgT4bXj5RaX6rZvf9eQ+snqXy/JbZ3TMHTf2p1qFBupIYnxarP3d0nSpHa1lJqa6vHaduYU+Ny3UQEAOMHbc/YJd3z7u9ILSypcn2gL1RudG3uxouOYtwEAAPyDGTmWDAsA8CU0fwF4lc1qcb131xsiQqy6tHqsPtiXrourxaiu/Xj4NwxDC3an62B+se5pWUsRBdmSYbh+xhs12qwWjx8DAICz5e05+4TetRI0e+fRMu9LkyTLf9ebURfzNgAAgH8wI8eSYQEAvoTmL4CAd1OT6vo9p0D3rN2pjikxSokI06b0XO12FGpog2SdFx+lQ4eyzS4TAABI6l83SWuP5mhnToHbxTOLpIYxEepfN6miHwUAAABMQYYFAPgSmr8AAl5seKheaN9Anx3I0NeHsrTHUag60Tbd3KS62iTZzS4PAACcJDLUqmfa1deSvce0bH+G0gtLlGgLVa9aCepfN0mRod6/YwIAAAA4FTIsAMCX0PwFEBQiQ63qVzdJ/Sr4pmVycrLeeecd12cAAGCeyFCrhjZM0dCGKWaXUqXIGwAAwNPIG+YJ1AwLAPA/Qf2Vo1GjRik8PFwxMTGKi4tT06ZNdccdd2jXrl1u21ksFm3cuFGSNH36dLVp08b7xQJ+LsEWqqENkpVg883vnISGhqp///7q37+/QkO9U6OvjwkAIDgxP5WvKsbFjLwBAACCSzDnDXJsWYwJAASnoG3+Goah0tJSjRkzRjk5OcrKytKyZcsUHh6utm3b6rfffjO7RCCgJNrCNLxRNSXawswuxWcwJgAAX8T8VD7GBQAAwLeR18piTAAgOAVV87d+/fp69tln1alTJ0VFRSkvL89tfYMGDfTqq6+qU6dOevzxx8/6OIWFhcrOznb7A8C3ORwODRw4UAMHDpTD4TC7HAA4LfIG4H/IGwD8DXkD8D/kDQAAEFTNX+n4Y5vfeecdORwO2Wy2crcZPHiwVq5cedbHePbZZxUXF+f6U6dOnbPeFwDvcDgcWrFihVasWMEvRwD8AnkD8D/kDQD+hrwB+B/yBgAACLrm75133qlmzZopJCRE4eHh5W5Tq1Ytpaenn/UxHn74YWVlZbn+7Nu376z3BQAAUB7yBgAA8DTyBgAAAOB/gu5N73Xr1j3tNvv371diYuJZH8Nms1V4VzEAAEBVIG8AAABPI28AAAAA/ifo7vy1Wk9/ygsWLNBll13m+WKAIHQwz6HxG7/RwTwePXQCYwIAQMWYJ90xHgAA4EwFc34I5nMHAASvoLvz91T27NmjiRMnas2aNfruu+/MLgcISAfzHXpi02r1q9NENaLsXj9+Xkmx/v3bek39/Sfty81WnehY3dz4Al2XWNvrtZxg9pgAAODLzJ4nd+Vk6oWf12rh3q3KLSlW+6RU3dOiva6t18zrtUjmjwcAAPA/ZuQHwzD0/t6teuXXH7Q+/bCiQ8M0uF4zPXh+R9W3x3ulBonsBAAITkF35++fTZo0STExMYqNjdXll1+u3NxcbdiwQS1atDC7NABVLK+kWFcun6PHN36jTik19eyF3dQppab+sfEbDf1huYwwvg8DAAD+Z3PGEbX/eLoW79ummxtfoCfbdJFF0qCvFumxH782uzwAAACf9X8/fq3BXy1WiNWqJ9t00c2NL9D7e7ap/Ufv6OeMo2aXBwBAQAuqTsfu3bvd/j59+nRNnz79tD9nGIbr86hRozRq1KiqLQyAV7z86zptSD+sr3uP0EUpNV3LxzS7UN2WzpC6tVfE52tMrBAAAPiS279bplpRMVrZa4QSbBGSpPtbdtQzP32rR3/8WoPrNVPrxOomVwkAAOBbfjx2SM9s/k7PXXiZ/t6qk2v5g+d3VLels3TnmmX6ps8NJlYIAEBgC6rmLwDfkV9SrNziIq8e8+1tm3R9/eZqGZ/sduzz45M1qE5TLb/GomeuuV5h0VFerS2/pNhrxwIAwF95Ozv8lnVM3x3dr1ldr1G41ep27LHNLtR/tqzXG1t/1L/a96j0PsOiozThlZddn8/mfMgNAADgVOLj4/Xqq6+6Pp/MW3nqja0/qlaUXXc0beN2PJs1RH8/v6NuXPWRfjx2SE1jEz1eC9kJABCMLMbJt7XCI7KzsxUXF6esrCzFxsaaXQ5gqg3HDqndR9PNLsMnrb96lC5MSjW7DMDnMa+Wj3FBoCI7lI/cAHgW82r5GBfAP5GnyE7wTcyrADwl6N/5CwAAAAAAAAAAAACBgMc+AzDFqt4j1MbL78h7+qdv9cpvP+jLnsN0fkKKa/nmjKPqsXy26m/Zp/N+P6TX3nhd9mi71+ramH5YXZbO9NrxAADwR97ODoZh6NJlsyRJn15+nexh4a51r23ZoL9v+Epf9xp+RneQOHIdGnvHncf3cZZ5g9wAAABOxeFw6NZbb5UkvfXWW7Lb/5c3vJWn1qUdVPfls/Wvdt11R7O2ruU5xUXq8/k8hVqt+qrXcI/XIZGdAADBieYvAFNEhoYp+qSLqN7wUKuLtfTALl3+2RyNatRKbZOqa8Oxw3pnx2Y1jo7Tnrde0IHCIhkFRYqO915tkaFhXjsWAAD+yozsMPni3uqxfLY6ffqeRje+QNUio/Xhvu366I8durdFe3VNrXtG+8spKNLnS5dJ0lnnDXIDAAA4FYfDoU8//dT1+eTmr7fy1GU16mlci3b62/oVWnl4n66u01hH8nP19vZNyigq1Je9hnkt15GdAADBiOYvgKARG27TV72Ga8IvazVl+096besG1Yi0677zOugvKfXVofABs0sEAAA+pH1yDa296i96ZvN3enrzdyooLVHbxOqafklf/aXR+WaXBwAA4LMmdrhCbROr65XfftDt3y1VZEioBtdrpkcu6KzmcUlmlwcAQECj+QsgqMSG2/TPtpfqn20vVanTqRDr8VefHzp0yOTKAACAL2oRn6z3ul6jd7tcLadhuLIDAAAAKmaxWDSq8QUa1fgClTqdslosslgsZpcFAEBQoPkLIGhx8RYAAFSWxWJRCBcsAQAAzhjXXwAA8C5mXgBeVSPSrsdbX6IakfbTbxwkGBMAACrGPOmO8QAAAGcqmPNDMJ87ACB4cecvAK+qEWXX+DZdzS7DpzAmAABUjHnSHeMBAADOVDDnh2A+dwBA8OLOXwAAAAAAAAAAAAAIANz5CwCS4uPj9fTTT7s+AwAAVDXyBgAA8DTyBgAAoPkLAJIiIiI0duxYs8sAAAABjLwBAAA8jbwBAAB47DMAj0nLcGjqvFVKy3CYXYpPY5wAAGZg/qk8xgoAAMC7yF+VwzgBAMpD8xeAxxzLcGja/NU65gMBdP+hDP17+he67aF3def/zdB7i75TZnaea73D4dDNN9+sm2++WQ6Hd+v1pXECAAQP5p/Kq6qxMjNvAACA4BAoeYOsWjmMEwCgPDz2GUDAW/PjDj06YbEiI8LUuV0jFRaWaPr8b7Xw0w165fGhqlcrSQ6HQ++//74k6ZlnnpHdbje5agAAgkdefpHmfbxOH3y2SccyHEpKsKvfla01pG8HRUWGm11elSFvAAAATyNvVL1gyaoAgMBB8xdAQHPkFurxlz9Q+wvq6Yn7+ivCFibp+GNx7ntyjp6Y+IGmvDDK3CIBAAhieflFunv8LG3fdUSGYUiSjqbnaOq81fpm3Xb9e/xwLqoBAADAFGRVAIA/ovkLwOMKi4qVX1BkyrE/WfGTCotKdPfIHjIMw1VHdGS4bhveTY+88L42/LxHiTFWV4jPLyjyar2FRcVeOxYAAH9m5jwtSTOXrHG7mHaCYRjavuuIZi5ZoxsGdDKpuuOYqwEAAMxBVj01cioAoDw0fwF43NjHZpldgoaNe6vCdfc+OVcFBVly5BVKkobc/YYiIuK8VRoAAKbyhXm6IoZh6N2F3+ndhd+ZXQoAAABMQFYFAODMWc0uAAAAAAAAAAAAAABw7rjzF4DHvfbP4WpSv7opx96z/5hufehdjf1Ld/W/so1ruWEYeuntz/Tt+t8185VblZlxTG1avyZJmvfvO5Samuq1GrfvPuzT32QFAAQ2M+dpSRo+7i2lZTgqXJ+cYNesV2/1YkVlMVcDAACYg6x6auRUAEB5aP4C8DhbeJgiI8JNOXbzRjU0oGdbTXpvhf44mKHuFzdXQVGxPvxsk1av/10P3NZLCXHRKszPkcVikSRFRoR7tV5beJjXjgUAwJ+ZOU9LUv+ebTR13uoy71GTJKvFov4925han8RcDQAAYBay6qmRUwEA5aH5CyDg3Tf6SqWmxGrexz9oyWcbJUn1aiXp8Xuv0RWXnCdJio+P1wMPPOD6DAAAvGNI3w76Zt12/b7riJwnXVSzWixq3KCahvTtYGJ1VYu8AQAAPI28UbWCKasCAAIHzV8AAc9qtWjEgE66/uoO2n84U6EhVtWsHu+601eSIiIi9Oijj5pYJQAAwSkqMlz/Hj9c8z5epw8+26RjGQ4lJdjV78rWGtK3g6Iizb3rtyqRNwAAgKeRN6pWMGVVAEDgoPkLIGiEhoaoXq0ks8sAAAB/EhUZrlGDL9GowZeYXQoAAADghqwKAPA3VrMLABC4khLsuum6S5SUYDe7lNNyOBy65557dM8998jhcHj12P40TgCAwMH8U3lVNVZm5g0AABAcAiVvkFUrh3ECAJTHYpT3tnpUqezsbMXFxSkrK0uxsbFmlwOgHIcOHVLz5s0lSVu2bFFqaqrJFQGoCPNq+RgXwPeRNwD/wbxaPsYF8H3kDcB/MK8C8BTu/AUAAAAAAAAAAACAAEDzFwAAAAAAAAAAAAACAM1fAAAAAAAAAAAAAAgANH8BAAAAAAAAAAAAIADQ/AUAAAAAAAAAAACAAEDzFwAAAAAAAAAAAAACQKjZBQCAL4iPj9eYMWNcnwEAAKoaeQMAAHgaeQMAAFgMwzDMLiLQZWdnKy4uTllZWYqNjTW7HAAA/BrzavkYFwAAqg7zavkYFwAAqg7zKgBP4bHPAAAAAAAAAAAAABAAaP4CgKSCggI98sgjeuSRR1RQUGB2OQAAIACRNwAAgKeRNwAAAM1fAJCUmZmpSZMmadKkScrMzDS7HAAAEIDIGwAAwNPIGwAAgOYvAAAAAAAAAAAAAAQAmr8AAAAAAAAAAAAAEABo/gIAAAAAAAAAAABAAKD5CwAAAAAAAAAAAAABgOYvAAAAAAAAAAAAAAQAmr8AAAAAAAAAAAAAEABCzS4AAHyB3W7XyJEjXZ8BAACqGnkDAAB4GnkDAABYDMMwzC4i0GVnZysuLk5ZWVmKjY01uxwAAPwa82r5GBcAAKoO82r5GBcAAKoO8yoAT+GxzwAAAAAAAAAAAAAQAGj+AqhQaXqacmZOVml6mtmleFzegT808+ZhmvCP/1NBQYHZ5QAAgLPg69mloKBATz/9tJ5++ulzzhu+fq4AAMAcVZk3PC2Y8kwwnSsAwHw0fwFUyJmeJsfst+X0UjA1iouVv/pL5cyZotyPF6g0M90rx5WkzH27tWHV13pj0iRlZmZ67bgAAKDqeDu7OB3Zylu6WDlzpij/q6UyCk99gTUzM1MTJkzQhAkTzjlvePtcAQCAf6jKvOFpwZRngulcAQDmCzW7AACQpKLfflLGc4/IeeyIrHEJcubmKPutl2QfdovsQ26SxWIxu0QAAACX3E8WKnvKRKm4RNaYWDmzMmSJjVP8X59QRPvOZpcHAACAkzjz85S7eLbyli6SMz1N1sRkRfUeqOgBw2SNjDK7PAAAqhTNXwCmKz16WOmP36PQ+o2V+MREhdVvLGdOlhwLZ8jx3hsKiU9UVK8BZpcJAAAgSSpY+7WyJz2vqN4DZR9xq0ISklWyf6+y335ZGU8/qOSX31FY/UZmlwkAAAAdb/wee+gOlezcJhnO48uOHZFj1lsqWLNSSc+9QQMYABBQaP4COC1nUaGcBfke27/jgzmSLIr/+zOyRtuPHyssXPahN6vkwD455r8jW9crZbF67kn1RlGRx/YNAAC8y+PZZe5UhZ3fVvbR98hischZkC9rUori/jpeaXffIMf77yluzN/L/JxRWOD6BcwoLDinGp1FhWf9swAAAL7E49ltwbsq2blVMgz3FYZTJTu3yrHgXdmvG+mx40tkNwCAd9H8BXBa6Q/e6pXjHB3Zt8J1R4Z09+ixS5xOj+4fAAB4j7eyy5HrLit3ecGXn6jgy0/KLC9yOjWyWvTxz3cO1mEPfrENAADAX3gru5XLMJQ7d6py5041rwYAAKoYVxsAAAAAAAAAAAAAIABw5y+A00p84S2FNWzqsf3nzJys/GVLlPz6XFmj7W7rMl96QiU7tyrp1RkefezzwXXfSaNv9Nj+AQCA93g6u6Q/cqcUblPC4y/LYrG4lhvFRUq7+waFX9Cu3Mc+Hz58WO+0biNJ+ttnC1S9evWzrqF45zZz75IBAACoIp7ObkdvGyxn+tEK11sTU5QyeYHHji+R3QAA3kXzF8BpWcNtskZEemz/9n5Dlb9ssTKff0RxY/6usPqN5czJkmPhDBV++6Xi7n5EIVHRHju+JEXHxevyanEy2neR3W4//Q8AAACf5fHsMuQmZTz1gBxTXpF9xK0KSUhWyf69yn77ZTmzMmS/9oZyjx+TlKx+117r+nwuNVrDbWf9swAAIHDZ7XZd+9+84S/XNzyd3aL6XCvHrLcko5xXflmsiupzrUePL5HdAADeRfMXgOlCUqor8YlXlPHcI0q7a7iscQly5uZIkuw33qHInv09XkN0VJTaxtt15ZNPKMxPfjkCAADmiOjUTbFj/q7sKROVt/wDWWPj5MxMlyU2TgmPvqCw+o3L/Tm73a6pU3mfHAAA8BzyRlnRA4apYM1Klezc5t4AtlgV2rCpogcMM684AAA8gOYvAJ8Q3uICVZuyWAXff6OSfbtktccq4pIeColPNLs0AACAMqKvGqTIS69UweovVZpxTKE1aiuiUzdZbBFmlwYAAICTWCOjlPTcG8pdPFt5SxfJmZ4ma2KyonoPVPSAYbJGRpldIgAAVYrmLwCfYQkNVWTn7pK6e/3YhYWF2pdXqKVz52rY/X9XRAQXbgEAwKlZ7bGK6jWg0tsXFBRoypQpkqTRo0eTNwAAQJUjb5TPGhmlmGGjFTNstNmlAADgcTR/AVTImpgs+7BbZE1MNrsUj8sJCdXftx7Qlo0T1Wf07UpNTTW7JAAAcIZ8PbtkZmbq0UcflSQNGjTonPKGr58rAAAwR1XmDU8LpjwTTOcKADAfzV8AFQpJTFbMiNvMLsMrLPFJ+jG32OwyAADAOQim7BJM5woAAAJTMOWZYDpXAID5rGYXAAAAAAAAAAAAAAA4dzR/AQAAAAAAAAAAACAA0PwFAAAAAAAAAAAAgABA8xeAz0svLNasHUeUXsg7eU/GuAAA/BHzV8UYGwAAAN9CPiuLMQEA30fzF4DPyygs0ZxdacooLDG1juyiEq05kqO1R3OUU1xqai2S74wLAABnwlfmL6dh6LfMPK0+nK3t2fkyDMPUeiTfGRsAAAAcRz4rizEBAN8XanYBAOAL7Ha7+vTp4/p8smKnoWnbD2vZ/gwVO49fGA63WtS7VoJGNamuUKvF6/UCAICztzk9V5O2HNT+vCLXsgb2CI07r4YaxUZ67LinyhsAAABVgbwRePJLnFqy95iW7c9QemGJEm2h6lUrQf3rJikylHu7AABl0fwFAB3/hWj27Nnlrpv02wGtPJStoQ2T1b1GvCTpiwOZmrfrqApKnbrrvJperBQAAJyLHdn5emLjXjWNjdTYFjVUzx6hbVn5mrHjiP5vwx69dFFD1YgK98ixT5U3AAAAqgJ5I7Dklzj1yPrd2plToBPPqTlWWKLZO49q7dEcPdOuPg1gAEAZNH8B+I1Cp6GCUqdXj3kgr0hfHMzSLU2rq2etBNfyAfWSZAuxaPr2I7qmbqKqR3rmIvGpFDrNfzwlAABny4x5XZJm7zqqlIgwPXRBbYWHHL9Q1jIhSo+1qaN71+7Uwt1puqVZqtfrkpjbAQAAfJVZ2XXhnjS3xu8JhqSdOQVauCdNg+sne7UmMisA+D6avwD8xkM/7PbYvp0lJcrauUWSFNewuayh7v88vr3tsN7edrjcn717zU6P1QUAQKDy5LxeGTd8va3c5csPZGr5gUyPHPN0eQMAAOBcFRQUaN68eZKkIUOGKCIiwuSKAoPZ2bU8hqR5u9I0b1ea2aUAAHwMVxsAQJKzqFC/vjlBktTxn5O4GAsAAKoceQMAAHhaZmamxo0bJ0nq2bOnUlPNeaIJAAAwD1cbAPiN59rXV8MYz3xj9dChQ2qz93dJ0qR2tVy/HP2UnqunNu3TE23rqkV8lNvP/JyRqyc37tPjbeqoZUK0R+o6lZ05BT75zVMAACrDk/P6qfx93S4l2EL10AV13JYbhqFH1u9RTFiIHmldp4KfPjcV5Y0TmNsBAAB8k1nZ9Y5vf1d6YUmF6xNtoXqjc2MvVkRmBQB/QPMXgN+wWS2K+O+7+apaRIhVMgzX5xPHaZ9sVz27TW9sOaRHW9dWXfvxoL/bUaDJWw+pgT1CFybZZbFYPFLXqdis3j8mAABVxZPz+qkMqJekl385oI/3patf3SSFWS0qLHVq9q407cgp0BNt63o9b5zA3A4AAOCbzMquvWslaPbOo2Xe+StJlv+u93ZdZFYA8H00fwHgFKwWix69oI7+8eMe3bVmpxrHRMiQtCOnQDWjwvVI69qmNH4BAMDZuSw1TntzC/XO70e0ZO8x1YqyaY+jQLklTt3cpLraJtnNLhEAAACQJPWvm6S1R3O0M6fArQFskdQwJkL96yaZVRoAwIfR/AWA00iNCtdrFzfSqsPZ2pSeK+n4XUOdq8UozOr9b30CAICzZ7FYNLJxdXVPjdMXB7OUXlis8+ITdUXNeNWICje7PAAAAMAlMtSqZ9rV15K9x7Rsf4bSC0uUaAtVr1oJ6l83SZGhXJcCAJTlt7PDzTffLIvFot9++8217KuvvpLFYlGXLl3cti0sLFRSUpIsFosyMzMlSePHj1doaKjsdrvbn3Xr1kmSRo0apfDwcMXExCguLk5NmzbVHXfcoV27dnntHAH4jjCrVd1rxOvelrV0b8ta6pYaR+MXAAA/VtceoZuaVNf959fWjY2r0fgFAACAT4oMtWpowxRN69pUS644T9O6NtXQhik0fgEAFfLLGSInJ0fz5s1TYmKipkyZ4rYuJiZGu3fv1vbt213LlixZomrVqpXZz9VXXy2Hw+H2p0OHDq71Y8aMUU5OjrKysrRs2TKFh4erbdu2bg1nAJ6XYAvV0AbJSrDxsIKTMS4AAH/E/FUxxgYAAMC3kM/KYkwAwPf5ZfN37ty5io6O1vPPP6/33ntPxcXFrnVWq1U33nijpk2b5lo2bdo03XTTTed0zAYNGujVV19Vp06d9Pjjj5/TvgCcmURbmIY3qqZEW5jHjmG329W9e3d1795ddrt/vOvPG+MCAEBVC+b563R5I5jHBgAAVA1/vL7hy8hnZTEmAOD7/LL5O2XKFI0YMUJDhw5Vbm6uPvzwQ7f1o0aN0rvvvqvS0lLt379fP/zwg/r3718lxx48eLBWrlx5ym0KCwuVnZ3t9geAb7Pb7Vq0aJEWLVrEL0cA/AJ5A/A/5A0A/oa8Afgf8gYAAPC75u+vv/6qNWvWaOTIkbLb7Ro4cGCZRz83a9ZM9erV0/Lly/XOO+/o+uuvl81mK7Ovjz/+WPHx8W5/CgsLT3n8WrVqKT09/ZTbPPvss4qLi3P9qVOnzpmfKAAAwCmQNwAAgKeRNwAAAAD/43fN3ylTpqh169Zq3bq1JGnkyJFatmyZ9u/f77bdTTfdpKlTp2r69OkVPvK5b9++yszMdPtTXpP4ZPv371diYuIpt3n44YeVlZXl+rNv374zOEMAZigpKdGSJUu0ZMkSlZSUmF0OAJwWeQPwP+QNAP6GvAH4H/IGAADwq+ZvcXGx3nvvPW3btk2pqalKTU3ViBEjVFpaqunTp7tte/3112vp0qWKjIxUu3btqqyGBQsW6LLLLjvlNjabTbGxsW5/AJy7g3kOjd/4jQ7mOap832lpaRo5cqRGjhyptLS0Kt+/J3lyXAD4LvIGUPU8Paf6Y94gZwDBjbwB+J+qzhvBngWC/fwBAP7Jr5q/H3zwgbKzs7VhwwZt3LhRGzdu1KZNm/TYY49p6tSpMgzDtW1MTIxWrFihefPmVcmx9+zZo/vuu09r1qzR+PHjq2SfAM7MwXyHnti0WgfzzQ/cf+Rm69fMNOUWF5ldik+NCwAA/sxX5tSsogL9knFUh/NzTa1D8p0xAQAA5gj2LBDs5w8A8E+hZhdwJqZMmaJhw4apefPmbsvHjRunCRMmuDV/Jal9+/an3N9HH30ku93utmzGjBkaMGCAJGnSpEmaMmWKLBaLqlWrph49emjDhg1q2LDhuZ8MAL+06vA+PbRhpVYf+UOSZA8N102NW+nZC7spOizc5OoAAIA/O1qQp7/98KXm7vpNhc5SSVKvmg30r/Y9dH5CisnVAQAAeI+juEgv/7pOk7dt1IF8h2pG2nVb0za677wOsnP9BQCAU/Kr5u8nn3xS7vLk5GTl5+dLkjIzM8vdpn79+m7N4fHjx5/yDt7p06eXeZQ0gOD2zeF9umL5HLVOqKZZXfupTnSMlh/YpZd+XadNGUf0ec+hCrOGmF0mAADwQ9lFhbps6SwdLczTk2276pJqtbU165gm/PK9unw6Q99ddaNaxCebXSYAAIDHOYqL1G3pTG1MPyKnjl/P/SMvR+M3rtLivdu0svcIGsAAAJyCXzV/AUCS8kuKq/xxy7klxTJO+lze/v+67gu1SaimpVcMUXjI8SZv28Tq6lKttnp9Pk8zd/yi6+o3L/NznpZfUuz1YwIAEMg8kTWkU+eNV3/7QTtyMvTdVTeqaWyiJKlNQjX1rtlQXZfO0CMbVmpG12uqvKbTIWcAAADJc/moPM9vXuPW+D3BKUMb04/o+c1r9FCrTl6phSwEAPBHNH8B+J0uS2dW/U4duVKoRZLUeMlkyR5d4aaJc18pd/lN336im74t/wkFAADAf3gka0iVyhsXfjS93B/d6ciSfdZLnqkLAADgNDyWj86QU4ae2vytntr8rdmlAADgs6xmFwAAAAAAAAAAAAAAOHfc+QvA76zqPUJtEqtX6T4duQ4NX7xGkjRrxH2yR9vd1mcUFajx+2/qsQs6697zOritK3E61fKDt9WzZgP9+6Irq7SuytiYfthnvoELAEAg8ETWkE6dN65d8b4yiwr1Za9hZX7u/h++1Pt7t+n3AbcpxOrd7++SMwAA8C92u10dO3Z0fa4qnspH5Wm6aLIO5DsqXF8z0q5tA2/zSi1kIQCAP6L5C8DvRIaGKTosvEr3GR2fqC8++6zi9WHhGtqghV76bZ361mns+oWn1OnUw5tWan+eQ3c3b1fldVVGZGiY148JAEAg80TWkE6dN+5q0U79vlyoKb//pLubt5PFcvzx0F8c3K3pv2/Wg+d3VKwtosprOh1yBgAA/sVut2vZsmVVvl9P5aPy3NGsrcZvXFXmnb+SZJVFdzRr67VayEIAAH9E8xcAKunlDlfop4yjavfRdPWq2UB1omP12YFd2uXI0sQOl6ttUqrZJQIAAD91de3Guv+8i3TP959r8raN6lKttrZkHdPKw/vUs2YDPXJBZ7NLBAAA8Ir7zuugxXu3aWP6EbcGsFUWtUmspvv+9EQ2AADgjuYvAEgqKSnRmjXHH8PYqVMnhYaW/ecxwRah1X1u0Iydv2jOrt+0/tghdateV3O7XagOyTW8XTIAAPAzp8obFotF/+rQQ1fVbqg3t23UD8cOKSUiSrO69tN19Zsr1MuPewYAAP6pMtc3fJ09LFwre4/Qy7+u0+RtG3Ug36GakXbd1rSN7juvg+wmPHUNAAB/4n+zPwB4QFpamq6++mpJ0pYtW5SaWv5dvJGhYbq1aRvd2rSNF6sDAACBoDJ5o0eN+upRo76XKwMAAIGistc3fJ09LFyPtb5Ej7W+xOxSAADwO3x9HIDfqBFp1+OtL1GNSLvZpfgUxgUAgKrBnFoWYwIAQHAL9iwQ7OcPAPBP3PkLwG/UiLJrfJuuZpfhcxgXAACqBnNqWYwJAADBLdizQLCfPwDAP3HnLwAAAAAAAAAAAAAEAJq/AAAAAAAAAAAAABAAaP4CAAAAAAAAAAAAQACg+QvA56VlODR13iqlZTjMLsXnMVYAADMxD50ZxgsAAMD7yGCVwzgBgP+i+QvA5x3LcGja/NU65sGwGRERodatW6t169aKiIg47faGYcjpNDxWz9nyxlgBAFARX52HfGXe/nPe8NXxAgAA/utMr28EIzJY5TBOAOC/Qs0uAAB8QXx8vFauXHna7fYfytB773+nL7/dovzCYjWsm6JBvS/U1Ze3ltVq8UKlAACgsrbvOqwZi9Zo1Q/bVVxSqhaNa+j6qzuoR+cWptTz57xxOP2QKXUAAIDAVdnrG/ANeflFmvfxOn3w2SYdy3AoKcH+/+3deXhU5f3//9dMJpnsOxAkBKRssqNCVVywdUMUEJeP4IZVq4JLt0/rUkRrf0Vt1VqX+gHcaEGlRUXLz7Vqi6iIFRBUAgjIYgJkTyDbJPf3DzpjQrZJMjNn5szzcV25HOacuc/7PifxvDLvnDOacuZoXTJ5nBIT4qwuDwAQoWj+AoCfduw+oJvuWqp4d6xmTB2vrIxkfbJ+h/6w8E19ua1Av7rhHDkcNIABAAgH677YpV/8f39Tr+xU/eiSk5WU6Na/P9mieQ+/qp17ivWjS062ukQAAABEsUPVdbr57qXaumO/jDl8l5oDJZV6etlqrVq7VY/ePZMGMACgS2j+AogYtXX1qq6pC8rYHo9HW7ZskSQNHjxYLlfL/z0+uPBtZaQl6qG5/6OUpMO3Tjrz5GE6/t/99eDCt3Tq+EEaOzwvKPX5q7au3tLtAwAgBfec7Y/GRqP7n3xdQwb00vxfTldc3OHz+tmnDtdfX/5Yz/xttU4eN1B9e2eGtK4j8wbnbQAAEGgej0f5+fmSpCFDhrT6/gYOszqzLlnxcbPGr5cxRlt37NeSFR/r8mknWFQd7zEBQCTj7A8gYsyZuzRoY9fUlOuzNx6SJB17zs8UH5/W5roX3vDnVp//1X3Lg1IbAACRJpjn7M7YW1im8655tNVl1/zyuRBX07m8AQAA0BVFRUWaMGGCJGnz5s3KycmxuKLwFS6ZtTXGGC1e/pEWL//I6lIAABHIaXUBAAAAAAAAAAAAAIDu48pfABHj8XtnalD/XkEZu7CwUGNGPy5JWvboDS3+MnZ/caUu/8ki/eLHZ+msU4Y3W9bYaHTNr57VsIFH6X+vPzso9flr6859Yf2XqwCA6BDMc7Y/vtpWoFvveUH333Zhi49kqKmt18xbFmrKmWM066KTQlrXkXmjssbBeRsAAMAiVmfWmbcsVFFpVZvLszOStfRP14WwouZ4jwkAIhfNXwARwx0Xq4T4uKCMnRAfJ4fD4Xt85Hb69cnS+NFH64VX1+rU8YOVkZbkW7b89f9ob2GZ7pg9OWj1+csdF2vp9gEAkIJ7zvbH2OF5Orpvtp5b/qHGDMtTYsLhWowxeu7vH+pgda2mnjkm5DUemTfqGkO6eQAAADRhdWadetYYPb1sdYvP/JUkp8OhqWeFPq82xXtMABC5aP4CgJ9+es2ZmjN3ia746VOaNHGEsjKS9cn6HVr7+U5dfO5xGnVMrtUlAgAASQ6HQ7+6cZJ+du+LuuKni3TOaSOUlOjWv9ds0Rdbv9XsKybqqF7pVpcJAACAKHbJ5HFatXartu3Yr8YmDWCnw6GBR/fUJZPHWVgdACCS0fwFAD/l9s7Qwvuv1AuvfqK3Vn2pQ9V1Gtivh+bder5+OOEYq8sDAABNDB90lBbOv1JLX/1EK95er7o6j4YPPkq/v+MinTD2e1aXBwAAgCiXmBCnR++eqWUr1+rVtzeouLRKWRnJmnLmaF0yeZzv7jUAAHQWzV8A6ISeWam65eozdMvVZ1hdCgAA6EBenyzdduMk6cZJVpcCAAAAtJCYEKdZF03QrIsmWF0KAMBGaP4CCHtZGcm6+uIJyspIDto24uPjdcwxx/geR6pQ7CsAANrCeah9R+YNl9vF/gIAAAFll/c3gonM6h/2EwBELodp7RPlEVAVFRVKS0tTeXm5UlNTrS4HAICIxnm1dewXAAACh/Nq69gvAAAEDudVAMHitLoAAAAAAAAAAAAAAED3cdtnAJDk8XhUVFQkScrOzpbLxf8eAQBAYJE3AABAsJE3AAAAV/4CgKSioiINHTpUQ4cO9f2SBAAAEEjkDQAAEGzkDQAAQPMXAAAAAAAAAAAAAGyA5i8AAAAAAAAAAAAA2ADNXwAAAAAAAAAAAACwAZq/AAAAAAAAAAAAAGADNH8BAAAAAAAAAAAAwAZo/gIAAAAAAAAAAACADbisLgAAwkF8fLwGDBjgewwAABBo5A0AABBs5A0AAEDzFwAkpaen67PPPrO6DAAAYGPkDQAAEGzkDQAAwG2fAQAAAAAAAAAAAMAGaP4CAAAAAAAAAAAAgA3Q/AUASYWFhUpPT1d6eroKCwutLgcAANgQeQMAAAQbeQMAAND8BQAAAAAAAAAAAAAboPkLAAAAAAAAAAAAADZA8xcAAAAAAAAAAAAAbIDmLwAAAAAAAAAAAADYAM1fAAAAAAAAAAAAALABmr8AAAAAAAAAAAAAYAMuqwsAgHAQHx+v3Nxc32MAAIBAI28AAIBgI28AAACu/AXCWENJkSqXLFBDSZHVpQSd1XNNT0/Xpk2btGnTJqWnp1tSAwAACA2rcocVecPqjAUAAPwTqHN2JLy/QT4BACC4aP4CYayxpEhVzy9SYxSE4WiaKwAAsFaoc4dn905VPPWISu79hcr+9FvVbvpMxpiQbJuMBQBAZAjlOdsYo9p1a1T2x9+o5N5fqOLZx+Qp2BP07XqRTwAACC5u+wxAktRYfUgHX3leh954WY0lRXJmZivxnAuUNG2GnAmJVpcHAAAQkQ6ueEEVix6WMzVdsYOHqW7TOlW/9aoSTp+ktJ/cJUdMjNUlAgCAKGLq61V6/x2q/fhfcuUdrZheR+nQG6/o4EtLlHbT7Uo8a4rVJQIAgG6i+QtAjdWHVHzbDfJs3yKZxsPPFe9X1dKFqvn4X8q670nbN4ALCws1fPhwSdIXX3yhnJwciysCAACRrnbTZ6pY+JCSpl+ulCtu0L7iEg0fPlznZSfrPsebcvUfqOQLr7C6TAAAYCMdvb9R9cIi1X76oTLuuF/uEyfK4XDI1NaoYuHDKn/sd4odOFSxAwZbUToAAAgQmr9ABGisq1VjTXXQxq/6+2J5tudLR95+0DTKsz1fVX9frOSLrwra9qXDc7RaQ0OD1SUAAIAQCnbGOvjK84rJ7a+kGdfKNDTI1NbI0dCglfvKdd+Vl+rgay8qYdJ0OZzB+zSecMhYAADAf93NJ9684X3cdCxTX6eDK5cr8Zxpijv2BJnaGnnfCUq++ibVfLpaVSueV9qNv+zOFDpEPgEAILho/gIRoOSX11m3cWN08MWndfDFp62rAQAAIAhClbH2XzxRklTX2KireiZJkupX/1OxTqf2X3J6SGoAAACRobv5pGneqLvxIu1r5Y/MDv3jbzr0j7+1+vqaf65UzT9XdqsGAABgreD9iTkAAAAAoFUOqwsAAAAAAAC2xJW/QATIfGBhUD9v5cCPL1JjyYE2lzsze6jHgr8HbfuSVL99i7VXOAMAgKgT7IxV9eIzOvTai8p65C+Kyeqhffv26bnRY+RySL84/WjF9eiljF//Pmjbl8hYAABEmu7mE2/ekKRfvP139erVq9nykttvlGJilHHPH+WI+e6tYU/BHhX/9ColX36Dks67uMvb9wf5BACA4KL5C0QAZ5xbzviEoI2fOGm6qpYulExjy4UOpxInTQ/q9qXDcwQAAAilYGespGkzVP3+6yqdd6uSL/ux1LOPxqcn6sa+mdK+vUr56V1kLAAA0Ex384nDHS9Pk8dHjpVy1WyV3HWryu67Q8kXX6WYXkepbsOnqly6QDE9eyvpXN4DAgAg0tH8BaCkaTNU8/G/5Nm+pXkD2OGUa8BgJU2bYV1xAAAAESomLUNZ859U+WPzVf7gPEnSUyNytfVgrfSTuxU3dKTFFQIAgGjjHjNeGXP/oIqFD6nkjtmHn3Q45D7+JKXddIecicnWFggAALqN5i8AORMSlXXfkzr4yvM69MbLaiwpkjMzW4nnXKCkaTPkTEi0usSgc7lcvlshuVz8rxEAAASGK6ePsn77mDwFe1Sy+QtdfetPtK22UatHHGt1aQAAwIb8eX8jftwEuY87UfXbNstUVSgmt59cPXuHskwAABBEdDgASDrcAE6ZcY1SZlxjdSmWyM7OVn5+vtVlAAAAm3L1zlXP3rla+flXVpcCAABszN/3NxxOp+IGDwtBRQAAINScVhcAoG3OzGwlz7hWzsxsq0sJumiaKwAAsFY05Y5omisAAJEsms7Z0TRXAACs4DDGGKuLsLuKigqlpaWpvLxcqampVpcDAEBE47zaOvYLAACBw3m1dewXAAACh/MqgGDhyl8AkFRYWKgePXqoR48eKiwstLocAABgQ+QNAAAQbOQNAADAZ/4CwH/V19dbXQIAALA58gYAAAg28gYAANGNK38BAAAAAAAAAAAAwAZo/gIIGyW19Vr69X6V1PIXqm1hHwEA0HWcR/3DfgIAAMFG3ugY+wgA0FU0fwGEjdJaj17YUaTSWo/Vpeigp0Fv7CnVs1v3acU3xWFRkxRe+wgAgEgTTudRY4zyyw9pydf7tXjbPn1WXKVGY6wuS1J47ScAAGBP5I2OsY8AAF3FZ/4CwBH+U16nxV9tVW1jo3rGx6qk1qPntu3TFQN76YJ+WVaXBwAAIlxMQqL+9E2VvqgqVWpsjGKdDv19Z7H6J7s1d0yeesTHWl0iAABA1Kj2NGrFrmK9ubdUJbUeZbpdOrtPhqbmZSnBxbVTAIDIQ/MXAJpIHjBEC3dX6aReqbpmUC9lxceqqr5By3Yc0DNb96lHfKxO7pVqdZkAACCCDZ59p74+5NHto3I1vkeKnJK+Kq/WQ5v26jfrdumPJwxQjMNhdZkAAAC2V+1p1B3/2antlTXy3oOluNaj57cf0JoDlfrdcf1pAAMAIg7NXwBhp7bRqKahMaTb9DicysjKUu6FVyozzqk5Q3srxulQTUOjXE6HZgzooR2VtVq+s0jHZyeHtLamahvD43aQAABEMiuyhnQ4b/Q6ZoR6nHi6Lu6dpLFZyar/77l9QEq8bhl2lH792Tdas79Sx5I3AABAF7hcLmVkZPgehzurcpnX8m+KmjV+vYyk7ZU1Wv5NkS7qn21FaWQyAECXhX8CABB1bvt0pyXbHfl/r0mSiuoaNeNf+W2ud8l7m0NVEgAACAKrsoYkDbl3gSTpL3ur9Je9rWeK+zbuCWVJAADARrKzs7Vjxw6ry/CblbmsI0bSsh1FWrajyOpSAADoFO5ZAQAAAAAAAAAAAAA2wJW/AMLOfcf314CUeEu2vSC/UP8pqtKfThggd8x3fx/TaIzmfbZLMU7p7rH9LKlNOnzLoXD+q1gAACKBlVmjqKZecz76WlcN6qlzczObLfu85KB+u2G35o3pq+EZSZbUJ5E3AABA6FiZyyTphg+3qaTW0+byTLdLT540MIQVfYdMBgDoKpq/AMKO2+lQfExob0xQVFSkkSNHKr53Xx33h2f1+417dM3gHB2dEq/C6jot+fqAtlRU6+6xeSGvrSm302HZtgEAsAsrsoZ0OG+MGzlS/a/5uf7qOFdup1M/PCpdLodDHx+o1JObC3RMWoKOzUqWw2HdOZ+8AQBA5PK+vyFJGzduVHa2NZ9X6y+rcpnXOX0y9Pz2Ay0+81eSHP9dblV9ZDIAQFfR/AUASR6PR9XV1arevkU356XouYJq3bpmu+KcDtU1GiW5nPrJ8KM0NivZ6lIBAECE8uaNzU/er6nTpmlBfqEWbimUUw55jNHYzCT9fEQfSxu/AAAgsnnzhvcx2jc1L0trDlRqe2VNswawQ9KAlHhNzcuyqjQAALqM5i8AHGFIcqwWTsjVf4qrVFBdp4w4l77fI6XZbaABAAC6ynjqNSs3WVcPz9RnxVVqaJSGZyTqaAtveQgAABCNElxO/e64/lqxq1hv7i1VSa1HmW6Xzu6Toal5WUpw8V4QACDy2ObsNWvWLMXFxSk5ObnZ1/79+yVJEydOlNvtVkpKitLS0jRixAj9/Oc/14EDB3xj7Ny5Uw6HQ2VlZZKku+++W9OmTWuxrYkTJ+qPf/xjCGYFwCoxTofG90jR1LwsnZqTRuMXAAAEXK+EOE3KzdR5eZk0fgEAACyS4HLq0gE99Mwpg7XijGF65pTBunRADxq/AICIZYszmDFGDQ0Nmj17tqqqqpp99ezZ07fe/fffr8rKSpWVlWnZsmXau3evjjvuOO3bt8/C6gF4ZbhduvTobGW4uSlBW9hHAAB0HedR/7CfAABAsJE3OsY+AgB0VcQ2f/v376/58+frhBNOUGJiog4dOuT3ax0Oh4YNG6a//vWvSk1N1YMPPhjESgH4K9Mdq5nf66lMd6zVpYQt9hEAAF3HedQ/7CcAABBs5I2OsY8AAF0V0X829Oyzz+rVV1/VwIEDdcUVV3T69S6XS9OmTdPbb78d0Lpqa2tVW1vr+3dFRUVAxwcAACBvAACAYCNvAAAAAJEnYq/8laQbb7xRQ4YMUUxMjOLi4vTnP/9Z6enpvq8hQ4Z0OEafPn1UUlLS5vKVK1c2GzM9PV0ffPBBu2POnz9faWlpvq++fft2em4AQsvlciklJUUpKSlyuSL672IARAnyBhB5yBsAIg15A4g85A0AABDRzd+8vLxm/77xxhtVVlbm+8rPz+9wjL179yozM7PN5ZMnT242ZllZmU4++eR2x7z99ttVXl7u+9q9e7d/EwJgmezsbO3evVu7d+9Wdna21eUAQIfIG0DkIW8AiDTkDSDykDcAAEBEN3+dzu6V7/F4tGLFCk2cODEwBf2X2+1Wampqsy8AnVdwqEp3r1+lgkNVVpcS1thPQHQibwDRh3N+x9hHQGCRN4DmOM98h30BAED4iujmb3ds3rxZV111lcrLy/Wzn/3M6nIAtKKgukr3bFitgurw+EXikKder+zaoue2bdSnRQUyxlhdkqTw208AACA4wu2cv62iVH/5epNe2PGlimuqrS5HUvjtIwCAvVh9njHG6OMDe/Xsts/12u6tqm3wWFKHZP2+AAAAbbPVBz888cQTWrRoUbPnVq1apbFjx0qSfvWrX2nu3LlyOp3q06ePJk2apE8//VQ9e/a0olwAYaSoqMj3/4p169a1uDXSgi3rddt/3ldpXY3vufHZvbXklCkamJoR0loBAEBk6ihvRIqyuhpd/cFKvbJ7q+85tzNGtxxzvOYfe5piunmHJgAA0NKXZUW6bNWrWl+y3/dctjtBD437oa743gjfc3bJGwAAoOsitvm7c+fOZv9+9tln9eyzz7a5/vvvv9/hmP379292Jd/dd9/d5bEARBaPx6PKykrf46aWbv9C13/0hn40cJRuG3mC+iWl6e2CHfrZ2nf1g7ee14bzf6QMd7wVZQMAgAjSXt6IFMYYTfnn37WprEhPn3SuLu4/VFWeOv1f/nr95vPVcjik+4873eoyAQCwlX3VB3X6m0vVMz5Jb57xP5qYk6evK0v1288/1JUf/ENpsW5NyRskyR55AwAAdE/ENn8BRI9qT70O1tcFdRsHPfUyTR57t9dojOatX6Xzcr+nR8b9UA6HQ/WNDZrYK0+vnn6hRr72lJ7M/0y3HHN8UOtrT7Wn3rJtAwCA0AtFNmrLe4XfaNX+PVpx+nT9sHd/SVKKK06/GD5edY0NeujLTzRnyLHKcidYUh+5CAAQCqE+Fz/y5Vod9NTr1R9cqJ7xiapvbFBeUqr+74SzVVBdpbvWr9IPcvLkcDjafH8j0DjnAgAQvmj+Agh7J7+xJPgbqToouRySpIErFkjJSc0Wb6ssU8rzD7f60jvW/Vt3rPt30EsEAACQQpSNOjD1vZfaXNZv+Z9DWAkAAKFn1bl4wEtPtrnM955FB+9vAAAA++PDmAAAAAAAAAAAAADABrjyF0DY++CcyzQms1dQt1G4b5+OvfNRSdJnU3+snF6Ht1fb4NGQVxbq4n5D9fvjm39+3UFPvYa+skBXfW+kfjv21KDW1571JfvC4gogAAAQGqHIRm1ZtvMr/ejD1/Xp5Ks0NC2r2bLFX2/SnDVvacP5P9KAlHRL6iMXAQBCIdTn4ls/eUcr936tL6dcq7iYmGbLbvz4Tb1fuEubplyjGKezzfc3Ao1zLgAA4YvmL4Cwl+CKVVJsXFC3keSKlaPJY+/2kmLjdPMxx+k3G1ZrbFYvzRo4UrHOGH17qFI//ugN1TQ06JZjjg96fe1JcMVatm0AABB6ochGbZk5YLju2bBas1b//3rxtKkalp4tY4ze+naH7lz3b12QN1gjM3taUptELgIAhEaoz8U/Gz5ei7dv0o8/fkOPf/8s9UxIUm2DR/+3Zb3+sv0LPXT8D5TqjpfU9vsbgcY5FwCA8EXzFwAkuVwuJSQk+B43defIk/RNVYV+/NEb+vW6f+uoxGRtKi1Sgsull0+frqMturIFAABElvbyRqRwx7j0+hmXaNI7yzR8xSINT8/WQU+9dlaV67ReffXUhHOtLhEAANsZlp6tF0+dqss/+Idy//64RqT30K6DFSqurdZNQ4/VrcPG+da1Q94AAADdQwIAAEnZ2dkqKChodVmM06mnJpyrW445Ti/s+EpldbW6dtBoXT5guNLi4kNcKQAAiFTt5Y1Ickx6tvIv+LGWf5Ov1fv3Ks7p1Pl9B+r0nH5yOBwdDwAAADrtgn5DtDsnT3/5epM2l5fo3D7f08wBwzQsPbvZenbJGwAAoOto/gKAn0Zn9tJoiz5fDwAAIJy4Y1yaOWC4Zg4YbnUpAABEjUx3QrOrfAEAAFrjtLoAAGhL74RkzRs9Qb0Tkq0uJayxnwAAiA6c8zvGPgIABBPnme+wLwAACF8OY4yxugi7q6ioUFpamsrLy5Wammp1OQBaUVRUpHHjDv/17Nq1a5Wdnd3BKwBYhfNq69gvQPgjbwCRg/Nq69gvQPgjbwCRg/MqgGDhts8AIMnj8ai0tNT3GAAAINDIGwAAINjIGwAAgNs+AwAAAAAAAAAAAIAN0PwFAAAAAAAAAAAAABug+QvAUkWlVXp62QcqKq2yupSIwn4DAKAlzo9dx74DAACBRr7oGvYbAKC7aP4CsFRxaZWe+dtqFYdZoPU0NGrLjn3a/HWBamvrrS6nhXDdbwAAWCmcz4+FB8q1ZccBudzJVpfSqnDedwAAIDKRL7qG/QYA6C6X1QUAQLh564N8rXjnJe0vrpQkpSTF68JJx2rWRRMUE8PfzAAAAP99/c0BPfLMO1r3xS5J0tBTblJl0TbtK6pUTk6OxdUBAABEl0PVdVq2cq1efXuDikurlJWRrClnjtYlk8cpMSHO6vIAAAgImr8A8F+xsbHK7DtOC1/8WGefOlzn/XC04mJj9O6Hm/WXlz5SUWmVfnXDJKvLBAAAEWLXtyW6ad4S9chM0V23nK/UROmyq29SVt4JuuuRN/T0A72VlRGeVwIDAIDIFRsba3UJYelQdZ1uvnuptu7YL2OMJOlASaWeXrZaq9Zu1aN3z6QBDACwBZq/AMJCbV29qmvqLNt+WnqmvsrfrktvXqBzTx+pGy+f6Ft2zf+crF7ZKfrTs+9q2pljlNcny7I6vWrrwu9W1AAAhAurc4XXM8s+UFKCWw/9+hIlJbolSRs/eUPFZVW67leLtXTFGl176SkWV3kY2QIAAHvIycnRgQMHrC6jmXDJZktWfNys8etljNHWHfu1ZMXHunzaCRZV9x1yGQCgu2j+AggLc+YutboEn5ffXKeX31zX6rJrb1sc4moAAEBnhVOukKQLrn+i1eeXrfxUy1Z+GuJqAAAAQivcsllrjDFavPwjLV7+kdWlAADQbXx4JQAAAAAAAAAAAADYAFf+AggLj987U4P697Js+0VFRTp78sXKGjxFt9/wQ50+YVSz5QX7yzTrF8/o59eepbNOHW5Rld/ZunNfRPzlLAAAVrA6V0iHrx655pfPadDRvXT77EmSDueNH5x+miTpBxfdKZcrVn+482Iry/QhWwAAYA9FRUWaMGGCJGn16tXKzs62uKLwyGaSNPOWhSoqrWpzeXZGspb+6boQVtQ6chkAoLto/gIIC+64WCXEx1m2/ViXU99+84USc8bruZfW6thRg5TTI02SVFlVoz8+/U+lJifo7NNGKN4da1mdXu4462sAACBcWZ0rvC489zg9+sw/dcq4QTrj5GMU63LqwIEDyup7vL7ctk+/+dnUsKhTIlsAAGAXHo9H+/bt8z0OB+GSzaaeNUZPL1vd4jN/JcnpcGjqWWPCok5yGQCgu2j+AkATuza+otx+P9elNy/QuFH95Y5zac36HYpxOvTA7ReHReMXAABEhgvOGqsvt36r3/zpNS1Z8bF690jW4JOulzspU+f/YJgmnjDE6hIBAACixiWTx2nV2q3atmO/Gps0gJ0OhwYe3VOXTB5nYXUAAAQOzV8AaKLuUKkevH2K1n21Xx9+9rUqD9Zq5tTxOv+MMcrOSLa6PAAAEEFiYpyae/N5OvvU4Xr9/U36dl+JDpXv1Z4vV+rKR9+Rw+GwukQAAICokZgQp0fvnqllK9fq1bc3qLi0SlkZyZpy5mhdMnmcEhOsv+oXAIBAoPkLAEdISojTRecer4vOPd7qUgAAQIRzOBz6/pgB+v6YASosLNTQP//E6pIAAACiVmJCnGZdNEGzLppgdSkAAASN0+oCAES3rIxkXX3xBGVxVW2nsN8AAGiJ82PXse8AAECgkS+6hv0GAOgurvwFYKnsjGT96JKTrS4j4rDfAABoifNj17HvAABAoJEvuob9BgDoLpq/APBfMTExVpcAAABsjrwBAACCjbwBAEB0o/kLAJJycnJUXFxsdRkAAMDGyBsAACDYyBsAAIDP/AUAAAAAAAAAAAAAG6D5CwAAAAAAAAAAAAA2QPMXACSVlZVpxIgRGjFihMrKyqwuBwAA2BB5AwAABBt5AwAA8Jm/ACCppqZGe/bs8T0GAAAINPIGAAAINvIGAADgyl8AAAAAAAAAAAAAsAGavwAAAAAAAAAAAABgAzR/AQAAAAAAAAAAAMAGaP4CAAAAAAAAAAAAgA3Q/AUAAAAAAAAAAAAAG3BZXUA0MMZIkioqKiyuBEBbKisrfT+rlZWVSkxMtLgiAG3xnk+9P7M4jLwBhD/yBhA5yButI28A4Y+8AUQO8gaAYKH5GwLFxcWSpL59+1pcCQB/DB482OoSAPihuLhYaWlpVpcRNsgbQGQhbwCRgbzRHHkDiCzkDSAykDcABBrN3xDIzMyUJO3atcv2/xOvqKhQ3759tXv3bqWmplpdTtBEyzwl5mpH0TJPibnaVXl5ufLy8nznVxxG3rCfaJmnxFztKlrmGi3zlKJrruSN1pE37Cla5hot85SYqx1Fyzyl6JoreQNAsND8DQGn8/BHK6elpdn+hOWVmpoaFXONlnlKzNWOomWeEnO1K+/5FYeRN+wrWuYpMVe7ipa5Rss8peiaK3mjOfKGvUXLXKNlnhJztaNomacUXXMlbwAINP6vAgAAAAAAAAAAAAA2QPMXAAAAAAAAAAAAAGyA5m8IuN1uzZs3T2632+pSgi5a5hot85SYqx1Fyzwl5mpX0TTXzoim/RItc42WeUrM1a6iZa7RMk+JuSK69gtztZ9omafEXO0oWuYpMVcACASHMcZYXQQAAAAAAAAAAAAAoHu48hcAAAAAAAAAAAAAbIDmLwAAAAAAAAAAAADYAM1fAAAAAAAAAAAAALABmr8AAAAAAAAAAAAAYAM0fwNk9erVGj16tBITEzVmzBh99NFHba67cuVKnXrqqcrIyFDPnj110UUXac+ePc3WeeWVVzRo0CAlJibq5JNP1ubNm4M9Bb91Zq4FBQWaMmWKjjrqKDkcDq1fv77Z8vfff18Oh0PJycm+r5tuuinIM/BPIOcp2eeYdrT+zp07WxzT888/P9hTaFN9fb1uuukmZWRkKDMzUzfffLM8Hk+X1u3MWKEWyHnOmjVLcXFxzY5hR98TodSZuT722GM6/vjj5Xa7NW3atBbLKyoqNHPmTKWmpqpXr1669957g1x95wRyrhMnTpTb7W52XL/99tsgz8A//s6ztrZW1113nY4++milpKRo6NChevrpp5utE+7HtLvIG62L5LwhRU/mIG/4ty55IzyQN+yXNyQyh7/IG60jbzQXrseVvOHfuuGcN6ToyRzkDfJGNOcNAEFm0G3FxcUmPT3dLFiwwNTU1JgFCxaYzMxMU1pa2ur6S5YsMf/4xz9MZWWlqaqqMldffbU58cQTfcs3b95sEhMTzWuvvWaqq6vN3LlzzeDBg019fX2IZtS2zs61sLDQPP7442bNmjVGklm3bl2z5e+9955JS0sLet2dFeh52umYdrT+jh07jKQ2Xx9qd911lxk9erT59ttvzbfffmtGjx5t7rnnni6t25mxQi2Q87zqqqvMrbfeGqLKO68zc12+fLl5+eWXzZw5c8zUqVNbLL/yyivN2WefbUpLS01+fr7p27evee6554I8A/8Fcq6nnXaaefjhh4NbcBf5O8+qqiozd+5cs23bNtPY2Gg++ugjk56ebt58803fOuF+TLuDvGG/vGFM9GQO8ob/65I3wgN5w355wxgyhz/IG+QNY8gb5A3rRUvmIG+QN6I1bwAIPpq/AbBo0SIzfPjwZs8NGzbMPP300369fsOGDcbpdPpC8q9//WszefJk3/K6ujqTnp5u3n333cAV3UXdmWsk/XIU6Hna6Zh2tH64/XKUm5tr/va3v/n+vWzZMpOXl9eldTszVqgFcp7h/IuRMV07DvPmzWvxC8PBgwdNXFycWbt2re+5Bx54wJx66qkBrbc7AjVXY8L7l6Pu/GxdcMEFZu7cucaYyDim3UHesF/eMCZ6Mgd5w/91yRvhgbxhv7xhDJnDH+QN8saRyBvkDStES+Ygb5A3jhQteQNA8HHb5wD4/PPPNWbMmGbPjRkzRp9//rlfr//Xv/6lY445Ri6Xq9XxYmNjNWzYML/HC6buzrU1VVVVOuqoo5Sbm6vLLrtMe/fu7WaV3RfoedrpmPq7/ogRI5STk6MpU6ZYdvun0tJS7dmzp1m9Y8aM0a5du1ReXt6pdTszVqgFcp5eixcvVmZmpoYPH64HH3xQjY2NwZ6GXwJ5HPLz81VXV9dirHD4uZQCO1ev3/72t8rMzNTYsWO1ePHiAFXaPd2ZZ01NjT755BONGjVKUvgf0+4ib9gvb0jRkznIG/6tS94gb4RatOQNiczhL/IGeaMr44XLcSVv+LduOOcNKXoyB3mDvHGkaMobAIKP5m8HzjvvPDkcjja/du7cqaqqKqWnpzd7XXp6uiorKzscf926dZo7d64efvhh33PdGa87gj3X1gwdOlTr16/X7t279emnn8oYo/PPPz+oIcyKedrpmHa0fnZ2ttasWaMdO3Zo8+bNGjRokM4880xVVFQEY4rtqqqq8tXXtFZJLebX0bqdGSvUAjlPSbrllluUn5+vAwcO6KmnntIjjzyiRx55JDjFd1Igj0NVVZWSkpJ8b0x5x7L6eHoF+ntu/vz5+vrrr7Vv3z7dd999uvnmm/Xyyy8HotRu6eo8jTG69tprNWjQIE2fPt03Vjgf0/aQN+yXN6ToyRzkDfIGeaP9scL53BQteUMic0jkDfJGerPXkTfIG+GYN6ToyRzkjcOPyRuH2SlvAAgPro5XiW5Lly5VXV1dm8szMzOVnJyskpKSZs+Xl5erR48e7Y69ceNGTZo0SY899pjOPPNM3/PJyckt/hKovLxcKSkpXZiB/4I517bk5OQoJyfH93jBggVKS0vTli1bNHTo0C6N2REr5mmnY9rR+snJyRo/frykw4HkD3/4g5YsWaIPP/xQ55xzTnem02nJycm++rKzs32PJbXY9x2t6/2F3Z+xQi2Q85SkY4891rf+CSecoNtuu02LFy/WT3/60yDOwj+dmas/Yx06dEgej8cXpEPxc+mvQM5Vkk488UTf47PPPlvXX3+9XnzxRV1wwQUBqLbrujJPY4xmz56t/Px8vfPOO3I6nb6xwvmYtoe88R275A0pejIHeYO8Qd5of6xwPjdFS96QyBwSeaMp8gZ5QyJvhGPekKInc5A3yBtedssbAMIDV/52IDU1VdnZ2W1+OZ1OjRo1SuvXr2/2uvXr12vkyJFtjrtx40adccYZmj9/vi6//PJmy44cr76+Xl9++WW74wVCsObaGQ6HIyDjtMeKedrpmHZ2fe9f4VohIyNDubm5zepdv369+vbtq7S0tE6t25mxQi2Q82yNN3SGg0AehyFDhig2NlYbNmxoNlawfy79FezvuXA5rp2dpzFGc+bM0Zo1a/TWW281Wyfcj2l7yBv2yxtS9GQO8gZ5g7zRtnA/N0VL3pDIHBJ5g7yxvtnryBvkjXDMG1L0ZA7yBnlDsmfeABAmrPmoYXspLi426enpZtGiRaa2ttYsWrTIZGZmmpKSklbX37Rpk+nZs6dZsGBBq8s3b95sEhMTzcqVK01NTY2ZN2+eGTRokKmvrw/mNPzS2bkaY0x1dbWprq42ksyaNWtMdXW1aWhoMMYY8+6775rt27ebxsZGU1RUZK644gozcuRI4/F4QjWlVgV6nnY6ph2t//HHH5svv/zSeDweU1lZaX75y1+a3r17m7KyslBOy2fu3Llm7NixpqCgwBQUFJixY8eae+65p0vrdmasUAvkPF988UVTXl5uGhsbzdq1a02/fv3MAw88EKqpdKgzc62vrzfV1dXmzjvvNOeff76prq42tbW1vuVXXHGFmTRpkikrKzNbtmwxeXl55rnnngvVVDoUqLmWlpaalStXmoMHDxqPx2Peeecdk5aWZpYtWxbK6bSpM/OcPXu2GTVqlCkqKmp1ebgf0+4gb9gvbxgTPZmDvOH/uuSN8EDesF/eMIbM4Q/yBnnDi7xB3rBStGQO8gZ5I1rzBoDgo/kbIKtWrTIjR4408fHxZtSoUWb16tW+Zd98841JSkoy33zzjTHGmFmzZhmHw2GSkpKafXmXG2PMSy+9ZAYOHGji4+PNSSedZL766quQz6ktnZmrMcZIavH13nvvGWOMefDBB01ubq5JTEw0OTk5ZsaMGc1ea6VAztMYex3T9tZfunSpGTBggElMTDTZ2dlm8uTJZuPGjSGdT1N1dXVm9uzZJj093aSnp5ubbrrJ9wvp9ddfb66//nq/1vVnuZUCOc9TTjnFpKWlmaSkJDN48GBz//33+37JDwedmeu8efNa/FyedtppvuXl5eXm0ksvNcnJyaZHjx5h9cuuMYGb6/79+8348eNNSkqKSUlJMSNHjjRPPfWUFVNqlb/z3Llzp5Fk3G53s/Nn0/0Q7se0u8gbh9kpbxgTPZmDvNHxuv4stxJ5g7wRyXnDGDKHv8gbh5E3yBvkDetES+Ygb5A3ojlvAAguhzHGBOACYgAAAAAAAAAAAACAhcLnhvgAAAAAAAAAAAAAgC6j+QsAAAAAAAAAAAAANkDzFwAAAAAAAAAAAABsgOYvAAAAAAAAAAAAANgAzV8AAAAAAAAAAAAAsAGavwAAAAAAAAAAAABgAzR/AQAAAAAAAAAAAMAGaP4CAAAAAAAAAAAAgA3Q/AUAAAAAAAAAAAAAG6D5CwAAAAAAAAAAAAA2QPMXAAAAAAAAAAAAAGyA5i8AAAAAAAAAAAAA2ADNXwAAAAAAAAAAAACwAZq/AAAAAAAAAAAAAGADNH8BAAAAAAAAAAAAwAZo/gIAAAAAAAAAAACADdD8BQAAAAAAAAAAAAAboPkLAAAAAAAAAAAAADZA8xcAAAAAAAAAAAAAbIDmLwAAAAAAAAAAAADYAM1fAAAAAAAAAAAAALABmr8AAAAAAAAAAAAAYAM0fwEAAAAAAAAAAADABmj+AgAAAAAAAAAAAIAN0PwFAAAAAAAAAAAAABug+QsAAAAAAAAAAAAANkDzFwAAAAAAAAAAAABsgOYvAAAAAAAAAAAAANgAzV8AAAAAAAAAAAAAsAGavwAAAIBNZWdny+Fw+L7aeq7p4450Zt1gObIGK2sKxrb9HTMcjgUAAAAAAAgvNH8BAACAEDuyAXtkMzZQiouLJUlOp1O9e/du87msrCzdcccdfo05adIkDRgwIKB1drb5LEnr1q2zrIam5s+f36yuQLj22mt9x6Y7tbWmZ8+eLb7nYmJimq3T0ffmkCFD2v2+7Uy9/vws0OQGAAAAAMB/DmOMsboIAAAAIFq43W7V1dX5/u10OtXY2Oj7dyDjubdh1nTM1p6zWmdqCtacujNGsPdpW+N3druFhYXNmspNv/emT5+u5cuXNxs3KyvLt+7QoUP1wQcfNFve9PXeGrzLXn/9dZ1zzjnt1uPvz0I4fs8CAAAAABCuuPIXAAAACCFvs2v69OkyxqihoUHGGBljNHr0aN96iYmJLa6E7NmzZ7OxWrta8oUXXvAta229I5878rEkpaSktHklZntXera2Hbfb3e6VnR3V2VRubm67+7a9q1GTkpJaLHe73R3W0FrdTzzxRKvbv+6669qt64033mhznm3tY3/2j79Xj/fp08f32Pu9572C+qWXXmqxflFRke/L2/htqqGhodn2mu6Xjhq/kv8/CwAAAAAAwH80fwEAAIAQGTJkiO+x9yrLptavX+/7b3V1dYvlBw4c8D1u2nSLjY31PZ4xY4ak5o2+rKwsjRs3rtXnjnTuueeqqqrK92+Xy9XunJrWERcX1+rzTTmd3/0K8sILL/hum+ytKSsrS5MmTWr1tXv37pXU/i2fm47vffzEE0/o0KFDLWqrq6vTfffd12x73hqa1nXk3ObMmdNs2cyZMyVJixYtarUm735ua17tufbaa1vU1vQ5r6bHqen3WVPeq2qbrjtmzJg2t920oXzZZZe1WP7CCy80uxrXu1/8uULX358FAAAAAADQOdz2GQAAAAiRpg3R9mK4d73ExEQdPHhQ0uFGpvc1xhi/bn/s7y2Smz7nfTx48GDl5+e3WZu/22h6a1/vOt65OBwONTY2+n1b387c/ri1ORUUFCgnJ0eSdN999+n2229vsU4ga2hredPvg0mTJumDDz5QZWWlnE5ns6tpO7rt8ZHPX3jhhb4reFur44knnvA1aEeMGKHLLrvMtw9a296RvMvHjx+vtWvXNqvDeywHDBig7du3+5aNGzdOn3zySZv7o61a25snAAAAAABoG81fAAAAIEQ62/xtr9nX3u19Z86cqSVLlnSr+etPE7Nps7E1bTV/R44cqU2bNvm9zfZq78yc2quzo7Fbe40/tR25vKCgQL1795bD4WixblvN3q7Ou706/JnTka9xuVyqr6/3a9zHH3+83SuBaf4CAAAAABAc3PYZAAAACJHBgwcHfEyXy9Xi63//938Dvp3WrFq1qs06mt4m+UgxMTGhKK+F1vZVa7e+9jqyodn0ltJd1bt3b0nf3YI51Lyfqeu9Dbg/DdXp06dLkjweT6vLW2vOzp49u90xg/GzAAAAAAAAaP4CAAAAIdP0NsoXXnhhi+VHfv5qUlKS73Fbjcf6+voWX+19jqu/2vrc2Kb279/fZh21tbXdrqEtI0eO7NLrvv/977eos7VbEh/J2zBtaGhodbn3M5fb+3zkESNGdKnmYPnkk0+0Z88evxrx7V3dnZubK0lKSUnp1PY7+7MAAAAAAAD8Q/MXAAAACCFvo/Cll16Sw+FQTEyMnE6nHA6HNmzYIOnwrYEl6dChQ77PU236eb9NeZd7x+joFscdmT9/viRpy5YtcjgciouL82tc71y863alDofDoX79+umyyy5rdfnrr78uSb5bRrdXg/ff0uHbYEvS6tWr5XA45Ha7fTUe2bSMiYlRv3799MQTTzQbNzY2ts05ea+IbeuWyJK0cePGFs81vfrVn6uKvfvnzjvv7HDd9sZoely9VyB7v6+882y6n7xau0p47969kqSKiopmz/szH39+FgAAAAAAQOfQ/AUAAABCqK6uThkZGb5/NzY2tmiq5eTkKCEhocVre/To4Xvc1ufFdtdtt90mt9vt+3d7Dc0jt9vVWxk3HWPXrl1aunRpq+udc845HY7VtAbv4yVLligrK8v3vPcziCVp9OjRkqQ+ffr4XrNr1y7NmTOnWV1t3fK4K7wN1aZXv7Z1VbEk3XHHHb7Hu3bt0u9+97tu19D0uHob/pKa3a676X6aMGFCizG881i3bp3vOW+j3bvvvLe5bo0/PwsAAAAAAKBzHIbfrgEAAABEiOzsbBUXF8vtdqumpsbqcnwN0HHjxvl1C2kAAAAAAIBgovkLAAAAIKJ4G67h8KtMONUCAAAAAABA8xcAAAAAAAAAAAAAbIDP/AUAAAAAAAAAAAAAG6D5CwAAAAAAAAAAAAA2QPMXAAAAAAAAAAAAAGyA5i8AAAAAAAAAAAAA2ADNXwAAAAAAAAAAAACwAZq/AAAAAAAAAAAAAGADNH8BAAAAAAAAAAAAwAZo/gIAAAAAAAAAAACADdD8BQAAAAAAAAAAAAAb+H9QbKgKcilh7gAAAABJRU5ErkJggg==
#     # Positions for each category within the group
#     dps = outcome_data["Variable"].unique()
#     positions = np.arange(len(dps))

#     # Offset for each group (DPs)
#     offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

#     # Set the colors
#     colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']

#     # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}
#     # Plotting the point estimates and confidence intervals
#     for j, cat in enumerate(Categories_values):
        
#         cat_data = outcome_data[outcome_data["Model"] == cat]
        
#         # is_sig = cat_data["ifsig"]
#         y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]

#         p_values = []
        
#         for idx, row in cat_data.iterrows():
#             p_value = row.get("p_adj", 0.05)
#             p_values.append(p_value)

#         sig_threshold = 0.05
#         is_significant = [p < sig_threshold for p in p_values]

#         for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
#             y_positions,
#             cat_data["CI_lower"],
#             cat_data["CI_upper"],
#             cat_data["Coefficient"],
#             is_significant
#         )):
#             ax.errorbar(x = coef,
#                         y = y_pos,
#                         xerr = [[coef - ci_lower], [ci_upper - coef]],
#                         fmt = "none",
#                         capsize = 4,
#                         color = colors[j % len(colors)],
#                         label = cat if n == 0 else None)
#             # plot point
#             if is_sig:
#                 # solid for significant results
#                 ax.scatter(coef, y_pos, s = 30,
#                            c = colors[j % len(colors)],
#                            edgecolors = colors[j % len(colors)], # edge color
#                            linewidths = 1,
#                            zorder = 3)
#             else:
#                 ax.scatter(coef, y_pos, s = 30, facecolors = "none",
#                            edgecolors = colors[j % len(colors)], # edge color
#                            linewidths = 1,
#                            zorder = 3)

#           # Customize the plot
#           ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
#           ax.set_yticks(positions)
#           ax.set_yticklabels(dps)
#     # ax.set_ylabel("DPs")
#     # ax.set_xlabel('Coefficient')
#          ax.set_title(f'{outcome}')
#          ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
#     # ax.legend()
#          ax.set_xlim(-0.2, 0.2) # set x-axis limits

#          plt.tight_layout(rect = [0.03, 0.03, 0.95, 0.95])
#          fig.supxlabel("Coefficient (beta) with Confidence Interval", fontsize = 12)
#          fig.supylabel("Dietary pattern Scores", fontsize = 12)

# Single shared legend from the first subplot
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc = "upper center", bbox_to_anchor = (0.5, 1.02),
#           ncol = len(Categories_values), fontsize = 12, frameon = True)

# Update the shared legend
# from matplotlib.lines import Line2D

# # Category color legend
# color_handles = [Line2D([0], [0], marker = "o", color = "w", markerfacecolor = colors[j % len(colors)],
#                        markeredgecolor = colors[j % len(colors)], markersize = 8, label = cat)
#                 for j, cat in enumerate(Categories_values)]

# # Solid or hollow
# sig_handles = [Line2D([0], [0], marker = "o", color = "w", markerfacecolor = "grey",
#                      markeredgecolor = "grey", markersize = 8, label = "P_adj < 0.05"),
#                Line2D([0], [0], marker = "o", color = "w", markerfacecolor = "white",
#                      markeredgecolor = "grey", markersize = 8, label = "P_adj >= 0.05")]

# all_handles = color_handles + sig_handles

# fig.legend(handles= all_handles, loc = "upper center", bbox_to_anchor = (0.5, 1.02),
#           ncol = len(Categories_values) + 2, fontsize = 12, frameon = True)


# plt.savefig("./Results/Continues_DPs_to_Liver_four_models_v2.pdf", bbox_inches='tight')
# plt.show()


# In[189]:


# Continue DPs and outcome- sensitivity analysis
results_continue_model1_sen_corrected.loc[:, "Model"] = "Model1"
results_continue_model2_sen_corrected.loc[:, "Model"] = "Model2"
results_continue_model3_sen_corrected.loc[:, "Model"] = "Model3"
results_continue_model4_sen_corrected.loc[:, "Model"] = "Model4"
results_continue_model4_sen_corrected


# In[190]:


DPs_to_Liver_Res_sen = pd.concat([results_continue_model1_sen_corrected, results_continue_model2_sen_corrected,
          results_continue_model3_sen_corrected, results_continue_model4_sen_corrected], ignore_index=True)
DPs_to_Liver_Res_sen.shape

Outcomes_map2 = {
    "elasticity_median_median_of_qboxes_clipped_rank_INT": "Liver Elasticity",
    "viscosity_median_median_of_qboxes_clipped_rank_INT": "Liver Viscosity",
    "dispersion_median_median_of_qboxes_clipped_rank_INT": "Liver Dispersion",
    "attenuation_coefficient_qbox_clipped_rank_INT": "Attenuation coefficient",
    "velocity_median_median_of_qboxes_clipped_rank_INT": "Liver Velocity",
    "speed_of_sound_qbox_clipped_rank_INT": "Speed of Sound"
}

DPs_to_Liver_Res_sen["Outcome"] = DPs_to_Liver_Res_sen["Outcome"].map(Outcomes_map2)
DPs_to_Liver_Res_sen["Variable"] = DPs_to_Liver_Res_sen["Variable"].map(DPs_names)
DPs_to_Liver_Res_sen


# In[218]:


results_continue_model1_sen2_corrected.loc[:, "Model"] = "Model1"
results_continue_model2_sen2_corrected.loc[:, "Model"] = "Model2"
results_continue_model3_sen2_corrected.loc[:, "Model"] = "Model3"
results_continue_model4_sen2_corrected.loc[:, "Model"] = "Model4"
results_continue_model4_sen2_corrected


# In[219]:


DPs_to_Liver_Res_sen2 = pd.concat([results_continue_model1_sen2_corrected, results_continue_model2_sen2_corrected,
          results_continue_model3_sen2_corrected, results_continue_model4_sen2_corrected], ignore_index=True)

DPs_to_Liver_Res_sen2["Outcome"] = DPs_to_Liver_Res_sen2["Outcome"].map(Outcome_name)
DPs_to_Liver_Res_sen2["Variable"] = DPs_to_Liver_Res_sen2["Variable"].map(DPs_names)


# ## Combine two sensitivtiy analysis plot

# In[273]:


# DPs_to_Liver_Res_sen, DPs_to_Liver_Res_sen2

# colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']

XLIM = (-0.2, 0.2)
MARKER_SIZE = 30
CAPSIZE = 4

MODEL_ORDER = ["Model1", "Model2", "Model3", "Model4"]

MODEL_COLOR_MAP = {
        "Model1" : "#E64B35",
        "Model2" : "#4DBBD5",
        "Model3" : "#00A087",
        "Model4" : "#3C5488",
    }

# helper function: plot one 2*3 block of forest plots
def plot_forest(df, axes_flat, 
                outcome_col = "Outcome",
                group_col = "Model",
                group_order = MODEL_ORDER,
                group_color_map = MODEL_COLOR_MAP):

    outcomes = df[outcome_col].unique()
    # model_values = df["Model"].unique()
    
    group_values = [g for g in group_order if g in df[group_col].values]
    
    for i, outcome in enumerate(outcomes):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        outcome_data = df[df[outcome_col] == outcome]

        dps = outcome_data["Variable"].unique()
        
        positions = np.arange(len(dps))
        offset = np.linspace(0.3, -0.3, len(group_values))

        for j, grp in enumerate(group_values):
            grp_data = outcome_data[outcome_data[group_col] == grp]

            color = group_color_map[grp]
            
            y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in grp_data["Variable"]]

            p_vals = grp_data["p_adj"].values if "p_adj" in grp_data.columns else np.full(len(grp_data), np.nan)
            is_sig = np.where(np.isnan(p_vals), False, p_vals < 0.05)

            for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
                y_positions,
                grp_data["CI_lower"],
                grp_data["CI_upper"],
                grp_data["Coefficient"],
                is_sig
            )):
                ax.errorbar(x = coef,
                            y = y_pos,
                            xerr = [[coef - ci_lower],[ci_upper - coef]],
                            fmt = "none",
                            capsize = CAPSIZE,
                            # color = colors[j % len(colors)],
                            color = color,
                            label = grp if n == 0 else None
                           )
                if is_sig:
                    ax.scatter(coef, y_pos, s = MARKER_SIZE,
                              # c = colors[j % len(colors)],
                              # edgecolors = colors[j % len(colors)],
                              c = color,
                              edgecolors = color,
                              linewidths = 1, zorder = 3)
                else:
                    ax.scatter(coef, y_pos, s = MARKER_SIZE,
                              facecolors = "none",
                              edgecolors = color,
                              # edgecolors = colors[j % len(colors)],
                              linewidths = 1, zorder = 3)

            # Panel format
            ax.axvline(x = 0, color = "black", linestyle = "--", alpha = 0.5)
            ax.set_yticks(positions)
            ax.set_yticklabels(dps)
            ax.set_title(f'{outcome}', fontsize = 12)
            ax.tick_params(axis = "both", labelsize = 10)
            ax.invert_yaxis()
            ax.set_xlim(XLIM)

            # share ylabel
            if i % 3 == 0:
                ax.set_ylabel("Dietary pattern scores", fontsize = 12)
                ax.set_yticklabels(dps)
            else:
                ax.set_ylabel("", fontsize = 12)
                ax.set_yticklabels([])

            if i >= 3:
                ax.set_xlabel("Coefficient (beta) with 95% CI", fontsize = 12)
            else:
                ax.set_xlabel("")
                
        # Hide unused axes
        for k in range(len(outcomes), len(axes_flat)):
            axes_flat[k].set_visible(False)
           


# In[244]:


fig = plt.figure(figsize = (16, 20))

# GridSpec: 4 rows, 3 cols, with gap between row 2 and 3
gs = gridspec.GridSpec(
    4, 3, figure = fig,
    height_ratios = [1, 1, 1, 1],
    hspace = 0.35, # vertical spacing between all rows
    wspace = 0.15,
    left = 0.08
)

# sensitivity 1
print(DPs_to_Liver_Res_sen["Model"].unique())

MODEL_ORDER = ["Model1", "Model2", "Model3", "Model4"]

axes_sen1 = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

# bottom sens2
axes_sen2 = [fig.add_subplot(gs[r, c]) for r in range(2, 4) for c in range(3)]

# Draw both blocks
DPs_to_Liver_Res_sen["Model"] = pd.Categorical(
    DPs_to_Liver_Res_sen["Model"],
    categories = MODEL_ORDER,
    ordered = True
)

plot_forest(DPs_to_Liver_Res_sen, 
            axes_sen1)
plot_forest(DPs_to_Liver_Res_sen2, 
            axes_sen2)

# DPs_to_Liver_Res_sen2.head()

# Row labels in the left margin
fig.text(0.01, 0.9, "A", fontsize = 16, fontweight = "bold")
fig.text(0.01, 0.5, "B", fontsize = 16, fontweight = "bold")

# share legend
handles, labels = axes_sen1[0].get_legend_handles_labels()
for ax in axes_sen1:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

# Category color legend
color_handles = [Line2D([0], [0], marker = "o", color = "w", 
                        markerfacecolor = MODEL_COLOR_MAP[cat],
                        markeredgecolor = MODEL_COLOR_MAP[cat], 
                        markersize = 8, label = cat)
                for cat in MODEL_ORDER]

fig.legend(color_handles, MODEL_ORDER,
          loc = "upper center", ncol = 4, frameon = True, fontsize = 12, 
          bbox_to_anchor = (0.5, 0.92))

# save the figure
fig.savefig("./Results/Combined_DPs_to_Liver_Sensitivity.pdf", bbox_inches = "tight", dpi = 300)
plt.show()


# In[377]:


outcomes = DPs_to_Liver_Res_sen["Outcome"].unique()
Categories_values = DPs_to_Liver_Res_sen["Model"].unique() # Model1-4
dps_values = DPs_to_Liver_Res_sen["Variable"].unique() # DPs

# Plot for the continuous MODEL2
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = DPs_to_Liver_Res_sen[DPs_to_Liver_Res_sen["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']

    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):     
        cat_data = outcome_data[outcome_data["Model"] == cat]
        # is_sig = cat_data["ifsig"]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]

        p_values = []
        for idx, row in cat_data.iterrows():
            p_value = row.get("p_adj", 0.05)
            p_values.append(p_value)

        sig_threshold = 0.05
        is_significant = [p < sig_threshold for p in p_values]

        for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
            y_positions,
            cat_data["CI_lower"],
            cat_data["CI_upper"],
            cat_data["Coefficient"],
            is_significant
        )):
            ax.errorbar(x = coef,
                        y = y_pos,
                        xerr = [[coef - ci_lower], [ci_upper - coef]],
                        fmt = "none",
                        capsize = 4,
                        color = colors[j % len(colors)],
                        label = cat if n == 0 else None)
            # plot point
            if is_sig:
                # solid for significant results
                ax.scatter(coef, y_pos, 
                           s = 30,
                           c = colors[j % len(colors)],
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)
            else:
                ax.scatter(coef, y_pos, 
                           s = 30, facecolors = "none",
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.5, 0.5) # set x-axis limits

plt.tight_layout()
plt.savefig("./Results/Continues_DPs_to_Liver_four_models_sensitivtiy1.pdf", bbox_inches='tight')
plt.show()


# In[379]:


DPs_to_Liver_Res_sen2 = pd.concat([results_continue_model1_sen2_corrected, results_continue_model2_sen2_corrected,
          results_continue_model3_sen2_corrected, results_continue_model4_sen2_corrected], ignore_index=True)

DPs_to_Liver_Res_sen2["Outcome"] = DPs_to_Liver_Res_sen2["Outcome"].map(Outcome_name)
DPs_to_Liver_Res_sen2["Variable"] = DPs_to_Liver_Res_sen2["Variable"].map(DPs_names)

outcomes = DPs_to_Liver_Res_sen2["Outcome"].unique()
Categories_values = DPs_to_Liver_Res_sen2["Model"].unique() # Model1-4
dps_values = DPs_to_Liver_Res_sen2["Variable"].unique() # DPs

# Plot for the continuous MODEL2
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = DPs_to_Liver_Res_sen2[DPs_to_Liver_Res_sen2["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']

    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):     
        cat_data = outcome_data[outcome_data["Model"] == cat]
        # is_sig = cat_data["ifsig"]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]

        p_values = []
        for idx, row in cat_data.iterrows():
            p_value = row.get("p_adj", 0.05)
            p_values.append(p_value)

        sig_threshold = 0.05
        is_significant = [p < sig_threshold for p in p_values]

        for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
            y_positions,
            cat_data["CI_lower"],
            cat_data["CI_upper"],
            cat_data["Coefficient"],
            is_significant
        )):
            ax.errorbar(x = coef,
                        y = y_pos,
                        xerr = [[coef - ci_lower], [ci_upper - coef]],
                        fmt = "none",
                        capsize = 4,
                        color = colors[j % len(colors)],
                        label = cat if n == 0 else None)
            # plot point
            if is_sig:
                # solid for significant results
                ax.scatter(coef, y_pos, s = 30,
                           c = colors[j % len(colors)],
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)
            else:
                ax.scatter(coef, y_pos, s = 30, facecolors = "none",
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.5, 0.5) # set x-axis limits

plt.tight_layout()
plt.savefig("./Results/Continues_DPs_to_Liver_four_models_sensitivtiy2.pdf", bbox_inches='tight')
plt.show()


# In[380]:


# results_df_model2_clean = results_df_model2.loc[~results_df_model2["P_val"].isnull(),:]
# results_df_model2_clean.shape

# results_df_model2_clean["P_adj"] = multipletests(
#         results_df_model2_clean["P_val"],
#         alpha=0.05, 
#         method='fdr_bh'
# )[1]

# results_df_model2_clean.loc[results_df_model2_clean["P_adj"] < 0.05,:]


# # Analysis For male and Female separately
# - Perform for the continous DPs

# In[245]:


print(merge_df_MAFLD_filter_3["sex_x"].value_counts(dropna = False))
df_male = merge_df_MAFLD_filter.loc[merge_df_MAFLD_filter_3["sex_x"] == 1]
df_male.shape

df_female = merge_df_MAFLD_filter.loc[merge_df_MAFLD_filter_3["sex_x"] == 0]
df_female.shape


# In[246]:


model2 = ['age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']

# for hPDI index adjusted for alcohol intake 
model2_2 = ['age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use',"alcohol_g"]

model3 = ['age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "bmi"]

model3_2 = ['age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "alcohol_g", 'bmi']

model4 = ['age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "NSAID_use", "CVD_f_history", "T2D_f_history"]

model4_2 = ['age', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "NSAID_use", "alcohol_g", "CVD_f_history", "T2D_f_history"]


# In[247]:


# Male
# Model1
Male_results_continue_model1 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    lm_fit = run_regression_formula(df = df_male, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    Male_results_continue_model1 = pd.concat([Male_results_continue_model1, lm_fit], ignore_index=True)

# Model2
Male_results_continue_model2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model2)
        Male_results_continue_model2 = pd.concat([Male_results_continue_model2, lm_fit], ignore_index=True)
    
# Model3
Male_results_continue_model3 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model3)
        Male_results_continue_model3 = pd.concat([Male_results_continue_model3, lm_fit], ignore_index=True)

# Model4
Male_results_continue_model4 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = df_male, predictors = predict_var, outcomes = livers, control_vars = model4)
        Male_results_continue_model4 = pd.concat([Male_results_continue_model4, lm_fit], ignore_index=True)

# Adjust for pvalue
Male_df_models = [Male_results_continue_model1, Male_results_continue_model2, 
             Male_results_continue_model3, Male_results_continue_model4]
Male_corrected_dfs = []
Male_significant_dfs = []

for i, df in enumerate(Male_df_models):
    df_corrected, df_sig = fdr_correction_(df)
    Male_corrected_dfs.append(df_corrected)
    Male_significant_dfs.append(df_sig)

# Access to results
Male_results_continue_model1_corrected, Male_results_continue_model2_corrected, Male_results_continue_model3_corrected, Male_results_continue_model4_corrected = Male_corrected_dfs
Male_results_continue_model1_sig, Male_results_continue_model2_sig, Male_results_continue_model3_sig, Male_results_continue_model4_sig = Male_significant_dfs


# In[248]:


Male_results_continue_model1_corrected.loc[:, "Model"] = "Model1"
Male_results_continue_model2_corrected.loc[:, "Model"] = "Model2"
Male_results_continue_model3_corrected.loc[:, "Model"] = "Model3"
Male_results_continue_model4_corrected.loc[:, "Model"] = "Model4"

Male_results_cont_combine = pd.concat([Male_results_continue_model1_corrected,
                                 Male_results_continue_model2_corrected,
                                 Male_results_continue_model3_corrected,
                                 Male_results_continue_model4_corrected], ignore_index=True
                                )
Male_results_cont_combine.shape


# In[249]:


Male_results_cont_combine.head()
Male_results_cont_combine["Variable"] = Male_results_cont_combine["Variable"].map(DPs_names)
Male_results_cont_combine["Outcome"] = Male_results_cont_combine["Outcome"].map(Outcome_name)


# In[250]:


Male_results_cont_combine.head()


# In[251]:


Male_results_cont_combine = Male_results_cont_combine.loc[Male_results_cont_combine["Model"] == "Model2"]
Male_results_cont_combine.shape


# In[388]:


# # Plot for the continuous MODEL2
# Categories_values = Male_results_cont_combine["Model"].unique() # Model1-4
# # dps_values = Male_results_cont_combine["Variable"].unique() # DPs

# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()

# for i, outcome in enumerate(outcomes):
#     ax = axes[i]
#     outcome_data = Male_results_cont_combine[Male_results_cont_combine["Outcome"] == outcome] # Corrected line

#     # Positions for each category within the group
#     dps = outcome_data["Variable"].unique()
#     positions = np.arange(len(dps))

#     # Offset for each group (DPs)
#     offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

#     # Set the colors
#     colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
#     # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

#     # Plotting the point estimates and confidence intervals
#     for j, cat in enumerate(Categories_values):
#         cat_data = outcome_data[outcome_data["Model"] == cat]
#         y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]

#         p_values = []
        
#         for idx, row in cat_data.iterrows():
#             p_value = row.get("p_adj", 0.05)
#             p_values.append(p_value)

#         sig_threshold = 0.05
#         is_significant = [p < sig_threshold for p in p_values]

#         for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
#             y_positions,
#             cat_data["CI_lower"],
#             cat_data["CI_upper"],
#             cat_data["Coefficient"],
#             is_significant
#         )):
#             ax.errorbar(x = coef,
#                         y = y_pos,
#                         xerr = [[coef - ci_lower], [ci_upper - coef]],
#                         fmt = "none",
#                         capsize = 4,
#                         color = colors[j % len(colors)],
#                         label = cat if n == 0 else None)
#             # plot point
#             if is_sig:
#                 # solid for significant results
#                 ax.scatter(coef, y_pos, s = 30,
#                            c = colors[j % len(colors)],
#                            edgecolors = colors[j % len(colors)], # edge color
#                            linewidths = 1,
#                            zorder = 3)
#             else:
#                 ax.scatter(coef, y_pos, s = 30, facecolors = "none",
#                            edgecolors = colors[j % len(colors)], # edge color
#                            linewidths = 1,
#                            zorder = 3)

#         # ax.errorbar(x=cat_data['Coefficient'], 
#         #             y=y_positions,
#         #             xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
#         #             fmt='o', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

#     # Customize the plot
#     ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
#     ax.set_yticks(positions)
#     ax.set_yticklabels(dps)
#     ax.set_ylabel("DPs")
#     ax.set_xlabel('Coefficient')
#     ax.set_title(f'Forest Plot for {outcome}')
#     ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
#     ax.legend()
#     ax.set_xlim(-0.5, 0.5) # set x-axis limits

# plt.tight_layout()
# # plt.savefig("./Results/Male_Continues_DPs_to_Liver_four_models_v2.pdf", bbox_inches='tight')
# plt.show()


# ## Female

# In[252]:


# Female
# Model1
Female_results_continue_model1 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    lm_fit = run_regression_formula(df = df_female, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    Female_results_continue_model1 = pd.concat([Female_results_continue_model1, lm_fit], ignore_index=True)

# Model2
Female_results_continue_model2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model2)
        Female_results_continue_model2 = pd.concat([Female_results_continue_model2, lm_fit], ignore_index=True)
    
# Model3
Female_results_continue_model3 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model3)
        Female_results_continue_model3 = pd.concat([Female_results_continue_model3, lm_fit], ignore_index=True)

# Model4
Female_results_continue_model4 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = df_female, predictors = predict_var, outcomes = livers, control_vars = model4)
        Female_results_continue_model4 = pd.concat([Female_results_continue_model4, lm_fit], ignore_index=True)

# Adjust for pvalue
Female_df_models = [Female_results_continue_model1, Female_results_continue_model2, 
             Female_results_continue_model3, Female_results_continue_model4]
Female_corrected_dfs = []
Female_significant_dfs = []

for i, df in enumerate(Female_df_models):
    df_corrected, df_sig = fdr_correction_(df)
    Female_corrected_dfs.append(df_corrected)
    Female_significant_dfs.append(df_sig)

# Access to results
Female_results_continue_model1_corrected, Female_results_continue_model2_corrected, Female_results_continue_model3_corrected, Female_results_continue_model4_corrected = Female_corrected_dfs
Female_results_continue_model1_sig, Female_results_continue_model2_sig, Female_results_continue_model3_sig, Female_results_continue_model4_sig = Female_significant_dfs

Female_results_continue_model1_corrected.loc[:, "Model"] = "Model1"
Female_results_continue_model2_corrected.loc[:, "Model"] = "Model2"
Female_results_continue_model3_corrected.loc[:, "Model"] = "Model3"
Female_results_continue_model4_corrected.loc[:, "Model"] = "Model4"

Female_results_cont_combine = pd.concat([Female_results_continue_model1_corrected,
                                 Female_results_continue_model2_corrected,
                                 Female_results_continue_model3_corrected,
                                 Female_results_continue_model4_corrected], ignore_index=True
                                )
Female_results_cont_combine.shape


# In[253]:


Female_results_cont_combine["Variable"] = Female_results_cont_combine["Variable"].map(DPs_names)
Female_results_cont_combine["Outcome"] = Female_results_cont_combine["Outcome"].map(Outcome_name)


# In[254]:


Female_results_cont_combine.shape

# only select the model2
Female_results_cont_combine = Female_results_cont_combine.loc[Female_results_cont_combine["Model"] == "Model2"]
print(Female_results_cont_combine.shape)


# In[255]:


Female_results_cont_combine["Subgroup"] = "Female" 
Female_results_cont_combine

Male_results_cont_combine["Subgroup"] = "Male" 
Male_results_cont_combine

Sex_group_results = pd.concat([Female_results_cont_combine, Male_results_cont_combine])
Sex_group_results.head()


# ## Plots

# In[393]:


# Plot for the continuous MODEL2
# By merge Female and Male 

Categories_values = Sex_group_results["Subgroup"].unique() # Sex group
# dps_values = Male_results_cont_combine["Variable"].unique() # DPs

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = Sex_group_results[Sex_group_results["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.2, 0.2, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Subgroup"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]
        
        p_values = []
        
        for idx, row in cat_data.iterrows():
            p_value = row.get("p_adj", 0.05)
            p_values.append(p_value)

        sig_threshold = 0.05
        is_significant = [p < sig_threshold for p in p_values]

        for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
            y_positions,
            cat_data["CI_lower"],
            cat_data["CI_upper"],
            cat_data["Coefficient"],
            is_significant
        )):
            ax.errorbar(x = coef,
                        y = y_pos,
                        xerr = [[coef - ci_lower], [ci_upper - coef]],
                        fmt = "none",
                        capsize = 4,
                        color = colors[j % len(colors)],
                        label = cat if n == 0 else None)
            # plot point
            if is_sig:
                # solid for significant results
                ax.scatter(coef, y_pos, s = 30,
                           c = colors[j % len(colors)],
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)
            else:
                ax.scatter(coef, y_pos, s = 30, facecolors = "none",
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)


        
        # ax.errorbar(x=cat_data['Coefficient'], 
        #             y=y_positions,
        #             xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
        #             fmt='o', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.5, 0.5) # set x-axis limits

plt.tight_layout()
# plt.savefig("./Results/Female_Continues_DPs_to_Liver_four_models_v2.pdf", bbox_inches='tight')
plt.savefig("./Results/Subgroup_Continues_DPs_to_Liver_four_models_v2.pdf", bbox_inches='tight')
plt.show()


# ## Sensitivity analysis
# - clipped outcome

# In[394]:


df_male_sen = merge_df_MAFLD_filter_3.loc[merge_df_MAFLD_filter_3["sex_x"] == 1]
df_female_sen = merge_df_MAFLD_filter_3.loc[merge_df_MAFLD_filter_3["sex_x"] == 0]


# In[395]:


# Male
# Model1
Male_results_continue_model1_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    lm_fit = run_regression_formula(df = df_male_sen, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    Male_results_continue_model1_sen = pd.concat([Male_results_continue_model1_sen, lm_fit], ignore_index=True)

# Model2
Male_results_continue_model2_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_male_sen, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_male_sen, predictors = predict_var, outcomes = livers, control_vars = model2)
        Male_results_continue_model2_sen = pd.concat([Male_results_continue_model2_sen, lm_fit], ignore_index=True)
    
# Model3
Male_results_continue_model3_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_male_sen, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = df_male_sen, predictors = predict_var, outcomes = livers, control_vars = model3)
        Male_results_continue_model3_sen = pd.concat([Male_results_continue_model3_sen, lm_fit], ignore_index=True)

# Model4
Male_results_continue_model4_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_male_sen, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = df_male_sen, predictors = predict_var, outcomes = livers, control_vars = model4)
        Male_results_continue_model4_sen = pd.concat([Male_results_continue_model4_sen, lm_fit], ignore_index=True)

# Adjust for pvalue
Male_df_models_sen = [Male_results_continue_model1_sen, 
                      Male_results_continue_model2_sen, 
                      Male_results_continue_model3_sen, 
                      Male_results_continue_model4_sen]
Male_corrected_dfs_sen = []
Male_significant_dfs_sen = []

for i, df in enumerate(Male_df_models_sen):
    df_corrected_sen, df_sig_sen = fdr_correction_(df)
    Male_corrected_dfs_sen.append(df_corrected_sen)
    Male_significant_dfs_sen.append(df_sig_sen)


# In[396]:


# Access to results
Male_results_continue_model1_corrected_sen, Male_results_continue_model2_corrected_sen, Male_results_continue_model3_corrected_sen, Male_results_continue_model4_corrected_sen = Male_corrected_dfs_sen
# Male_results_continue_model1_sig_sen, Male_results_continue_model2_sig_sen, Male_results_continue_model3_sig_sen, Male_results_continue_model4_sig_sen = Male_significant_dfs_sen


Male_results_continue_model1_corrected_sen.loc[:, "Model"] = "Model1"
Male_results_continue_model2_corrected_sen.loc[:, "Model"] = "Model2"
Male_results_continue_model3_corrected_sen.loc[:, "Model"] = "Model3"
Male_results_continue_model4_corrected_sen.loc[:, "Model"] = "Model4"

Male_results_cont_sen_combine = pd.concat([Male_results_continue_model1_corrected_sen,
                                 Male_results_continue_model2_corrected_sen,
                                 Male_results_continue_model3_corrected_sen,
                                 Male_results_continue_model4_corrected_sen], ignore_index=True
                                )
Male_results_cont_sen_combine.shape


# ## Female sensitivity analysis

# In[397]:


# Model1 (female)
Female_results_continue_model1_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    lm_fit = run_regression_formula(df = df_female_sen, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    Female_results_continue_model1_sen = pd.concat([Female_results_continue_model1_sen, lm_fit], ignore_index=True)
Female_results_continue_model1_sen


# In[398]:


# Model2
Female_results_continue_model2_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_female_sen, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_female_sen, predictors = predict_var, outcomes = livers, control_vars = model2)
        Female_results_continue_model2_sen = pd.concat([Female_results_continue_model2_sen, lm_fit], ignore_index=True)
Female_results_continue_model2_sen


# In[399]:


# Model3
Female_results_continue_model3_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_female_sen, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = df_female_sen, predictors = predict_var, outcomes = livers, control_vars = model3)
        Female_results_continue_model3_sen = pd.concat([Female_results_continue_model3_sen, lm_fit], ignore_index=True)


# In[400]:


# Model4
Female_results_continue_model4_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_female_sen, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = df_female_sen, predictors = predict_var, outcomes = livers, control_vars = model4)
        Female_results_continue_model4_sen = pd.concat([Female_results_continue_model4_sen, lm_fit], ignore_index=True)


# In[401]:


# Adjust for pvalue
Female_df_models_sen = [Female_results_continue_model1_sen, Female_results_continue_model2_sen, 
             Female_results_continue_model3_sen, Female_results_continue_model4_sen]


# In[402]:


Female_corrected_dfs_sen = []
Female_significant_dfs_sen = []

for i, df in enumerate(Female_df_models_sen):
    df_corrected_female_sen, df_sig_female_sen = fdr_correction_(df)
    Female_corrected_dfs_sen.append(df_corrected_female_sen)
    Female_significant_dfs_sen.append(df_sig_female_sen)


# In[403]:


# Access to results
Female_results_continue_model1_corrected_sen, Female_results_continue_model2_corrected_sen, Female_results_continue_model3_corrected_sen, Female_results_continue_model4_corrected_sen = Female_corrected_dfs_sen
Female_results_continue_model1_sig_sen, Female_results_continue_model2_sig_sen, Female_results_continue_model3_sig_sen, Female_results_continue_model4_sig_sen = Female_significant_dfs_sen

Female_results_continue_model1_corrected_sen.loc[:, "Model"] = "Model1"
Female_results_continue_model2_corrected_sen.loc[:, "Model"] = "Model2"
Female_results_continue_model3_corrected_sen.loc[:, "Model"] = "Model3"
Female_results_continue_model4_corrected_sen.loc[:, "Model"] = "Model4"

Female_results_cont_sen_combine = pd.concat([Female_results_continue_model1_corrected_sen,
                                 Female_results_continue_model2_corrected_sen,
                                 Female_results_continue_model3_corrected_sen,
                                 Female_results_continue_model4_corrected_sen], ignore_index=True
                                )
Female_results_cont_sen_combine.shape


# In[404]:


# Female_results_cont_sen_combine.Outcome.unique()
# Female_results_cont_sen_combine.head()


# In[405]:


Outcomes_map2 = {'elasticity_median_median_of_qboxes_clipped_rank_INT': 'Liver Elasticity',
 'viscosity_median_median_of_qboxes_clipped_rank_INT': 'Liver Viscosity',
 'dispersion_median_median_of_qboxes_clipped_rank_INT': 'Liver Dispersion',
 'attenuation_coefficient_qbox_clipped_rank_INT': 'Attenuation coefficient',
 'velocity_median_median_of_qboxes_clipped_rank_INT': 'Liver Velocity',
 'speed_of_sound_qbox_clipped_rank_INT': 'Speed of Sound'}

# Female_sensitivity
Female_results_cont_sen_combine["Variable"] = Female_results_cont_sen_combine["Variable"].map(DPs_names)
Female_results_cont_sen_combine["Outcome"] = Female_results_cont_sen_combine["Outcome"].map(Outcomes_map2)


# In[406]:


# Male_sensitivity
Male_results_cont_sen_combine["Variable"] = Male_results_cont_sen_combine["Variable"].map(DPs_names)
Male_results_cont_sen_combine["Outcome"] = Male_results_cont_sen_combine["Outcome"].map(Outcomes_map2)


# In[407]:


# Only plot the Model2
Female_results_cont_sen_combine["Subgroup"] = "Female"
Female_results_cont_sen_combine = Female_results_cont_sen_combine.loc[Female_results_cont_sen_combine["Model"] == "Model2"]

Male_results_cont_sen_combine["Subgroup"] = "Male"
Male_results_cont_sen_combine = Male_results_cont_sen_combine.loc[Male_results_cont_sen_combine["Model"] == "Model2"]

Sex_group_sens_combine = pd.concat([Female_results_cont_sen_combine, Male_results_cont_sen_combine])
Sex_group_sens_combine


# In[408]:


# outcomes


# In[409]:


Categories_values = Sex_group_sens_combine["Subgroup"].unique() # Sex group

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = Sex_group_sens_combine[Sex_group_sens_combine["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.2, 0.2, len(Categories_values))  # Adjust -0.2 and 0.2 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Subgroup"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]
        
        p_values = []
        
        for idx, row in cat_data.iterrows():
            p_value = row.get("p_adj", 0.05)
            p_values.append(p_value)

        sig_threshold = 0.05
        is_significant = [p < sig_threshold for p in p_values]

        for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
            y_positions,
            cat_data["CI_lower"],
            cat_data["CI_upper"],
            cat_data["Coefficient"],
            is_significant
        )):
            ax.errorbar(x = coef,
                        y = y_pos,
                        xerr = [[coef - ci_lower], [ci_upper - coef]],
                        fmt = "none",
                        capsize = 4,
                        color = colors[j % len(colors)],
                        label = cat if n == 0 else None)
            # plot point
            if is_sig:
                # solid for significant results
                ax.scatter(coef, y_pos, s = 30,
                           c = colors[j % len(colors)],
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)
            else:
                ax.scatter(coef, y_pos, s = 30, facecolors = "none",
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)


        
        # ax.errorbar(x=cat_data['Coefficient'], 
        #             y=y_positions,
        #             xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
        #             fmt='o', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.5, 0.5) # set x-axis limits

plt.tight_layout()
# plt.savefig("./Results/Male_Continues_DPs_to_Liver_four_models_sensitivity1.pdf", bbox_inches='tight')
plt.show()


# ## Export results for Table

# In[410]:


# Export multiple dataframe to separate CSV files

# tables = {
#     "Male_results_continue_model1_corrected": Male_results_continue_model1_corrected,
#     "Male_results_continue_model2_corrected" : Male_results_continue_model2_corrected,
#     "Male_results_continue_model3_corrected" : Male_results_continue_model3_corrected,
#     "Male_results_continue_model4_corrected" : Male_results_continue_model4_corrected,
#     "Male_results_continue_model1_sig" : Male_results_continue_model1_sig,
#     "Male_results_continue_model2_sig" : Male_results_continue_model2_sig,
#     "Male_results_continue_model3_sig" : Male_results_continue_model3_sig,
#     "Male_results_continue_model4_sig" : Male_results_continue_model4_sig,
#     "Female_results_continue_model1_corrected": Female_results_continue_model1_corrected,
#     "Female_results_continue_model2_corrected" : Female_results_continue_model2_corrected,
#     "Female_results_continue_model3_corrected" : Female_results_continue_model3_corrected,
#     "Female_results_continue_model4_corrected" : Female_results_continue_model4_corrected,
#     "Female_results_continue_model1_sig" : Female_results_continue_model1_sig,
#     "Female_results_continue_model2_sig" : Female_results_continue_model2_sig,
#     "Female_results_continue_model3_sig" : Female_results_continue_model3_sig,
#     "Female_results_continue_model4_sig" : Female_results_continue_model4_sig,
#     "results_continue_model1_corrected": results_continue_model1_corrected,
#     "results_continue_model2_corrected" : results_continue_model2_corrected,
#     "results_continue_model3_corrected" : results_continue_model3_corrected,
#     "results_continue_model4_corrected" : results_continue_model4_corrected,
#     "results_continue_model1_sig" : results_continue_model1_sig,
#     "results_continue_model2_sig" : results_continue_model2_sig,
#     "results_continue_model3_sig" : results_continue_model3_sig,
#     "results_continue_model4_sig" : results_continue_model4_sig,
#     "results_cat_model1" : results_cat_model1,
#     "results_cat_model2" : results_cat_model2,
#     "results_cat_model3" : results_cat_model3,
#     "results_cat_model4" : results_cat_model4
# }

tables = {
    "Female_results_cont_sen_combine" : Female_results_cont_sen_combine,
    "Male_results_cont_sen_combine" : Male_results_cont_sen_combine,
    "Female_results_cont_combine" : Female_results_cont_combine,
    "Male_results_cont_combine" : Male_results_cont_combine
}

for name, df in tables.items():
    df.to_csv(f'./Results/{name}.csv', index = False)


# # Divide by the BMI >= 25

# In[411]:


print(merge_df_MAFLD_filter_3["bmi"].isnull().sum())
merge_df_MAFLD_filter_3.shape


# In[256]:


merge_df_MAFLD_filter_4 = merge_df_MAFLD_filter_3.dropna(subset = ["bmi"])
merge_df_MAFLD_filter_4.shape

df_highBMI = merge_df_MAFLD_filter_4.loc[merge_df_MAFLD_filter_4["bmi"] >= 25]
print(df_highBMI.shape)

df_lowerBMI = merge_df_MAFLD_filter_4.loc[merge_df_MAFLD_filter_4["bmi"] < 25]
df_lowerBMI.shape


# In[257]:


# Model2
model2 = ["age", 'sex_x', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']

# for hPDI index adjusted for alcohol intake 
model2_2 = ["age", 'sex_x',  'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use',"alcohol_g"]

# Case 1, not sensitivity analysis
df_highBMI_results_continue_model2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_highBMI, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_highBMI, predictors = predict_var, outcomes = livers, control_vars = model2)
        df_highBMI_results_continue_model2 = pd.concat([df_highBMI_results_continue_model2, lm_fit], ignore_index=True)

display(df_highBMI_results_continue_model2.head())

df_lowerBMI_results_continue_model2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_lowerBMI, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_lowerBMI, predictors = predict_var, outcomes = livers, control_vars = model2)
        df_lowerBMI_results_continue_model2 = pd.concat([df_lowerBMI_results_continue_model2, lm_fit], ignore_index=True)
        
display(df_lowerBMI_results_continue_model2.head())


# Case 2,  sensitivity analysis
df_highBMI_results_continue_model2_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_highBMI, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_highBMI, predictors = predict_var, outcomes = livers, control_vars = model2)
        df_highBMI_results_continue_model2_sen = pd.concat([df_highBMI_results_continue_model2_sen, lm_fit], ignore_index=True)

display(df_highBMI_results_continue_model2_sen.head())

df_lowerBMI_results_continue_model2_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_lowerBMI, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_lowerBMI, predictors = predict_var, outcomes = livers, control_vars = model2)
        df_lowerBMI_results_continue_model2_sen = pd.concat([df_lowerBMI_results_continue_model2_sen, lm_fit], ignore_index=True)
        
df_lowerBMI_results_continue_model2_sen.head()


# In[258]:


df_lowerBMI_results_continue_model2["Subgroup"] = "BMI < 25kg/m2"
df_highBMI_results_continue_model2["Subgroup"] = "BMI >= 25kg/m2"

# Adjust for pvalue
BMI_sub_models = [df_lowerBMI_results_continue_model2, df_highBMI_results_continue_model2]
BMI_sub_models_correct = []
BMI_sub_models_correct_sig = []

for i, df in enumerate(BMI_sub_models):
    df_corrected_sen, df_sig_sen = fdr_correction_(df)
    BMI_sub_models_correct.append(df_corrected_sen)
    BMI_sub_models_correct_sig.append(df_sig_sen)

df_lowerBMI_results_continue_correct_model2, df_highBMI_results_continue_correct_model2 = BMI_sub_models_correct
display(df_lowerBMI_results_continue_correct_model2.head())
df_highBMI_results_continue_correct_model2.head()

BMI_sub_models_correct_combine = pd.concat([df_lowerBMI_results_continue_correct_model2, df_highBMI_results_continue_correct_model2])


# In[260]:


BMI_sub_models_correct_combine["Subgroup"].value_counts()

BMI_sub_models_correct_combine["Variable"] = BMI_sub_models_correct_combine["Variable"].map(DPs_names)
BMI_sub_models_correct_combine["Outcome"] = BMI_sub_models_correct_combine["Outcome"].map(Outcome_name)


# In[416]:


Categories_values = BMI_sub_models_correct_combine["Subgroup"].unique() # BMI group

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = BMI_sub_models_correct_combine[BMI_sub_models_correct_combine["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.2, 0.2, len(Categories_values))  # Adjust -0.2 and 0.2 for desired spacing

    # Set the colors
    colors = [# '#E64B35', '#4DBBD5', 
              '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Subgroup"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]
        
        p_values = []
        
        for idx, row in cat_data.iterrows():
            p_value = row.get("p_adj", 0.05)
            p_values.append(p_value)

        sig_threshold = 0.05
        is_significant = [p < sig_threshold for p in p_values]

        for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
            y_positions,
            cat_data["CI_lower"],
            cat_data["CI_upper"],
            cat_data["Coefficient"],
            is_significant
        )):
            ax.errorbar(x = coef,
                        y = y_pos,
                        xerr = [[coef - ci_lower], [ci_upper - coef]],
                        fmt = "none",
                        capsize = 4,
                        color = colors[j % len(colors)],
                        label = cat if n == 0 else None)
            # plot point
            if is_sig:
                # solid for significant results
                ax.scatter(coef, y_pos, s = 30,
                           c = colors[j % len(colors)],
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)
            else:
                ax.scatter(coef, y_pos, s = 30, facecolors = "none",
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)


        
        # ax.errorbar(x=cat_data['Coefficient'], 
        #             y=y_positions,
        #             xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
        #             fmt='o', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.55, 0.55) # set x-axis limits

plt.tight_layout()
plt.savefig("./Results/BMI_Subgroup_Continues_DPs_to_Liver_four_models_v2.pdf", bbox_inches='tight')
plt.show()


# ## Plot the combined results

# In[263]:


# cobine the results for Effect modification analysis
display(BMI_sub_models_correct_combine.head())
Sex_group_results.head()


# In[265]:


BMI_sub_models_correct_combine["Subgroup"].value_counts()


# In[279]:


GROUP_ORDER1 = ["Male", "Female"]
GROUP_ORDER2 = ["BMI < 25kg/m2", "BMI >= 25kg/m2"]

GROUP_COLOR_MAP = {
    "Male": "#4DBBD5",
    "Female" : "#E64B35",
    "BMI < 25kg/m2" : "#00A087",
    "BMI >= 25kg/m2" : "#3C5488",
}

fig = plt.figure(figsize = (16, 20))

# GridSpec: 4 rows, 3 cols, with gap between row 2 and 3
gs = gridspec.GridSpec(
    4, 3, figure = fig,
    height_ratios = [1, 1, 1, 1],
    hspace = 0.45, # vertical spacing between all rows
    wspace = 0.15,
    left = 0.08
)

# sensitivity 1
print(Sex_group_results["Subgroup"].unique())

axes_sen1_1 = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

# bottom sens2
axes_sen2_1 = [fig.add_subplot(gs[r, c]) for r in range(2, 4) for c in range(3)]

plot_forest(Sex_group_results, 
            axes_sen1_1,
           group_col= "Subgroup",
           group_order= GROUP_ORDER1,
           group_color_map = GROUP_COLOR_MAP)

plot_forest(BMI_sub_models_correct_combine, 
            axes_sen2_1,
           group_col= "Subgroup",
           group_order= GROUP_ORDER2,
           group_color_map = GROUP_COLOR_MAP)

# Row labels in the left margin
fig.text(0.01, 0.9, "A", fontsize = 16, fontweight = "bold")
fig.text(0.01, 0.48, "B", fontsize = 16, fontweight = "bold")

# Legend A
color_handles_A = [Line2D([0], [0], marker = "o", color = "w", 
                        markerfacecolor = GROUP_COLOR_MAP[cat],
                        markeredgecolor = GROUP_COLOR_MAP[cat], 
                        markersize = 8, label = cat)
                for cat in GROUP_ORDER1]

fig.legend(color_handles_A, GROUP_ORDER1,
          loc = "upper center", ncol = 4, frameon = True, fontsize = 12, 
          bbox_to_anchor = (0.5, 0.92))

# Legend B
color_handles_B = [Line2D([0], [0], marker = "o", color = "w", 
                        markerfacecolor = GROUP_COLOR_MAP[cat],
                        markeredgecolor = GROUP_COLOR_MAP[cat], 
                        markersize = 8, label = cat)
                for cat in GROUP_ORDER2]

fig.legend(color_handles_B, GROUP_ORDER2,
          loc = "upper center", ncol = 4, frameon = True, fontsize = 12, 
          bbox_to_anchor = (0.5, 0.5))


# save the figure
fig.savefig("./Results/Combined_DPs_to_Liver_Sensitivity_stratified.pdf", bbox_inches = "tight", dpi = 300)
plt.show()


# In[417]:


df_lowerBMI_results_continue_model2_sen["Subgroup"] = "BMI < 25kg/m2"
df_highBMI_results_continue_model2_sen["Subgroup"] = "BMI >= 25kg/m2"

# Adjust for pvalue
BMI_sub_models_sen = [df_lowerBMI_results_continue_model2_sen, df_highBMI_results_continue_model2_sen]

BMI_sub_models_sen_correct = []
BMI_sub_models_sen_correct_sig = []

for i, df in enumerate(BMI_sub_models_sen):
    df_corrected_sen, df_sig_sen = fdr_correction_(df)
    BMI_sub_models_sen_correct.append(df_corrected_sen)
    BMI_sub_models_sen_correct_sig.append(df_sig_sen)

BMI_sub_models_sen_correct_combine = pd.concat(BMI_sub_models_sen_correct, ignore_index=True )
BMI_sub_models_sen_correct_combine


# In[418]:


BMI_sub_models_sen_correct_combine["Variable"] = BMI_sub_models_sen_correct_combine["Variable"].map(DPs_names)
BMI_sub_models_sen_correct_combine["Outcome"] = BMI_sub_models_sen_correct_combine["Outcome"].map(Outcomes_map2)


# In[419]:


Categories_values = BMI_sub_models_sen_correct_combine["Subgroup"].unique() # Sex group

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = BMI_sub_models_sen_correct_combine[BMI_sub_models_sen_correct_combine["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.2, 0.2, len(Categories_values))  # Adjust -0.2 and 0.2 for desired spacing

    # Set the colors
    colors = [# '#E64B35', '#4DBBD5', 
              '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Subgroup"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]
        
        p_values = []
        
        for idx, row in cat_data.iterrows():
            p_value = row.get("p_adj", 0.05)
            p_values.append(p_value)

        sig_threshold = 0.05
        is_significant = [p < sig_threshold for p in p_values]

        for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
            y_positions,
            cat_data["CI_lower"],
            cat_data["CI_upper"],
            cat_data["Coefficient"],
            is_significant
        )):
            ax.errorbar(x = coef,
                        y = y_pos,
                        xerr = [[coef - ci_lower], [ci_upper - coef]],
                        fmt = "none",
                        capsize = 4,
                        color = colors[j % len(colors)],
                        label = cat if n == 0 else None)
            # plot point
            if is_sig:
                # solid for significant results
                ax.scatter(coef, y_pos, s = 30,
                           c = colors[j % len(colors)],
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)
            else:
                ax.scatter(coef, y_pos, s = 30, facecolors = "none",
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)


        
        # ax.errorbar(x=cat_data['Coefficient'], 
        #             y=y_positions,
        #             xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
        #             fmt='o', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.55, 0.55) # set x-axis limits

plt.tight_layout()
# plt.savefig("./Results/Male_Continues_DPs_to_Liver_four_models_sensitivity1.pdf", bbox_inches='tight')
plt.show()


# # Divide by the mean of participants' age

# In[420]:


merge_df_MAFLD_filter_3.shape


# In[421]:


merge_df_MAFLD_filter_3["age"].isnull().sum()


# In[422]:


merge_df_MAFLD_filter_3["age"].mean()


# In[423]:


df_older = merge_df_MAFLD_filter_3.loc[merge_df_MAFLD_filter_3["age"] >= merge_df_MAFLD_filter_3["age"].mean()]
print(df_older.shape)

df_younger = merge_df_MAFLD_filter_3.loc[merge_df_MAFLD_filter_3["age"] < merge_df_MAFLD_filter_3["age"].mean()]
df_younger.shape


# In[424]:


model2 = ['sex_x', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']

# for hPDI index adjusted for alcohol intake 
model2_2 = ['sex_x',  'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use',"alcohol_g"]

# Case 1, not sensitivity analysis
df_older_results_continue_model2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model2)
        df_older_results_continue_model2 = pd.concat([df_older_results_continue_model2, lm_fit], ignore_index=True)

display(df_older_results_continue_model2.head())

df_younger_results_continue_model2 = pd.DataFrame()
for livers in Outcomes:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_younger, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_younger, predictors = predict_var, outcomes = livers, control_vars = model2)
        df_younger_results_continue_model2 = pd.concat([df_younger_results_continue_model2, lm_fit], ignore_index=True)
        
display(df_younger_results_continue_model2.head())


# In[425]:


# Case 2,  sensitivity analysis
df_older_results_continue_model2_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model2)
        df_older_results_continue_model2_sen = pd.concat([df_older_results_continue_model2_sen, lm_fit], ignore_index=True)

display(df_older_results_continue_model2_sen.head())

df_younger_results_continue_model2_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_younger, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_younger, predictors = predict_var, outcomes = livers, control_vars = model2)
        df_younger_results_continue_model2_sen = pd.concat([df_younger_results_continue_model2_sen, lm_fit], ignore_index=True)
        
df_younger_results_continue_model2_sen.head()


# model3 = ['sex_x', 'smoking_status', 'edu_status', 
#          'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "bmi"]

# model3_2 = ['sex_x', 'smoking_status', 'edu_status', 
#          'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "alcohol_g", 'bmi']

# model4 = ['sex_x',  'smoking_status', 'edu_status', 
#          'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "NSAID_use", "CVD_f_history", "T2D_f_history"]

# model4_2 = ['sex_x', 'smoking_status', 'edu_status', 
#          'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "NSAID_use", "alcohol_g", "CVD_f_history", "T2D_f_history"]


# In[426]:


df_older_results_continue_model2["Subgroup"] = "Age > 52"
df_younger_results_continue_model2["Subgroup"] = "Age <= 52"

# Adjust for pvalue
Age_sub_models = [df_older_results_continue_model2, df_younger_results_continue_model2]
Age_sub_models_correct = []
Age_sub_models_correct_sig = []

for i, df in enumerate(Age_sub_models):
    df_corrected_sen, df_sig_sen = fdr_correction_(df)
    Age_sub_models_correct.append(df_corrected_sen)
    Age_sub_models_correct_sig.append(df_sig_sen)

Age_sub_models_correct_combine = pd.concat(Age_sub_models_correct, ignore_index=True )
Age_sub_models_correct_combine.head()

Age_sub_models_correct_combine["Variable"] = Age_sub_models_correct_combine["Variable"].map(DPs_names)
Age_sub_models_correct_combine["Outcome"] = Age_sub_models_correct_combine["Outcome"].map(Outcome_name)

Categories_values = Age_sub_models_correct_combine["Subgroup"].unique() 
Categories_values

Age_sub_models_correct_combine


# In[427]:


Age_sub_models_correct_combine[Age_sub_models_correct_combine["p_adj"] < 0.05]


# In[428]:


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = Age_sub_models_correct_combine[Age_sub_models_correct_combine["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.2, 0.2, len(Categories_values))  # Adjust -0.2 and 0.2 for desired spacing

    # Set the colors
    colors = [# '#E64B35', '#4DBBD5', 
              # '#00A087', '#3C5488', 
              '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Subgroup"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]
        
        p_values = []
        
        for idx, row in cat_data.iterrows():
            p_value = row.get("p_adj", 0.05)
            p_values.append(p_value)

        sig_threshold = 0.05
        is_significant = [p < sig_threshold for p in p_values]

        for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
            y_positions,
            cat_data["CI_lower"],
            cat_data["CI_upper"],
            cat_data["Coefficient"],
            is_significant
        )):
            ax.errorbar(x = coef,
                        y = y_pos,
                        xerr = [[coef - ci_lower], [ci_upper - coef]],
                        fmt = "none",
                        capsize = 4,
                        color = colors[j % len(colors)],
                        label = cat if n == 0 else None)
            # plot point
            if is_sig:
                # solid for significant results
                ax.scatter(coef, y_pos, s = 30,
                           c = colors[j % len(colors)],
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)
            else:
                ax.scatter(coef, y_pos, s = 30, facecolors = "none",
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)


        
        # ax.errorbar(x=cat_data['Coefficient'], 
        #             y=y_positions,
        #             xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
        #             fmt='o', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.5, 0.5) # set x-axis limits

plt.tight_layout()
# plt.savefig("./Results/Age_Subgroup_Continues_DPs_to_Liver_four_models_v2.pdf", bbox_inches='tight')
plt.show()


# In[429]:


df_older_results_continue_model2_sen["Subgroup"] = "Age > 52"
df_younger_results_continue_model2_sen["Subgroup"] = "Age <= 52"

# Adjust for pvalue
Age_sub_models_sens = [df_older_results_continue_model2_sen, df_younger_results_continue_model2_sen]
Age_sub_models_sens_correct = []
Age_sub_models_sens_correct_sig = []

for i, df in enumerate(Age_sub_models_sens):
    df_corrected_sen, df_sig_sen = fdr_correction_(df)
    Age_sub_models_sens_correct.append(df_corrected_sen)
    Age_sub_models_sens_correct_sig.append(df_sig_sen)

Age_sub_models_correct_sen_combine = pd.concat(Age_sub_models_sens_correct, ignore_index=True )
Age_sub_models_correct_sen_combine.head()

Age_sub_models_correct_sen_combine["Variable"] = Age_sub_models_correct_sen_combine["Variable"].map(DPs_names)
Age_sub_models_correct_sen_combine["Outcome"] = Age_sub_models_correct_sen_combine["Outcome"].map(Outcomes_map2)

Categories_values = Age_sub_models_correct_sen_combine["Subgroup"].unique()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = Age_sub_models_correct_sen_combine[Age_sub_models_correct_sen_combine["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.2, 0.2, len(Categories_values))  # Adjust -0.2 and 0.2 for desired spacing

    # Set the colors
    colors = [# '#E64B35', '#4DBBD5', 
              # '#00A087', '#3C5488', 
              '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Subgroup"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]
        
        p_values = []
        
        for idx, row in cat_data.iterrows():
            p_value = row.get("p_adj", 0.05)
            p_values.append(p_value)

        sig_threshold = 0.05
        is_significant = [p < sig_threshold for p in p_values]

        for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
            y_positions,
            cat_data["CI_lower"],
            cat_data["CI_upper"],
            cat_data["Coefficient"],
            is_significant
        )):
            ax.errorbar(x = coef,
                        y = y_pos,
                        xerr = [[coef - ci_lower], [ci_upper - coef]],
                        fmt = "none",
                        capsize = 4,
                        color = colors[j % len(colors)],
                        label = cat if n == 0 else None)
            # plot point
            if is_sig:
                # solid for significant results
                ax.scatter(coef, y_pos, s = 30,
                           c = colors[j % len(colors)],
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)
            else:
                ax.scatter(coef, y_pos, s = 30, facecolors = "none",
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)


        
        # ax.errorbar(x=cat_data['Coefficient'], 
        #             y=y_positions,
        #             xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
        #             fmt='o', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.55, 0.55) # set x-axis limits

plt.tight_layout()
# plt.savefig("./Results/Age_Subgroup_Continues_DPs_to_Liver_four_models_v2.pdf", bbox_inches='tight')
plt.show()


# In[430]:


Age_sub_models_correct_sen_combine


# In[431]:


Age_sub_models_correct_sen_combine[Age_sub_models_correct_sen_combine["p_adj"] < 0.05]


# # End here

# In[432]:


model2 = ['sex_x', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use']

# for hPDI index adjusted for alcohol intake 
model2_2 = ['sex_x',  'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use',"alcohol_g"]

model3 = ['sex_x', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "bmi"]

model3_2 = ['sex_x', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "alcohol_g", 'bmi']

model4 = ['sex_x',  'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "NSAID_use", "CVD_f_history", "T2D_f_history"]

model4_2 = ['sex_x', 'smoking_status', 'edu_status', 
         'Vit_use', 'MET_hour', 'sleep_hours_daily', 'Hormone_use', "NSAID_use", "alcohol_g", "CVD_f_history", "T2D_f_history"]


# In[433]:


df_older_results_continue_model2_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model2)
        df_older_results_continue_model2_sen = pd.concat([df_older_results_continue_model2_sen, lm_fit], ignore_index=True)


# In[434]:


display(df_older_results_continue_model2_sen)


# In[435]:


model2


# In[436]:


# Older participants
# Model1
df_older_results_continue_model1_sen = pd.DataFrame()

for livers in Outcomes2:
    # print(metabolites)
    lm_fit = run_regression_formula(df = df_older, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    df_older_results_continue_model1_sen = pd.concat([df_older_results_continue_model1_sen, lm_fit], ignore_index=True)

# Model2
df_older_results_continue_model2_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model2)
        df_older_results_continue_model2_sen = pd.concat([df_older_results_continue_model2_sen, lm_fit], ignore_index=True)
    
# Model3
df_older_results_continue_model3_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model3)
        df_older_results_continue_model3_sen = pd.concat([df_older_results_continue_model3_sen, lm_fit], ignore_index=True)

# Model4
df_older_results_continue_model4_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = df_older, predictors = predict_var, outcomes = livers, control_vars = model4)
        df_older_results_continue_model4_sen = pd.concat([df_older_results_continue_model4_sen, lm_fit], ignore_index=True)


# In[437]:


# Adjust for pvalue
Older_df_models_sen = [#df_older_results_continue_model1_sen, 
    df_older_results_continue_model2_sen
    # , df_older_results_continue_model3_sen, df_older_results_continue_model4_sen
]

Older_corrected_dfs_sen2 = []
Older_significant_dfs_sen2 = []

for i, df in enumerate(Older_df_models_sen):
    df_corrected_sen, df_sig_sen = fdr_correction_(df)
    Older_corrected_dfs_sen2.append(df_corrected_sen)
    Older_significant_dfs_sen2.append(df_sig_sen)


# In[438]:


Older_significant_dfs_sen2


# In[439]:


Older_results_continue_model1_corrected_sen, Older_results_continue_model2_corrected_sen, Older_results_continue_model3_corrected_sen, Older_results_continue_model4_corrected_sen = Older_corrected_dfs_sen


# In[ ]:


Older_corrected_dfs_sen


# In[ ]:


Older_results_continue_model2_corrected_sen


# In[ ]:


df_older_results_continue_model2_sen
from statsmodels.stats.multitest import multipletests

# apply FDR correction
reject, pval_corrected, _, _ = multipletests(df_older_results_continue_model2_sen["P_value"], method = "fdr_bh")
df_older_results_continue_model2_sen["pval_fdr"] = pval_corrected
df_older_results_continue_model2_sen


# In[ ]:


# Access to results
Older_results_continue_model1_corrected_sen, Older_results_continue_model2_corrected_sen, Older_results_continue_model3_corrected_sen, Older_results_continue_model4_corrected_sen = Older_corrected_dfs_sen
# Male_results_continue_model1_sig_sen, Male_results_continue_model2_sig_sen, Male_results_continue_model3_sig_sen, Male_results_continue_model4_sig_sen = Male_significant_dfs_sen


Older_results_continue_model1_corrected_sen.loc[:, "Model"] = "Model1"
Older_results_continue_model2_corrected_sen.loc[:, "Model"] = "Model2"
Older_results_continue_model3_corrected_sen.loc[:, "Model"] = "Model3"
Older_results_continue_model4_corrected_sen.loc[:, "Model"] = "Model4"

Older_results_cont_sen_combine = pd.concat([Older_results_continue_model1_corrected_sen,
                                 Older_results_continue_model2_corrected_sen,
                                 Older_results_continue_model3_corrected_sen,
                                 Older_results_continue_model4_corrected_sen], ignore_index=True
                                )
Older_results_cont_sen_combine.shape

Older_results_cont_sen_combine["Variable"] = Older_results_cont_sen_combine["Variable"].map(DPs_names)
Older_results_cont_sen_combine["Outcome"] = Older_results_cont_sen_combine["Outcome"].map(Outcomes_map2)


# In[ ]:


Older_results_cont_sen_combine[Older_results_cont_sen_combine["Model"] == "Model2"]


# In[ ]:


# Plot the results
# Older population
Categories_values = Older_results_cont_sen_combine["Model"].unique() # Model1-4
# dps_values = Male_results_cont_combine["Variable"].unique() # DPs

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = Older_results_cont_sen_combine[Older_results_cont_sen_combine["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Model"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]
        
        p_values = []
        
        for idx, row in cat_data.iterrows():
            p_value = row.get("p_adj", 0.05)
            p_values.append(p_value)

        sig_threshold = 0.05
        is_significant = [p < sig_threshold for p in p_values]

        for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
            y_positions,
            cat_data["CI_lower"],
            cat_data["CI_upper"],
            cat_data["Coefficient"],
            is_significant
        )):
            ax.errorbar(x = coef,
                        y = y_pos,
                        xerr = [[coef - ci_lower], [ci_upper - coef]],
                        fmt = "none",
                        capsize = 4,
                        color = colors[j % len(colors)],
                        label = cat if n == 0 else None)
            # plot point
            if is_sig:
                # solid for significant results
                ax.scatter(coef, y_pos, s = 30,
                           c = colors[j % len(colors)],
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)
            else:
                ax.scatter(coef, y_pos, s = 30, facecolors = "none",
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)


        
        # ax.errorbar(x=cat_data['Coefficient'], 
        #             y=y_positions,
        #             xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
        #             fmt='o', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.55, 0.55) # set x-axis limits

plt.tight_layout()
# plt.savefig("./Results/Older_Continues_DPs_to_Liver_four_models_sensitivity1.pdf", bbox_inches='tight')
plt.show()


# In[ ]:


Older_results_continue_model2_corrected_sen[Older_results_continue_model2_corrected_sen["p_adj"] < 0.05]


# In[ ]:


df_younger_results_continue_model1_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    lm_fit = run_regression_formula(df = df_younger, 
                                      predictors = predictors, 
                                      outcomes = livers, control_vars = None)
    lm_fit = pd.DataFrame(lm_fit)
    df_younger_results_continue_model1_sen = pd.concat([df_younger_results_continue_model1_sen, lm_fit], ignore_index=True)

# Model2
df_younger_results_continue_model2_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_younger, predictors = predict_var, outcomes = livers, control_vars = model2_2)
        else:
            lm_fit = run_regression_formula(df = df_younger, predictors = predict_var, outcomes = livers, control_vars = model2)
        df_younger_results_continue_model2_sen = pd.concat([df_younger_results_continue_model2_sen, lm_fit], ignore_index=True)
    
# Model3
df_younger_results_continue_model3_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_younger, predictors = predict_var, outcomes = livers, control_vars = model3_2)
        else:
            lm_fit = run_regression_formula(df = df_younger, predictors = predict_var, outcomes = livers, control_vars = model3)
        df_younger_results_continue_model3_sen = pd.concat([df_younger_results_continue_model3_sen, lm_fit], ignore_index=True)

# Model4
df_younger_results_continue_model4_sen = pd.DataFrame()
for livers in Outcomes2:
    # print(metabolites)
    for predict_var in predictors:
        if predict_var == "hPDI_score_eadj_scaled":
            lm_fit = run_regression_formula(df = df_younger, predictors = predict_var, outcomes = livers, control_vars = model4_2)
        else:
            lm_fit = run_regression_formula(df = df_younger, predictors = predict_var, outcomes = livers, control_vars = model4)
        df_younger_results_continue_model4_sen = pd.concat([df_younger_results_continue_model4_sen, lm_fit], ignore_index=True)

# Adjust for pvalue
Younger_df_models_sen = [df_younger_results_continue_model1_sen, df_younger_results_continue_model2_sen, 
             df_younger_results_continue_model3_sen, df_younger_results_continue_model4_sen]

Younger_corrected_dfs_sen = []
Younger_significant_dfs_sen = []

for i, df in enumerate(Younger_df_models_sen):
    df_corrected_sen, df_sig_sen = fdr_correction_(df)
    Younger_corrected_dfs_sen.append(df_corrected_sen)
    Younger_significant_dfs_sen.append(df_sig_sen)


# In[ ]:


# Access to results
Younger_results_continue_model1_corrected_sen, Younger_results_continue_model2_corrected_sen, Younger_results_continue_model3_corrected_sen, Younger_results_continue_model4_corrected_sen = Younger_corrected_dfs_sen
# Male_results_continue_model1_sig_sen, Male_results_continue_model2_sig_sen, Male_results_continue_model3_sig_sen, Male_results_continue_model4_sig_sen = Male_significant_dfs_sen


# In[ ]:


Younger_results_continue_model1_corrected_sen


# In[ ]:


Younger_results_continue_model1_corrected_sen.loc[:, "Model"] = "Model1"
Younger_results_continue_model2_corrected_sen.loc[:, "Model"] = "Model2"
Younger_results_continue_model3_corrected_sen.loc[:, "Model"] = "Model3"
Younger_results_continue_model4_corrected_sen.loc[:, "Model"] = "Model4"

Younger_results_cont_sen_combine = pd.concat([Younger_results_continue_model1_corrected_sen,
                                 Younger_results_continue_model2_corrected_sen,
                                 Younger_results_continue_model3_corrected_sen,
                                 Younger_results_continue_model4_corrected_sen], ignore_index=True
                                )
Younger_results_cont_sen_combine.shape


# In[ ]:


Younger_results_cont_sen_combine["Variable"] = Younger_results_cont_sen_combine["Variable"].map(DPs_names)
Younger_results_cont_sen_combine["Outcome"] = Younger_results_cont_sen_combine["Outcome"].map(Outcomes_map2)

# Plot the results
# Older population
Categories_values = Younger_results_cont_sen_combine["Model"].unique() # Model1-4
# dps_values = Male_results_cont_combine["Variable"].unique() # DPs

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    outcome_data = Younger_results_cont_sen_combine[Younger_results_cont_sen_combine["Outcome"] == outcome] # Corrected line

    # Positions for each category within the group
    dps = outcome_data["Variable"].unique()
    positions = np.arange(len(dps))

    # Offset for each group (DPs)
    offset = np.linspace(-0.3, 0.3, len(Categories_values))  # Adjust -0.3 and 0.3 for desired spacing

    # Set the colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    # color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(Categories_values)}

    # Plotting the point estimates and confidence intervals
    for j, cat in enumerate(Categories_values):
        cat_data = outcome_data[outcome_data["Model"] == cat]
        y_positions = [positions[np.where(dps == dp)[0][0]] + offset[j] for dp in cat_data['Variable']]
        
        p_values = []
        
        for idx, row in cat_data.iterrows():
            p_value = row.get("p_adj", 0.05)
            p_values.append(p_value)

        sig_threshold = 0.05
        is_significant = [p < sig_threshold for p in p_values]

        for n, (y_pos, ci_lower, ci_upper, coef, is_sig) in enumerate(zip(
            y_positions,
            cat_data["CI_lower"],
            cat_data["CI_upper"],
            cat_data["Coefficient"],
            is_significant
        )):
            ax.errorbar(x = coef,
                        y = y_pos,
                        xerr = [[coef - ci_lower], [ci_upper - coef]],
                        fmt = "none",
                        capsize = 4,
                        color = colors[j % len(colors)],
                        label = cat if n == 0 else None)
            # plot point
            if is_sig:
                # solid for significant results
                ax.scatter(coef, y_pos, s = 30,
                           c = colors[j % len(colors)],
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)
            else:
                ax.scatter(coef, y_pos, s = 30, facecolors = "none",
                           edgecolors = colors[j % len(colors)], # edge color
                           linewidths = 1,
                           zorder = 3)


        
        # ax.errorbar(x=cat_data['Coefficient'], 
        #             y=y_positions,
        #             xerr=[cat_data['Coefficient'] - cat_data['CI_lower'], cat_data['CI_upper'] - cat_data['Coefficient']],
        #             fmt='o', capsize=5, label=cat, color = colors[j % len(colors)]) # change point (o) to square (s)

    # Customize the plot
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Vertical line at zero
    ax.set_yticks(positions)
    ax.set_yticklabels(dps)
    ax.set_ylabel("DPs")
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Forest Plot for {outcome}')
    ax.invert_yaxis()  # Invert y-axis to have Q1 at the top
    ax.legend()
    ax.set_xlim(-0.5, 0.5) # set x-axis limits

plt.tight_layout()
# plt.savefig("./Results/Younger_Continues_DPs_to_Liver_four_models_sensitivity1.pdf", bbox_inches='tight')
plt.show()


# In[ ]:


Younger_results_cont_sen_combine[Younger_results_cont_sen_combine["p_adj"] < 0.05]


# In[ ]:





# In[ ]:





# In[ ]:






# ============================================================
Source: A0_helpfunction_Copy1.py
# ============================================================

#### I will write several function for using in this .py files
#### Author: Keyong Deng, LUMC
#### Leiden University Medical Center


# Make the MLR model for DPs separately, adjusted for age, bmi, and sex
## Write a function to run the multiple linear regression analysis\

import statsmodels.formula.api as smf
import scipy.stats as ss
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

    results_list = pd.DataFrame(results_list)
    return results_list


# Categories function to run 
def run_GLM_Group(df, predictors, outcomes, control_vars = None):

    import statsmodels.formula.api as smf
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

    import pandas as pd
    import numpy as np
    import scipy.stats as ss
    
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
    Estimate_ci = f"{Estimate:.3f}[{Lower_ci:.3f}, {Upper_ci:.3f}]"
    ACME_Pval = med_summary.loc[med_summary["index"] == "ACME (average)"]["P-value"].values[0]

    Estimate_ADE = med_summary.loc[med_summary["index"] == "ADE (average)"]["Estimate"].values[0]
    Lower_ci_ADE = med_summary.loc[med_summary["index"] == "ADE (average)"]["Lower CI bound"].values[0]
    Upper_ci_ADE = med_summary.loc[med_summary["index"] == "ADE (average)"]["Upper CI bound"].values[0]
    
    # comebine them into a string
    Estimate_ci_ADE = f"{Estimate_ADE:.3f}[{Lower_ci_ADE:.3f}, {Upper_ci_ADE:.3f}]"
    ADE_Pval = med_summary.loc[med_summary["index"] == "ADE (average)"]["P-value"].values[0]

    # Total effect 
    Estimate_total = med_summary.loc[med_summary["index"] == "Total effect"]["Estimate"].values[0]
    Lower_ci_total = med_summary.loc[med_summary["index"] == "Total effect"]["Lower CI bound"].values[0]
    Upper_ci_total = med_summary.loc[med_summary["index"] == "Total effect"]["Upper CI bound"].values[0]
    # comebine them into a string
    Estimate_ci_total = f"{Estimate_total:.3f}[{Lower_ci_total:.3f}, {Upper_ci_total:.3f}]"
    Total_Pval = med_summary.loc[med_summary["index"] == "Total effect"]["P-value"].values[0]

    # proportion mediation (Prop. mediated (average))
    proportion_mediation = med_summary.loc[med_summary["index"] == "Prop. mediated (average)"]["Estimate"].values[0]
    Lower_ci_proportion = med_summary.loc[med_summary["index"] == "Prop. mediated (average)"]["Lower CI bound"].values[0]
    Upper_ci_proportion = med_summary.loc[med_summary["index"] == "Prop. mediated (average)"]["Upper CI bound"].values[0]     
    # combine them into a string
    proportion_mediation_all = f"{proportion_mediation:.3f}[{Lower_ci_proportion:.3f}, {Upper_ci_proportion:.3f}]"
    print(proportion_mediation_all)
            
    # pvalue
    proportion_mediation_Pval = med_summary.loc[med_summary["index"] == "Prop. mediated (average)"]["P-value"].values[0]
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