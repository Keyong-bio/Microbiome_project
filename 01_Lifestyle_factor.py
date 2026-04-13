#!/usr/bin/env python
# coding: utf-8

# ## Lifestyle factors in 10K
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from pheno_utils import PhenoLoader
from pheno_utils.config import (
    DATASETS_PATH, 
    BULK_DATA_PATH
    )

import pandas as pd
import numpy as np
import random
import os


# ## sex and age

# In[2]:


# population data includes sex
population_df = PhenoLoader('population')
population_df = population_df[population_df.fields].reset_index()
population_df


# In[3]:


# events including age, sex 
events_df = PhenoLoader('events')
events_df_base1 = events_df[["age_at_research_stage", "sex"]].loc[:,:,"00_00_visit",:,:]
events_df_base1.head()
events_df_base1 = events_df_base1.reset_index()
events_df_base1 = events_df_base1[['participant_id', 'age_at_research_stage', "sex"]]
events_df_base1


# In[4]:


age_sex = pd.merge(population_df, events_df_base1, on = "participant_id", 
                   how = "outer")
print(age_sex.shape)

age_sex.describe().round(1)

# absoulute number
display(age_sex['sex_x'].value_counts())
# Percentage
age_sex['sex_x'].value_counts(normalize=True).round(2)


# In[5]:


age_sex['sex_x'] = age_sex['sex_x'].combine_first(age_sex['sex_y'])
age_sex


# In[6]:


# BMI
BMI = PhenoLoader('anthropometrics')
BMI.dict
col = 'bmi'
BMI_base = BMI[[col] + ["age", "sex", "collection_date"]].loc[:,:,"00_00_visit",:,:]
# reset index for BMI_base
BMI_base_1 = BMI_base.reset_index()
BMI_base_1 = BMI_base_1.drop(BMI_base_1.columns[[1, 2]], axis=1)
BMI_base_1


# In[7]:


age_sex_bmi = pd.merge(age_sex, BMI_base_1, on = "participant_id", 
                       how = "outer")
age_sex_bmi


# In[8]:


age_sex_bmi.participant_id.nunique()


# In[10]:


# age_sex_bmi.to_csv("age_sex_bmi.csv", index = False)


# ## UKBB Survey

# In[9]:


dl = PhenoLoader('lifestyle_and_environment')
dl


# In[10]:


dl.dict
df_lifestyle = dl.dfs['lifestyle_and_environment']
df_lifestyle


# In[11]:


df_lifestyle_base = df_lifestyle[df_lifestyle.index.get_level_values('research_stage') == '00_00_visit']
df_lifestyle_base = df_lifestyle_base.reset_index()
display(df_lifestyle_base['research_stage'].value_counts())
df_lifestyle_base['participant_id'].nunique() # 8581 unique 


# In[12]:


df_lifestyle_base.shape


# In[13]:


# check the duplicates based on participant_id
df_lifestyle_base[df_lifestyle_base['participant_id'].duplicated(keep=False)]


# In[14]:


# remove the duplicated based only on one column, specify the subset of it, keep = Last
df_lifestyle_base = df_lifestyle_base.drop_duplicates(subset=['participant_id'], keep = "last")
df_lifestyle_base


# ## Smoking

# In[15]:


# smoking
[col for col in df_lifestyle_base.columns if "smoking" in col]

# smoking_former_age_stop_number, smoking_current_status
df_lifestyle_base_smoke = df_lifestyle_base[['participant_id','smoking_current_status', 'smoking_former_age_stop_number']]
df_lifestyle_base_smoke.head()

df_lifestyle_base_smoke1 = df_lifestyle_base_smoke.dropna(subset=['smoking_current_status'])
df_lifestyle_base_smoke1['smoking_current_status'].value_counts()


# In[16]:


col = 'smoking_current_status'

# Define conditions for classification
conditions = [
    (df_lifestyle_base_smoke1[col] == "Yes, on most or all days") | (df_lifestyle_base_smoke1[col] == "Only occasionally"),
    (df_lifestyle_base_smoke1[col] == "No") & (df_lifestyle_base_smoke1['smoking_former_age_stop_number'].isna()),
    (df_lifestyle_base_smoke1[col] == "No") & ~(df_lifestyle_base_smoke1['smoking_former_age_stop_number'].isna())
]

# Define corresponding choices for conditions
choices = ["Current", "Never", "Former"]

# Assign the smoking status based on conditions
df_lifestyle_base_smoke1['smoking_status'] = np.select(conditions, choices, default = np.nan)

display(df_lifestyle_base_smoke1['smoking_status'].value_counts())

# Show as the percentage
df_lifestyle_base_smoke1['smoking_status'].value_counts(normalize=True).round(3)


# In[17]:


print(df_lifestyle_base_smoke1.shape)

age_sex_bmi_smoke = pd.merge(age_sex_bmi, df_lifestyle_base_smoke1, on = 'participant_id', how = 'left')
print(age_sex_bmi_smoke.shape)
age_sex_bmi_smoke['smoking_status'].isna().sum() # 3486 missing


# ## Education

# In[18]:


pl_socio = PhenoLoader('sociodemographics', preferred_language = 'coding')
pl_socio = pl_socio.dfs['ukbb']

pl_socio_base = pl_socio[pl_socio.index.get_level_values('research_stage') == '00_00_visit']
pl_socio_base.index.get_level_values('participant_id') # 8636 obs, but 8581 unique individual


# In[19]:


pl_socio.head()


# In[20]:


[col for col in pl_socio_base.columns if "education" in col]


# In[21]:


pl_socio_base.index.get_level_values('participant_id').nunique()


# In[22]:


pl_socio_base1 = pl_socio_base.reset_index()
pl_socio_base1


# In[23]:


# 'domestic_education_qualifications'
pl_socio_base_edu = pl_socio_base1[['participant_id','domestic_education_qualifications']]
pl_socio_base_edu = pl_socio_base_edu.dropna()
display(pl_socio_base_edu.shape)
pl_socio_base_edu


# In[24]:


# single = dl.dict[dl.dict['field_type'] == 'Categorical (single)'].index.values.tolist()

# # pl[single_question][single_question].value_counts()
# single
from pheno_utils.questionnaires_handler import transform_answers
pl_socio = PhenoLoader('sociodemographics', preferred_language = 'coding')

df_codes = pl_socio.data_codings
df_codes.head()


# In[25]:


single = pl_socio.dict[pl_socio.dict['field_type'] == 'Categorical (single)'].index.values.tolist()

single_question = 'domestic_education_qualifications'
pl_socio[single_question][single_question].value_counts()


# In[26]:


print(single_question, 'mapping:', pl_socio.dict.loc[single_question]['data_coding'])


# In[27]:


print(df_codes["code_number"].value_counts())
df_codes[df_codes["code_number"] == "P100305"]


# In[28]:


pl_socio[single_question][single_question].value_counts().head()

tranformed_english = transform_answers(single_question, 
                                       pl_socio[single_question][single_question], 
                                       transform_from='coding', transform_to='english', 
                                       dict_df=pl_socio.dict, mapping_df=df_codes)


# In[29]:


from collections import Counter

# Flatten the list of lists into a single list
flattened = [item for sublist in tranformed_english.dropna() for item in sublist]

# Count the frequency of each answer
answer_counts = Counter(flattened)

# Display the counts
print(answer_counts)


# In[30]:


pl_socio_base_edu['domestic_education_qualifications'].str.split(",")

# select the max education level
pl_socio_base_edu['domestic_education_qualifications'] = pl_socio_base_edu['domestic_education_qualifications'].apply(lambda x: x[-1] if len(x) > 0 else None)

# select the max education level
pl_socio_base_edu1 = pl_socio_base_edu.groupby(['participant_id'], as_index=False)['domestic_education_qualifications'].max()

# Recode the educational level
pl_socio_base_edu1['edu_status'] = np.where(pl_socio_base_edu1['domestic_education_qualifications'] >= 2, "High", "Low")

display(pl_socio_base_edu1['edu_status'].value_counts())
# check the precentage
display(pl_socio_base_edu1['edu_status'].value_counts(normalize=True))


# In[31]:


age_sex_bmi_smoke_edu = pd.merge(age_sex_bmi_smoke, pl_socio_base_edu1, on = 'participant_id', how = 'left')
age_sex_bmi_smoke_edu


# ## Sleep

# In[32]:


[col for col in df_lifestyle_base.columns if "sleep" in col]

# sleep_hours_daily
df_lifestyle_base_sleep = df_lifestyle_base[['participant_id','sleep_hours_daily']].dropna()

print(df_lifestyle_base_sleep['participant_id'].nunique()) # 8445

# df_lifestyle_base_sleep[df_lifestyle_base_sleep.duplicated(subset='participant_id', keep=False)]
df_lifestyle_base_sleep1 = df_lifestyle_base_sleep.groupby(['participant_id'], as_index=False)['sleep_hours_daily'].mean()

df_lifestyle_base_sleep1.shape # 8445 obs

# merge with previous data
age_sex_bmi_smoke_edu_sleep = pd.merge(age_sex_bmi_smoke_edu, df_lifestyle_base_sleep1, on = "participant_id", how = 'left')
age_sex_bmi_smoke_edu_sleep['sleep_hours_daily'].describe().round(1)


# ## Physical activity

# In[33]:


Activities = [col for col in df_lifestyle_base.columns if "activity" in col]
cols = Activities[0:6]
cols


# In[34]:


df_lifestyle_base_activity = df_lifestyle_base[['participant_id'] + cols]
df_lifestyle_base_activity1 = df_lifestyle_base_activity.drop_duplicates(subset = ['participant_id'], keep = 'first')
display(df_lifestyle_base_activity1.head())

# fill in NA with 0
df_lifestyle_base_activity2 = df_lifestyle_base_activity1.fillna(0)
df_lifestyle_base_activity2.head()


# In[35]:


df_lifestyle_base_activity2['MET'] = df_lifestyle_base_activity2['activity_walking_10min_days_weekly']*df_lifestyle_base_activity2['activity_walking_minutes_daily']*3.3 + df_lifestyle_base_activity2['activity_moderate_days_weekly']*df_lifestyle_base_activity2['activity_moderate_minutes_daily']*4 + df_lifestyle_base_activity2['activity_vigorous_days_weekly']*df_lifestyle_base_activity2['activity_vigorous_minutes_daily']*8


# In[36]:


df_lifestyle_base_activity2['MET_hour'] = df_lifestyle_base_activity2['MET']/60


# In[37]:


age_sex_bmi_smoke_edu_sleep_physical = pd.merge(age_sex_bmi_smoke_edu_sleep, df_lifestyle_base_activity2, on = 'participant_id', how = 'left')
age_sex_bmi_smoke_edu_sleep_physical['MET_hour'].describe().round(1)


# In[38]:


age_sex_bmi_smoke_edu_sleep_physical.head()


# In[39]:


age_sex_bmi_smoke_edu_sleep_physical.columns


# ## Medication use

# In[40]:


medications = PhenoLoader('medications')
medications
medications.dfs['medications'].index.nlevels # 4 indexes

medications.dfs['medications'].shape


# In[41]:


# select on the columns with xs, only based on 00 visitm
medications_base = medications.dfs['medications'].xs("00_00_visit", level = "research_stage")
medications_base = medications_base.reset_index(level = "participant_id")
display(medications_base.head())

medications.dfs['medications'].xs("00_00_visit", level = "research_stage").index.unique(level = "participant_id") # 10269 


# In[42]:


medications_base.loc[:, 'atc5']


# In[43]:


# import re

# def extract_brackets(text):
#     pattern = r'\[(.*?)\]'
#     matches = re.findall(pattern, text)
#     return matches

# medications_base['atc5'].apply(lambda x: extract_brackets(str(x)))


# In[44]:


medications_base.dtypes


# In[45]:


# If they're numpy arrays, convert to tuples
medications_base['atc5'] = medications_base['atc5'].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)

# Then convert to category
medications_base['atc5'] = medications_base['atc5'].astype('category')


# In[46]:


medications_base["atc5"].value_counts()

# check the tuple with more than 2 elements
mask = medications_base["atc5"].apply(lambda x: len(x) >= 2 if isinstance (x, tuple) else False)
res_check = medications_base[mask]
res_check


# In[47]:


medications_base['atc5_new'] = medications_base['atc5'].str[0]


# In[48]:


medications_base


# In[49]:


medications_base["participant_id"].nunique()
medications_base[medications_base["participant_id"] == 4677768245]


# In[50]:


# filter rows where any elements in the atc5 tuple start with J01
mask = medications_base["atc5"].apply(
    lambda x: any(str(item).startswith("J01") for item in x) if isinstance(x, tuple) else str(x).startswith("J01")
)

antibotic_check = medications_base[mask]
antibotic_check


# In[56]:


antibotic_check["participant_id"].nunique()

# only select specific columns

antibotic_check = antibotic_check[["participant_id", "collection_date", "atc5"]].drop_duplicates(subset = ["participant_id"])
display(antibotic_check.head())
antibotic_check.shape


# In[51]:


# Read in the data for diet

pl = PhenoLoader('diet_logging', age_sex_dataset = None)

events_df_path = os.path.join(DATASETS_PATH, pl.dataset, 'diet_logging_events.parquet')
events_df = pd.read_parquet(events_df_path)
events_df

# select the baseline visit 
events_df_base = events_df[events_df.index.get_level_values('research_stage') == "00_00_visit"]

## remove those null rows in specific columns
events_df_base_noNA = events_df_base.dropna(axis = 0, subset = ["short_food_name", "product_name", "food_category"] , how = "all")
events_df_base_noNA1 = events_df_base_noNA.reset_index()
events_df_base_noNA1.shape


# In[52]:


print(events_df_base_noNA1["participant_id"].nunique()) # 9737 individuals

events_df_base_noNA1 = events_df_base_noNA1.drop_duplicates(subset = "participant_id")[["participant_id", "collection_date"]]
events_df_base_noNA1


# In[59]:


antibotic_check.head()

events_df_base_noNA1.rename(columns = {"collection_date": "diet_collection_date"}, inplace = True)

# merge with participant ID
antibotic_check_merge = pd.merge(antibotic_check, events_df_base_noNA1, on = "participant_id", how = "left")
antibotic_check_merge


# In[66]:


dl = PhenoLoader('gut_microbiome')
dl

dl_collecttime = dl['collection_date']
display(dl_collecttime.head())
dl['collection_date'].index.get_level_values("research_stage").value_counts()

gut_base = dl_collecttime[dl_collecttime.index.get_level_values("research_stage") == "00_00_visit"]
gut_base = gut_base.reset_index()[["participant_id", "collection_date"]]
gut_base.rename(columns = {"collection_date": "stool_collection_date"}, inplace = True)
gut_base.head()


# In[68]:


antibotic_check_merge_2 = pd.merge(antibotic_check_merge, gut_base, on = "participant_id", how = "left")
antibotic_check_merge_2.head(30)


# In[53]:


# display(antibotic_check["atc5"].value_counts())

vc_df = antibotic_check["atc5"].value_counts().reset_index()
vc_df.columns = ["atc_5", "count"]

print(vc_df["count"].sum())

# Show All rows (no truncation)
pd.set_option("display.max_rows", None)

vc_df[vc_df["count"] > 0]


# In[165]:


# medication pivotal table

# 1. with atc code
df_pivot_med_name = medications_base.pivot_table(index=['participant_id'],
                                      columns= 'atc5_new',
                                      aggfunc= 'size', 
                                      fill_value=0) # replace missing values with 0
display(df_pivot_med_name.head())
print(df_pivot_med_name.shape) # 6082 individuals with ATC code (the others without taking medication)

# 2. with medication information
df_pivot_med_name_medication = medications_base.pivot_table(index=['participant_id'],
                                      columns= 'medication',
                                      aggfunc= 'size', 
                                      fill_value=0) # replace missing values with 0
display(df_pivot_med_name_medication.head())
print(df_pivot_med_name_medication.shape) # 10269 individuals

# set the first column name as count
medication_counts = medications_base['medication'].value_counts().reset_index().rename(columns= {0:'count'})
medication_counts.head(20)


# In[166]:


df_pivot_med_name_medication.head()


# In[167]:


## read in the medicine mapping data
medication_mapping_path = os.path.join(DATASETS_PATH, medications.dataset, 'medication_registry_v12_2024.csv')

medication_codes = pd.read_csv(medication_mapping_path)
medication_codes.head(10)


# In[168]:


temp = medication_codes[medication_codes['atc5_updated'].str.contains(r'\bJ01', na = False)]
temp


# In[169]:


df_pivot_med_name.columns = df_pivot_med_name.columns.str.upper().str.strip()
df_pivot_med_name


# In[170]:


# medication_codes.loc[np.where(medication_codes['atc5_updated'].str.contains(r'\bJ01')), :]
temp.drop_duplicates(subset = ['atc5_updated']).to_csv("Antibiotic_list.csv", index = False)
antibio_drug = set(temp.drop_duplicates(subset = ['atc5_updated']).atc5_updated.unique())


# In[171]:


temp_antibio = [col for col in df_pivot_med_name.columns if any(drug in col for drug in antibio_drug)]


# In[172]:


# Vitamine start with A11
temp2 = medication_codes[medication_codes['atc5_updated'].str.contains(r'\b^A11', na = False)]

vitamin_drug = set(temp2.drop_duplicates(subset = ['atc5_updated']).atc5_updated.unique())
vitamin_drug


# In[173]:


Vitamine_code = ["A11AA03", "A11CC03", "A11CC05", "A11DA01", "A11GA01", "A11HA02", "A11HA03"]
col_1 = [col for col in df_pivot_med_name.columns if any(drug in col for drug in Vitamine_code)]
len(col_1) 


# In[174]:


# Hormone use 
temp_hormone = medication_codes[medication_codes['atc5_updated'].str.contains(r'\b^G03|^H01|^H02|^H03|^H04|^H05', na = False)]
temp_hormone = temp_hormone['atc5_upadated'].unique()
temp_hormone

col_2 = [col for col in df_pivot_med_name.columns if any(drug in col for drug in temp_hormone)]
len(col_2) # 30


# In[175]:


# Aspirin
temp_aspirin = medication_codes[medication_codes['atc5_updated'].str.contains(r'\b^B01|^N02', na = False)]
temp_aspirin = temp_aspirin['atc5_updated'].unique()
temp_aspirin = temp_aspirin[~np.isin(temp_aspirin, ['N02BE01;R01BA02;R06AB04', 'N02BE01;R01BA02;R05DA09', 'N02BE01;N06BC01;R05DA04'])]
temp_aspirin

# Extract aspirin
col_3 = [col for col in df_pivot_med_name.columns if any(drug in col for drug in temp_aspirin)]


# In[176]:


temp_PPI = medication_codes[medication_codes['atc5_updated'].str.contains(r'\b^A02BC', na = False)]
temp_PPI = temp_PPI['atc5_updated'].unique()
display(temp_PPI)

# Extract PPI
col_4 = [col for col in df_pivot_med_name.columns if any(drug in col for drug in temp_PPI)]
col_4


# In[177]:


# NSAID drug
temp_NSAID = medication_codes[medication_codes['atc5_updated'].str.contains(r'\b^M01|^N02', na = False)]
temp_NSAID = temp_NSAID['atc5_updated'].unique()

temp_NSAID = temp_NSAID[~np.isin(temp_NSAID, ['N02BE01;R01BA02;R06AB04', 'N02BE01;R01BA02;R05DA09', 'N02BE01;N06BC01;R05DA04'])]

# Extract NSAID
col_5 = [col for col in df_pivot_med_name.columns if any(drug in col for drug in temp_NSAID)]
col_5


# In[178]:


# sum
# 1. vitamin use
df_pivot_med_name['Vit_all'] = df_pivot_med_name[col_1].sum(axis = 1)
df_pivot_med_name['Vit_all'].value_counts()

# 2. Hormone use
df_pivot_med_name['Hormone_treat'] = df_pivot_med_name[col_2].sum(axis = 1)
df_pivot_med_name['Hormone_treat'].value_counts()

# 3. Aspirin use
df_pivot_med_name['Aspirin_treat'] = df_pivot_med_name[col_3].sum(axis = 1)
print(df_pivot_med_name['Aspirin_treat'].value_counts())

# 4. PPI use
df_pivot_med_name['PPI_treat'] = df_pivot_med_name[col_4].sum(axis = 1)
df_pivot_med_name['PPI_treat'].value_counts()

# 5. NSAID use
df_pivot_med_name['NSAID_treat'] = df_pivot_med_name[col_5].sum(axis = 1)
df_pivot_med_name['NSAID_treat'].value_counts()


# In[179]:


# 6. antibody use
df_pivot_med_name['Antibio_treat'] = df_pivot_med_name[temp_antibio].sum(axis = 1)
df_pivot_med_name['Antibio_treat'].value_counts()


# In[180]:


# if do it in a for loop
# med_names = ['Vit_all', 'Hormone_treat', 'Aspirin_treat', 'PPI_treat', 'NSAID_treat']

# # Process each medication using a for loop
# for i, med_name in enumerate(med_names, start=1):
#     df_pivot_med_name[med_name] = df_pivot_med_name[f'col_{i}'].sum(axis=1)


# ## Binary medication use

# In[181]:


# Define the medication mappings
medication_mappings = {
    'Vit_use': 'Vit_all',
    'Hormone_use': 'Hormone_treat',
    'NSAID_use': 'NSAID_treat',
    'Aspirin_use': 'Aspirin_treat',
    'PPI_use' : "PPI_treat",
    # "NSAID_use" : "NSAID_treat",
    "Antibio_use" : "Antibio_treat"
}

# Create a function to handle the condition
def create_medication_flags(df):
    for new_col, source_col in medication_mappings.items():
        df[new_col] = np.where(df[source_col] >= 1, "Use", "not use")
    return df

df_pivot_med_name2 = create_medication_flags(df = df_pivot_med_name)
df_pivot_med_name2

# # assign number
# medication_base['Vit_all'] = np.where(medication_base['Vit_all'] >= 1, "Use", "not use")
# medication_base['Hormone_treat'] = np.where(medication_base['Hormone_treat'] >= 1, "Use", "not use")
# medication_base['NSAID_treat'] = np.where(medication_base['NSAID_treat'] >= 1, "Use", "not use")
# medication_base['Aspirin_treat'] = np.where(medication_base['Aspirin'] >= 1, "Use", "not use")


# In[182]:


df_pivot_med_name2 = df_pivot_med_name2.reset_index()
df_pivot_med_name2


# In[183]:


df_pivot_med_name_medication1 = pd.DataFrame(df_pivot_med_name_medication.reset_index()["participant_id"])
df_pivot_med_name_medication1


# In[184]:


df_pivot_med_name3 = pd.merge(df_pivot_med_name_medication1, df_pivot_med_name2, on = "participant_id", how = "outer")
df_pivot_med_name3.shape


# In[185]:


df_pivot_med_name3.head()


# In[186]:


df_pivot_med_name3["Vit_use"].value_counts()


# In[187]:


df_pivot_med_name3 = df_pivot_med_name3.fillna("not use")


# In[188]:


age_sex_bmi_smoke_edu_sleep_physical_medication = pd.merge(age_sex_bmi_smoke_edu_sleep_physical, df_pivot_med_name3, on = 'participant_id', how = 'left')
age_sex_bmi_smoke_edu_sleep_physical_medication['Vit_use'].value_counts()


# ## Calculate medication frequency

# In[189]:


# counts
# df_pivot_med_name2 = df_pivot_med_name2.reset_index()

# # select columns
# select_col = ['participant_id', 'Vit_use', 'Hormone_use', 'NSAID_use', 'Aspirin_use', 'PPI_use', "NSAID_use", "Antibio_use"]
# df_pivot_med_name2 = df_pivot_med_name2[select_col]

# age_sex_bmi_smoke_edu_sleep_physical_medication = pd.merge(age_sex_bmi_smoke_edu_sleep_physical, df_pivot_med_name2, on = 'participant_id', how = 'left')

# print(age_sex_bmi_smoke_edu_sleep_physical_medication['Vit_use'].value_counts())
# print(age_sex_bmi_smoke_edu_sleep_physical_medical['Aspirin_treat'].value_counts())
# print(age_sex_bmi_smoke_edu_sleep_physical_medical['NSAID_treat'].value_counts())
# print(age_sex_bmi_smoke_edu_sleep_physical_medical['Hormone_treat'].value_counts())


# ## Family history of disease

# In[190]:


p_his = PhenoLoader('family_history')
p_his

p_his1 = p_his.dfs['initial_medical']
p_his1.shape


# In[191]:


p_his1

p_his1_base = p_his1.xs("00_00_visit", level = "research_stage")
p_his1_base = p_his1_base.reset_index(level = "participant_id")
display(p_his1_base.head())

p_his1_base['participant_id'].nunique() # 6680 unique observations


# In[192]:


p_his1_base = p_his1_base.drop_duplicates(subset = "participant_id", keep = "last")
p_his1_base.shape


# In[193]:


p_his1_base.columns


# In[194]:


p_his1_base['type_2_diabetes_family_number'].value_counts(dropna = False)


# In[195]:


p_his1_base['stroke_family_number'].value_counts(dropna = False)


# In[196]:


# can also use map() or apply() function to perform
# p_his1_base['CVD_f_history'] = p_his1_base['stroke_family_number'].apply(lambda x: 'yes' if x >= 1 else 'no')

p_his1_base['CVD_f_history'] = np.where(
    (p_his1_base['stroke_family_number'].fillna(0) >= 1),
    "yes",
    "no"
)
p_his1_base['CVD_f_history'].value_counts(normalize = True)

# the same for the T2DM
p_his1_base['T2D_f_history'] = np.where(
    (p_his1_base['type_2_diabetes_family_number'].fillna(0) >= 1),
    "yes",
    "no"
)
p_his1_base['T2D_f_history'].value_counts(normalize = True)


# In[197]:


# merge with other information
age_sex_bmi_smoke_edu_sleep_physical_medication
# p_his1_base[['participant_id', 'CVD_f_history', 'T2D_f_history']]

lifestyle_factor_all = pd.merge(age_sex_bmi_smoke_edu_sleep_physical_medication, p_his1_base[['participant_id', 'CVD_f_history', 'T2D_f_history']], 
                                on = "participant_id", how = "left")
lifestyle_factor_all.shape


# In[198]:


lifestyle_factor_all.to_csv("lifestyle_factor_all.csv", index = False)


# ## To see the baseline disease states
# - from Folder called ruifang

# In[199]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

from pheno_utils import PhenoLoader
from pheno_utils.config import (
    DATASETS_PATH, 
    )


# In[200]:


directory1 = '/home/ec2-user/studies/ruifang/Baseline Conditions.csv'
Base_cond = pd.read_csv(directory1, index_col = 0)
Base_cond


# In[201]:


Base_cond['research_stage'].value_counts()
Base_cond = Base_cond[Base_cond['research_stage'] == 'baseline']
Base_cond


# In[202]:


Base_cond['Consolidated name'].value_counts()


# In[203]:


Base_cond['Participant_ID'] = Base_cond['RegistrationCode'].str.split("_").str[1]
# Manually define the column order
Base_cond = Base_cond[['Participant_ID'] + [col for col in Base_cond.columns if col != 'Participant_ID']]
Base_cond = Base_cond.drop('RegistrationCode', axis = 1)
Base_cond


# In[204]:


# read in all the lifestyle factor
lifestyle_factor_all = pd.read_csv("lifestyle_factor_all.csv")
lifestyle_factor_all


# In[205]:


Base_cond1 = Base_cond.rename({'Participant_ID' : "participant_id"}, axis = 1)
Base_cond1


# In[206]:


Base_cond1['participant_id'] = pd.to_numeric(Base_cond1['participant_id'])


# In[207]:


Base_cond1['participant_id'].nunique()


# In[208]:


Base_cond1['Consolidated name'].unique()


# In[209]:


Base_cond1[Base_cond1['participant_id'].duplicated()]


# In[210]:


Base_cond1[Base_cond1['participant_id'].duplicated()]['Consolidated name'].unique()


# In[211]:


# change from long to wide format
Base_cond1_wide = Base_cond1.pivot(index = ["participant_id", "medical_condition"], 
                                   columns = "Group", 
                                   values = "Consolidated name")
Base_cond1_wide


# In[212]:


Base_cond1_wide.head(30)


# In[213]:


# Group by participant_id and aggregate the conditions
Base_cond1_wide_combined = Base_cond1_wide.reset_index().groupby('participant_id').agg({
    'medical_condition': lambda x: '|'.join(x.dropna()),
    'Cardiovascular': lambda x: '|'.join(x.dropna()),
    'Metabolic': lambda x: '|'.join(x.dropna())
}).reset_index()
Base_cond1_wide_combined


# In[214]:


Base_cond1_wide_combined['Metabolic'].value_counts()


# In[215]:


Base_cond1_wide_combined['Cardiovascular'].value_counts()


# In[ ]:


lifestyle_factor_all_disease = pd.merge(lifestyle_factor_all, Base_cond1_wide_combined, on = "participant_id", how = "left")
lifestyle_factor_all_disease


# In[217]:


lifestyle_factor_all_disease.to_csv("lifestyle_factor_all_disease.csv", index = False)

