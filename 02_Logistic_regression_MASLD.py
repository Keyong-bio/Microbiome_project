
# ============================================================
Source: 01_Dyslipidemia.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## LUMC, Clinical Epidemiology Department
# - Total cholestrol TC
# - TG
# - LDL
# - HDL
# - Using Lipid-lowering drugs

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pheno_utils import PhenoLoader


# In[2]:


pl_blood = PhenoLoader('blood_tests')
pl_blood


# In[3]:


pl_blood[pl_blood.fields]


# In[4]:


# Assign the reseach stage
from pheno_utils.basic_analysis import assign_nearest_research_stage


# In[5]:


pop_df = PhenoLoader('events').dfs['events']
blood_w_rs = assign_nearest_research_stage(pl_blood[pl_blood.fields], pop_df)
blood_w_rs


# In[6]:


blood_w_rs.index.get_level_values("research_stage").value_counts()


# In[7]:


# Extract the baseline visit
blood_w_rs_baseline = blood_w_rs.loc[blood_w_rs.index.get_level_values("research_stage") == "00_00_visit",:]
blood_w_rs_baseline


# In[8]:


blood_w_rs_baseline.columns


# In[9]:


# select cols 
cols_select = ["bt__hdl_cholesterol_float_value", "bt__total_cholesterol_float_value", 
               "bt__triglycerides_float_value", 
               "bt__ldl_cholesterol_float_value"]
blood_w_rs_baseline_Select = blood_w_rs_baseline[cols_select]


# In[10]:


# change between  mg/dL -> mmol/L 
# TC, LDL and HDL divide the value by 38.67, TG divide the value by 88.57
blood_w_rs_baseline_Select


# In[11]:


division_factors = pd.Series([38.67, 38.67, 88.57, 38.67], index=['bt__hdl_cholesterol_float_value', 
                                                      'bt__total_cholesterol_float_value', 
                                                      'bt__triglycerides_float_value', 
                                                      'bt__ldl_cholesterol_float_value'])
blood_w_rs_baseline_Select = blood_w_rs_baseline_Select.div(division_factors, axis=1)
blood_w_rs_baseline_Select


# In[12]:


(blood_w_rs_baseline_Select["bt__total_cholesterol_float_value"] >= 6.2).value_counts()


# In[13]:


(blood_w_rs_baseline_Select["bt__hdl_cholesterol_float_value"] < 1).value_counts()


# In[14]:


(blood_w_rs_baseline_Select["bt__triglycerides_float_value"] >= 2.3).value_counts()


# In[15]:


(blood_w_rs_baseline_Select["bt__ldl_cholesterol_float_value"] >= 4.1).value_counts()


# In[16]:


Conditions_Lipids = (blood_w_rs_baseline_Select["bt__total_cholesterol_float_value"] >= 6.2) | (blood_w_rs_baseline_Select["bt__hdl_cholesterol_float_value"] < 1) | (blood_w_rs_baseline_Select["bt__triglycerides_float_value"] >= 2.3) | (blood_w_rs_baseline_Select["bt__ldl_cholesterol_float_value"] >= 4.1) 
Conditions_Lipids.value_counts()


# In[17]:


Conditions_Lipids = (blood_w_rs_baseline_Select["bt__total_cholesterol_float_value"] >= 6.2) & (blood_w_rs_baseline_Select["bt__hdl_cholesterol_float_value"] < 1) & (blood_w_rs_baseline_Select["bt__triglycerides_float_value"] >= 2.3) & (blood_w_rs_baseline_Select["bt__ldl_cholesterol_float_value"] >= 4.1) 
Conditions_Lipids.value_counts()


# In[18]:


blood_w_rs_baseline_Select.loc[:, "lipid_dignosis"] = Conditions_Lipids.map({True : "Yes", False : "No"})


# ## Considering Drug Intake

# In[40]:


pl_atc = PhenoLoader('medications')

df_atc = pl_atc[pl_atc.fields].reset_index()

# Select the baseline state
df_atc_base = df_atc[df_atc["research_stage"] == "00_00_visit"]


# In[48]:


df_atc_base["participant_id"].nunique()


# In[73]:


# 1000942861
df_atc_base[df_atc_base["participant_id"] == 1002087123]


# In[74]:


df_atc_base["participant_id"].nunique()


# In[81]:


df_atc_base.info()


# In[80]:


df_atc_base["atc4"].value_counts()


# In[49]:


df_atc_base.head()


# In[45]:


df_atc.groupby("research_stage")["participant_id"].nunique()


# In[ ]:


# filter the medication code
import pandas as pd

def filter_on_value(df, column, value):
    return df[(df[column].apply(lambda x: value in x))]

df_lipids_drug = filter_on_value(df_atc_base, 'atc3', 'C10')

display(df_lipids_drug.head())


# In[46]:


df_lipids_drug.shape


# In[20]:


# Using Omega_3 intake 
df_lipids_drug[df_lipids_drug["medication"] == "Omega 3"].shape


# ## Pivotal the dataframe

# In[69]:


def pivot_atc_api(df, pivot_column, index_columns = ['participant_id', 'cohort', 'research_stage']):
    
    # Explode the specified column to separate rows
    exploded_df = df.explode(pivot_column)
    
    # Create a True value for each row to indicate the presence of the ATC/API code
    exploded_df['value'] = True
    
    # Pivot the DataFrame to get True/False for the presence of each ATC/API code
    pivot_df = exploded_df.pivot_table(index = index_columns, 
                                       columns = pivot_column, 
                                       values = 'value', fill_value = False)
    pivot_df = pivot_df.astype(bool)
    
    return pivot_df

pivot_column = 'atc4'  # Change this to 'atc3', 'atc4', 'atc5', or 'api' as needed
pivot_df_atc = pivot_atc_api(df_atc, pivot_column)
pivot_df_atc.head()

# C10
atc_lipid = pivot_df_atc.filter(regex = "C10").reset_index()

atc_lipid_base = atc_lipid.query('research_stage == "00_00_visit"')
atc_lipid_base


# In[52]:


atc_lipid["research_stage"].value_counts()


# In[23]:


cols_to_sum = atc_lipid_base.columns[3:atc_lipid_base.shape[1]-1]
atc_lipid_base.loc[:, "Lipid_intake"] = atc_lipid_base[cols_to_sum].sum(axis = 1)
# atc_lipid_base = atc_lipid_base.drop(columns = "T2D_intake")
atc_lipid_base


# In[24]:


# Icosapent ethyl is classified in C10AX06 - omega-3-triglycerides incl. other esters and acids.
# Sulodexide is classified in B01AB.
atc_lipid_base["C10AX"].value_counts()


# In[25]:


condition_lipids = atc_lipid_base["Lipid_intake"] >= 1
condition_lipids.value_counts()


# In[26]:


atc_lipid_base.loc[:, "Lipid_drug"] = condition_lipids.map({True: "Yes", False: "No"})
atc_lipid_base


# In[27]:


blood_w_rs_baseline_Select.head()


# In[28]:


blood_w_rs_baseline_Select.index.get_level_values("participant_id").nunique()


# In[29]:


blood_w_rs_baseline_Select.columns


# In[30]:


atc_lipid_base["participant_id"].isin(blood_w_rs_baseline_Select.index.get_level_values("participant_id")).value_counts()


# In[31]:


blood_w_rs_baseline_Select.index.get_level_values("participant_id").isin(atc_lipid_base["participant_id"]).sum()


# In[32]:


# merge two dataframe
blood_w_rs_baseline_Select = blood_w_rs_baseline_Select.reset_index()
blood_w_rs_baseline_Select[["participant_id", "lipid_dignosis"]]


# In[33]:


atc_lipid_base


# In[34]:


dylipidemia = pd.merge(atc_lipid_base[["participant_id", "Lipid_drug"]], 
                       blood_w_rs_baseline_Select[["participant_id", "lipid_dignosis"]],
                       on = "participant_id", how = "outer")
dylipidemia


# In[35]:


Condition_dylipidemia = (dylipidemia["Lipid_drug"] == "Yes") | (dylipidemia["lipid_dignosis"] == "Yes")
Condition_dylipidemia.value_counts()


# In[36]:


pd.crosstab(dylipidemia["Lipid_drug"], dylipidemia["lipid_dignosis"], dropna = False)


# In[37]:


dylipidemia.loc[:, "dylipidemia"] = Condition_dylipidemia.map({True: "Yes", False: "No"})


# In[38]:


dylipidemia["dylipidemia"].value_counts()


# In[39]:


dylipidemia.to_csv("Dylipidemia.csv", index = False)


# ## Read in the Diets information

# In[61]:


import os
os.listdir()


# In[62]:


DPs = pd.read_csv("../DPs_Diet_patterns/DPs_Final_score_outlieradj.csv")
DPs


# In[64]:


DPs["participant_id"].isin(df_atc_base["participant_id"].unique()).sum()


# In[66]:


DPs["participant_id"].isin(dylipidemia["participant_id"].unique()).sum()


# In[67]:


6036 + 1777 + 2174



# ============================================================
Source: 02_Logistic_regression_model.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# # Logistic regression model
# ### Keyong Deng, Clinical Epidemiology,LUMC
# Fit a logistic regression model containing additive effects for all covariates.  
# The `GLM` function fits many types of generalized linear models (GLMs).  
# Choosing the `Binomial` family makes it a logistic regression.

# In[1]:


import pandas as pd


# In[2]:


Dyslipidemia = pd.read_csv("Dylipidemia.csv")
Dyslipidemia.head()


# ### Read the other dataset

# In[3]:


# Diet_pattern scores

DPs = pd.read_csv("../DPs_Diet_patterns/DPs_Final_score_outlieradj.csv")
DPs.head()


# ## Define function to scale the variables

# In[4]:


def std_columns(df, start_col = 2, end_col = 10, lastfix = "_scaled"):
    result_df = df.copy()

    cols_select = df.columns[start_col:end_col+1]

    for cols in cols_select:
        mean = df[cols].mean()
        std = df[cols].std()
        result_df[f"{cols}{lastfix}"] = (df[cols] - mean)/ std

    return result_df

DPs_scale = std_columns(DPs, start_col=1, end_col=5)
DPs_scale.to_csv("DPs_scale.csv", index = False)


# ## Read in the Alcohol intake data

# In[5]:


Nutrients = pd.read_csv("/home/ec2-user/studies/ruifang/Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
Nutrients_2.shape


# ## lifestyle information

# In[6]:


lifestyles = pd.read_csv("../Lifestyle factors/lifestyle_factor_all_disease.csv")

lifestyles = pd.merge(lifestyles, Nutrients_2, on = "participant_id", how = "left")
lifestyles.head()


# In[7]:


merge_df = pd.merge(DPs_scale, lifestyles, on = "participant_id", how = "left")
merge_df


# In[8]:


merge_df_2 = pd.merge(merge_df, Dyslipidemia[["participant_id", "dylipidemia"]],
                     on = "participant_id", how = "left")
merge_df_2


# In[9]:


import statsmodels.api as sm
# model1 = sm.GLM.from_formula("low ~ age + smoke + race + lwt + ptl + ht + ui + ftv", family=sm.families.Binomial(), data=data)
# result1 = model1.fit()
# print(result1.summary())


# ### A function to run the logistic regression model 

# In[10]:


def multiple_glm_model (data, dependent_var, covariate_sets, family = sm.families.Binomial()):
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


# In[11]:


merge_df_2.columns[merge_df_2.columns.str.contains("sex")]
print(merge_df_2.sex_x.value_counts()) # select this variable
print(merge_df_2.sex_y.value_counts())
merge_df_2.sex.value_counts()

# merge_df_2.columns[merge_df_2.columns.str.contains("bmi")]


# In[12]:


# merge_df_2.dropna(subset = ["sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"])
# merge_df_2.dylipidemia.dtypes


# In[13]:


# merge_df_2.shape # 9617
merge_df_2["dylipidemia"].value_counts()


# In[14]:


# merge_df_2_complete = merge_df_2.dropna(subset = ["dylipidemia"])
# print(merge_df_2_complete.shape) # 6859
merge_df_2["dylipidemia"] = merge_df_2["dylipidemia"].replace({"Yes": 1, "No": 0})


# In[15]:


# read in the GGM data
CGM = pd.read_csv("../CGM_Data/Data_CGM_base.csv")
CGM.head()


# In[16]:


merge_df_CGM = pd.merge(merge_df, CGM[["participant_id", "T2DM_Diagnose"]],
                     on = "participant_id", how = "left")

merge_df_CGM["T2DM_Diagnose"] = merge_df_CGM["T2DM_Diagnose"].replace({"Yes": 1, "No":0})
merge_df_CGM["T2DM_Diagnose"].value_counts(dropna = False)


# ## Read in the hypertension data

# In[17]:


df_hypertension = pd.read_csv("../Blood_pressure/Hypertension_baseline.csv")

merge_df_HP = pd.merge(merge_df, df_hypertension[["participant_id", "hypertension_diagnosed"]],
                     on = "participant_id", how = "left")

merge_df_HP["hypertension_diagnosed"] = merge_df_HP["hypertension_diagnosed"].replace({"Yes":1, "No": 0})
merge_df_HP["hypertension_diagnosed"].value_counts(dropna = False)


# # Read in the NAFLD data
# 

# In[18]:


import pandas as pd

df_NAFLD = pd.read_csv("../Medication_status/Mediaction_baseline.csv")
merge_df_NAFLD = pd.merge(merge_df, df_NAFLD[["participant_id", "NAFLD_baseline_diagnosis"]],
                          on = "participant_id", how = "left")

merge_df_NAFLD["NAFLD_baseline_diagnosis"] = merge_df_NAFLD["NAFLD_baseline_diagnosis"].replace({"Yes":1, "No": 0})
merge_df_NAFLD["NAFLD_baseline_diagnosis"].value_counts(dropna = False)


# # Read in another NAFLD data

# In[19]:


df_MAFLD = pd.read_csv("../MAFLD_baseline.csv")

merge_df_MAFLD = pd.merge(merge_df, df_MAFLD[["participant_id", "mafld_diagnosed"]],
                          on = "participant_id", how = "left")

merge_df_MAFLD["MAFLD_baseline_diagnosis"] = merge_df_MAFLD["mafld_diagnosed"].replace({"Yes":1, "No": 0})
merge_df_MAFLD["MAFLD_baseline_diagnosis"].value_counts(dropna = False)


# # Define the Cardiometabolic disease

# In[57]:


merge_df_2 # combine with T2DM
merge_df_CMD = pd.merge(merge_df_2, CGM[["participant_id", "T2DM_Diagnose"]],
                     on = "participant_id", how = "left")
merge_df_CMD
merge_df_CMD["T2DM_Diagnose"] = merge_df_CMD["T2DM_Diagnose"].replace({"Yes": 1, "No":0})


# In[58]:


print(merge_df_CMD["T2DM_Diagnose"].value_counts())
merge_df_CMD["dylipidemia"].value_counts()


# In[59]:


# to calculate both the dylipidemia and T2DM_Diagnose
# conditions = (merge_df_CMD["dylipidemia"] == 1) | (merge_df_CMD["T2DM_Diagnose"] == 1)
# has_na = merge_df_CMD["dylipidemia"].isna() | merge_df_CMD["T2DM_Diagnose"].isna()
# conditions.value_counts() # 1715 cases 
# merge_df_CMD.loc[:, "CMD_baseline_diagnosis"] = np.where(has_na, np.nan, conditions.map({True:1, False:0}))
# merge_df_CMD

def determine_value(var1, var2):
    if var1 == 1 or var2 == 1:
        return 1
    elif var1 == 0 and var2 == 0:
        return 0
    elif (var1 == 1 and pd.isna(var2)) or (pd.isna(var1) and var2 == 1):
        return 1
    else:
        return np.nan

# apply function row-wise
merge_df_CMD["CMD_baseline_diagnosis"] = merge_df_CMD.apply(
    lambda row: determine_value(row["dylipidemia"], row["T2DM_Diagnose"]), 
    axis=1
)
merge_df_CMD["CMD_baseline_diagnosis"].value_counts()


# In[61]:


merge_df_CMD["CMD_baseline_diagnosis"].value_counts(dropna = False)


# ## The same function for three outcomes

# In[20]:


# change the level of factor variables
# merge_df_2_complete["dylipidemia"]= pd.Categorical(merge_df_2_complete['dylipidemia'], 
#                             categories=['0', '1'],
#                             ordered=True)


# In[21]:


covariate_sets_test_AHEI = {"model1": ["AHEI_2010_score_eadj_scaled"],
                      
                       "model2": ["AHEI_2010_score_eadj_scaled", "age", 
                             "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"],
                       
                       "model3": ["AHEI_2010_score_eadj_scaled", "age", "bmi",
                             "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"],
                      
                       "model4": ["AHEI_2010_score_eadj_scaled", "age",
                                 "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use", 
                                  'NSAID_use', 'CVD_f_history', 'T2D_f_history']}


# In[22]:


## Define a function to modify the element in each list and return a new dictionary
def modify_dict(original, index, new_value, new_element = None):
    new_dict = {}
    for key, value_list in original.items():
        new_list = value_list.copy()
        # change the elements in the list by new value
        new_list[index] = new_value
        
        if key not in ["model1"]:
            if new_element is not None:
                new_list.append(new_element)
                
        new_dict[key] = new_list
    return new_dict

covariate_sets_test_rDII = modify_dict(covariate_sets_test_AHEI, 0, "rDII_score_eadj_scaled")
covariate_sets_test_AMED = modify_dict(covariate_sets_test_AHEI, 0, "AMED_score_eadj_scaled")
covariate_sets_test_rEDIH = modify_dict(covariate_sets_test_AHEI, 0, "rEDIH_score_all_eadj_scaled")
covariate_sets_test_hPDI = modify_dict(covariate_sets_test_AHEI, 0, "hPDI_score_eadj_scaled", new_element= "alcohol_g")


# ## Run the logistic model

# In[23]:


def logist_model_df(data, dependent_var):
    dfs = []
    for covariates in [covariate_sets_test_AHEI, 
                       covariate_sets_test_rDII, 
                       covariate_sets_test_AMED, 
                       covariate_sets_test_rEDIH,
                  covariate_sets_test_hPDI]:
        models_test = multiple_glm_model(data = data, 
                                         dependent_var = dependent_var, 
                                         covariate_sets = covariates)
        Exposure = covariates.get("model1", [])[0]
        
        # Also combine the other Dataframes: model1, model2, model3...

        for model in ["model1", "model2", "model3", "model4"]:
            coef = models_test[model].params[Exposure]
            pvalue = models_test[model].pvalues[Exposure]
            conf_int = models_test[model].conf_int(alpha = 0.05).loc[Exposure]
            lower_ci, upper_ci = conf_int
            
           # Create a new Pandas Dataframe
            result_all = pd.DataFrame({
                                "Model": [model],
                                "Exposure" : [Exposure],
                                "Outcome" : [dependent_var],
                                "Coef": [coef],
                                "pval" : [pvalue],
                                "lower_ci" : [lower_ci],
                                "upper_ci" : [upper_ci],
                                "OR": [np.exp(coef)],
                                "lower_OR_ci" : [np.exp(lower_ci)],
                                "upper_OR_ci" : [np.exp(upper_ci)]
                            })
            # dfs.append(result_all)
            dfs = dfs + [result_all]
    return dfs


# In[25]:


merge_df_NAFLD["NAFLD_baseline_diagnosis"].value_counts()


# In[26]:


import numpy as np
# for Dyslipidemia
results_dyslipidemia = logist_model_df(data=merge_df_2, dependent_var="dylipidemia")
results_hypertension = logist_model_df(data=merge_df_HP, dependent_var="hypertension_diagnosed") # merge_df_HP["hypertension_diagnosed"] 
results_diabetes = logist_model_df(data=merge_df_CGM, dependent_var="T2DM_Diagnose")


# In[62]:


results_CMD = logist_model_df(data=merge_df_CMD, dependent_var="CMD_baseline_diagnosis")
results_CMD_all = pd.concat(results_CMD)
results_CMD_all


# In[66]:


results_CMD_all.round(2)


# In[27]:


# merge_df_NAFLD = merge_df_NAFLD.dropna(subset = "hypertension_diagnosed")
results_NAFLD = logist_model_df(data=merge_df_NAFLD, dependent_var="NAFLD_baseline_diagnosis")
results_MAFLD = logist_model_df(data=merge_df_MAFLD, dependent_var="MAFLD_baseline_diagnosis")


# In[28]:


# change the decimal to three

result1 = pd.concat(results_dyslipidemia)
result2 = pd.concat(results_diabetes)
result3 = pd.concat(results_hypertension)
result4 = pd.concat(results_NAFLD)
result5 = pd.concat(results_MAFLD)

results_all = pd.concat([result1, result2, result3, result4, result5])

results_all['OR'] = results_all['OR'].apply(lambda x: float("{:.3f}".format(x)))
results_all['lower_OR_ci'] = results_all['lower_OR_ci'].apply(lambda x: float("{:.3f}".format(x)))
results_all['upper_OR_ci'] = results_all['upper_OR_ci'].apply(lambda x: float("{:.3f}".format(x)))

results_all["Outcome"] = results_all["Outcome"].replace({
    "dylipidemia" : "Dyslipidemia",
    "T2DM_Diagnose" : "Type 2 Diabetes",
    "hypertension_diagnosed" : "Hypertension",
    "NAFLD_baseline_diagnosis" : "NAFLD",
    "MAFLD_baseline_diagnosis" : "MAFLD"
})


# In[29]:


# Capitalized the first column
results_all["Model"] = results_all["Model"].str.capitalize()
results_all.head()
# results_all.to_csv("Results_for_all.csv", index = False)


# ## Plot the results

# In[30]:


results_all


# In[31]:


enumerate(results_all.groupby("Outcome"))


# In[32]:


results_all["Exposure"].value_counts()


# In[33]:


results_all["Exposure"] = results_all["Exposure"].replace({
    "AHEI_2010_score_eadj_scaled": "AHEI",
    "rDII_score_eadj_scaled" : "rDII",
    "AMED_score_eadj_scaled" : "AMED",
    "rEDIH_score_all_eadj_scaled" : "rEDIH",
    "hPDI_score_eadj_scaled" : "hPDI"
})


# In[34]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl


# In[35]:


fig, axes = plt.subplots(2, 3, figsize = (12,6))
axes_flat = axes.flatten()  # Flatten the 2D array to make indexing easier
# colormap = cm["tab20"]

# my_colors = ["#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
# my_cmap = mlt.colors.ListedColormap(my_colors, name="my_cmap")
my_cmap = mpl.colormaps['Paired'].resampled(4)

for i, (group_name, group_data) in enumerate(results_all.groupby("Outcome")):
    # choose the suplot
    # row, col = i // 3, i % 3  # For positioning in a 2×3 grid
    if i < 5:
        ax = axes_flat[i]
    #ax = axes[row,col]
        all_exposure = group_data["Exposure"].unique()
    # print(all_exposure)

    # name the subplot
        ax.set_title(f"{group_name}", fontsize=12)
    
        exposure_position = {exp: idx for idx, exp in enumerate(all_exposure)}

        models = group_data["Model"].unique()
    
        colors = {model: my_cmap(i/len(models)) for i, model in enumerate(models)}
    # print(colors)

    # group by model to allow for dodging
        for j, (model_name, model_data) in enumerate(group_data.groupby("Model")):

            dodge_positions = j * 0.2 - 0.3
            model_color = colors[model_name]

        # convert string exposure to numeric positions
            x_positions = np.array([exposure_position[exp] for exp in model_data["Exposure"]]) + dodge_positions
        # print(x_positions)

        # plot each point with error bars
            ax.errorbar(
                x = x_positions,
                y = model_data["OR"],
                yerr = [model_data["OR"] - model_data["lower_OR_ci"],
                   model_data["upper_OR_ci"]- model_data["OR"]],
                fmt = "o",
                capsize = 4,
                capthick = 1.2,
                linewidth = 1.2,
                color = model_color
        )
            ax.scatter(x_positions, model_data["OR"], color = model_color, label = model_name)
        # add reference line
            ax.axhline(y = 1, linestyle = "--", color = "grey")
    
    # set the ylim
       
        ax.set_ylim(bottom = 0.6, top = 1.4)

    # set new label for the exposure
        ax.set_xticks(np.arange(len(all_exposure)))
        ax.set_xticklabels(all_exposure)
        ax.set_ylabel("Odds Ratio with 95%CI")

# add legend
# ax.legend()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')

plt.tight_layout()
# Make room for the legend
plt.subplots_adjust(bottom = 0.2, wspace = 0.3)  # wspace means width space between subplots
plt.savefig("Logistic_regression_DPs.pdf")
plt.show()


# In[36]:


##### Another Diet index
# import numpy as np
# models_test = multiple_glm_model(data = merge_df_2, dependent_var = "dylipidemia", covariate_sets = covariate_sets_test_AMED)

# # Also combine the other Dataframes: model1, model2, model3...
# dfs = []
# for model in ["model1", "model2", "model3", "model4"]:
#     coef = models_test[model].params["AMED_score_eadj_scaled"]
#     pvalue = models_test[model].pvalues["AMED_score_eadj_scaled"]
#     conf_int = models_test[model].conf_int(alpha = 0.05).loc["AMED_score_eadj_scaled"]
#     lower_ci, upper_ci = conf_int
    
#     # Create a new Pandas Dataframe
#     result_all = pd.DataFrame({
#         "Model": [model],
#         "Exposure" : ["AMED_score_eadj_scaled"],
#         "Coef": [coef],
#         "pval" : [pvalue],
#         "lower_ci" : [lower_ci],
#         "upper_ci" : [upper_ci],
#         "OR": [np.exp(coef)],
#         "lower_OR_ci" : [np.exp(lower_ci)],
#         "upper_OR_ci" : [np.exp(upper_ci)]
#     })
#     dfs.append(result_all)

# result_all = pd.concat(dfs, ignore_index=True)
# result_all


# In[37]:


# res_m2 = models_test["model2"].summary().tables[1]
# res_m2.loc["AHEI_2010_score_eadj_scaled", ["coef", "P>|z|"]]


# In[38]:


# display(models_test["model2"].summary2().tables[1].loc["AHEI_2010_score_eadj_scaled"])
# print(np.exp(models_test["model2"].summary2().tables[1].loc["AHEI_2010_score_eadj_scaled"][0]))
# np.exp(models_test["model2"].conf_int().loc["AHEI_2010_score_eadj_scaled"])


# In[39]:


# def summarize_models(models, variable_interest = None, alpha = 0.05):
#     summary = pd.DataFrame()

#     # Basic model status
#     for model_name, result in models.items():
#         model_stats = {
#             "Model": model_name,
#             "AIC": result.aic,
#             "BIC": result.bic,
#             "N": result.nobs
#         }

#         if variable_interest:
#             for var in variable_interest:
#                 var_params = [param for param in result.param.index if var in param]

#                 for param in var_params:
#                     coef = result.param.get(param, None)
#                     if coef is not None:
#                         # add coefficient
#                         model_stats[f"{param}_coef"] = coef

#                         # add pvalue
#                         p_value = result.pvalues.get(param, None)
#                         model_stats[f"{param}_pvalue"] = p_value

#                         # add CI
#                         conf_int = result.conf_int(alpha = alpha)
#                         if param in conf_int.index:
#                             model_stats[f"{param}_CI_low"] = conf_int.loc[param, 0]
#                             model_stats[f"{param}_CI_up"] = conf_int.loc[param, 1]

#         summary = pd.concat([summary, pd.DataFrame(model_stats, index = [0])], ignore_index = True)

#     return summary



# ============================================================
Source: 03_Logistic_regression_model-V2.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# # Logistic regression model
# ### Keyong Deng, Clinical Epidemiology,LUMC
# Fit a logistic regression model containing additive effects for all covariates.  
# The `GLM` function fits many types of generalized linear models (GLMs).  
# Choosing the `Binomial` family makes it a logistic regression.

# In[1]:


import pandas as pd
# from Function_Logistic_Model import logist_model_df


# In[2]:


Dyslipidemia = pd.read_csv("Dylipidemia.csv")
Dyslipidemia.shape #7813 obs


# ### Read the other dataset

# In[5]:


# Diet_pattern scores

DPs = pd.read_csv("../01_DPs_Diet_patterns/Adj/DPs_Final_score_outlieradj_v2.csv")
DPs.shape


# ## Define function to scale the variables

# In[6]:


def std_columns(df, start_col = 2, end_col = 10, lastfix = "_scaled"):
    result_df = df.copy()

    cols_select = df.columns[start_col:end_col+1]

    for cols in cols_select:
        mean = df[cols].mean()
        std = df[cols].std()
        result_df[f"{cols}{lastfix}"] = (df[cols] - mean)/ std

    return result_df

DPs_scale = std_columns(DPs, start_col=1, end_col=5)
DPs_scale.to_csv("./Data/DPs_scale_v2.csv", index = False)


# ## Read in the Alcohol intake data

# In[7]:


Nutrients = pd.read_csv("/home/ec2-user/studies/ruifang/Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2 = Nutrients_2.loc[~(Nutrients_2["alcohol_g"] < 0), ]

Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
Nutrients_2.shape


# In[8]:


Nutrients_2["alcohol_g"].describe()


# ## lifestyle information

# In[9]:


lifestyles = pd.read_csv("../02_Lifestyle factors/lifestyle_factor_all_disease_V2.csv")

lifestyles = pd.merge(lifestyles, Nutrients_2, on = "participant_id", how = "left")
lifestyles.head()


# In[10]:


lifestyles.shape


# In[11]:


merge_df = pd.merge(DPs_scale, lifestyles, on = "participant_id", how = "left")
merge_df


# In[12]:


merge_df_2 = pd.merge(merge_df, 
                      Dyslipidemia[["participant_id", "dylipidemia"]],
                     on = "participant_id", how = "left")
merge_df_2


# In[13]:


import statsmodels.api as sm
# model1 = sm.GLM.from_formula("low ~ age + smoke + race + lwt + ptl + ht + ui + ftv", family=sm.families.Binomial(), data=data)
# result1 = model1.fit()
# print(result1.summary())


# ### A function to run the logistic regression model 

# In[45]:


def logist_model_df(data, dependent_var):
    dfs = []
    for covariates in [covariate_sets_test_AHEI, 
                       covariate_sets_test_rDII, 
                       covariate_sets_test_AMED, 
                       covariate_sets_test_rEDIH,
                  covariate_sets_test_hPDI]:
        models_test = multiple_glm_model(data = data, 
                                         dependent_var = dependent_var, 
                                         covariate_sets = covariates)
        Exposure = covariates.get("model1", [])[0]
        
        # Also combine the other Dataframes: model1, model2, model3...
        for model in ["model1", "model2", "model3", "model4"]:
            coef = models_test[model].params[Exposure]
            pvalue = models_test[model].pvalues[Exposure]
            conf_int = models_test[model].conf_int(alpha = 0.05).loc[Exposure]
            lower_ci, upper_ci = conf_int
            
           # Create a new Pandas Dataframe
            result_all = pd.DataFrame({
                                "N": models_test[model].nobs,
                                "Model": [model],
                                "Exposure" : [Exposure],
                                "Outcome" : [dependent_var],
                                "Coef": [coef],
                                "pval" : [pvalue],
                                "lower_ci" : [lower_ci],
                                "upper_ci" : [upper_ci],
                                "OR": [np.exp(coef)],
                                "lower_OR_ci" : [np.exp(lower_ci)],
                                "upper_OR_ci" : [np.exp(upper_ci)]
                            })
            # dfs.append(result_all)
            dfs = dfs + [result_all]
            
    return dfs

def multiple_glm_model (data, dependent_var, covariate_sets, family = sm.families.Binomial()):
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


# In[15]:


merge_df_2.columns[merge_df_2.columns.str.contains("sex")]
print(merge_df_2.sex_x.value_counts()) # select this variable
print(merge_df_2.sex_y.value_counts())
merge_df_2.sex.value_counts()

# merge_df_2.columns[merge_df_2.columns.str.contains("bmi")]


# In[60]:


# merge_df_2.dropna(subset = ["sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"])
# merge_df_2.dylipidemia.dtypes


# In[16]:


# merge_df_2.shape # 9617
merge_df_2["dylipidemia"].value_counts(dropna=False)


# In[17]:


# merge_df_2_complete = merge_df_2.dropna(subset = ["dylipidemia"])
# print(merge_df_2_complete.shape) # 6859
# merge_df_2["dylipidemia"] = merge_df_2["dylipidemia"].replace({"Yes": 1, "No": 0})

# Using map function to recode variable
merge_df_2["dylipidemia"] = merge_df_2["dylipidemia"].map({"Yes": 1, "No": 0})
merge_df_2["dylipidemia"].value_counts(dropna = False)


# In[18]:


# read in the GGM data
CGM = pd.read_csv("../10_CGM_Data/Data_CGM_base.csv")
CGM.head()


# In[23]:


CGM.shape
CGM["participant_id"].nunique()


# In[28]:


CGM[CGM.duplicated(subset = ["participant_id"], keep = False)]


# In[29]:


CGM = CGM.drop_duplicates(subset = ["participant_id"], keep = "first") # keep the first one of the duplicates
CGM.shape


# In[30]:


merge_df_CGM = pd.merge(merge_df, CGM[["participant_id", "T2DM_Diagnose"]],
                     on = "participant_id", how = "left")

# merge_df_CGM["T2DM_Diagnose"] = merge_df_CGM["T2DM_Diagnose"].replace({"Yes": 1, "No":0})
merge_df_CGM["T2DM_Diagnose"] = merge_df_CGM["T2DM_Diagnose"].map({"Yes": 1, "No":0})
merge_df_CGM["T2DM_Diagnose"].value_counts(dropna = False)


# ## Read in the hypertension data

# In[66]:


# df_hypertension = pd.read_csv("../09_Blood_pressure/Hypertension_baseline.csv")

# merge_df_HP = pd.merge(merge_df, df_hypertension[["participant_id", "hypertension_diagnosed"]],
#                      on = "participant_id", how = "left")

# # merge_df_HP["hypertension_diagnosed"] = merge_df_HP["hypertension_diagnosed"].replace({"Yes":1, "No": 0})
# merge_df_HP["hypertension_diagnosed"] = merge_df_HP["hypertension_diagnosed"].map({"Yes":1, "No": 0})
# merge_df_HP["hypertension_diagnosed"].value_counts(dropna = False)


# # Read in the NAFLD data
# 

# In[67]:


# import pandas as pd

# df_NAFLD = pd.read_csv("../05_Medication_status/Medication_baseline.csv")
# merge_df_NAFLD = pd.merge(merge_df, df_NAFLD[["participant_id", "NAFLD_baseline_diagnosis"]],
#                           on = "participant_id", how = "left")

# merge_df_NAFLD["NAFLD_baseline_diagnosis"] = merge_df_NAFLD["NAFLD_baseline_diagnosis"].map({"Yes":1, "No": 0})
# merge_df_NAFLD["NAFLD_baseline_diagnosis"].value_counts(dropna = False)


# # Read in another NAFLD data

# In[68]:


# df_MAFLD = pd.read_csv("../05_Medication_status/MAFLD_baseline.csv")

# merge_df_MAFLD = pd.merge(merge_df, df_MAFLD[["participant_id", "mafld_diagnosed"]],
#                           on = "participant_id", how = "left")

# merge_df_MAFLD["MAFLD_baseline_diagnosis"] = merge_df_MAFLD["mafld_diagnosed"].map({"Yes":1, "No": 0})
# merge_df_MAFLD["MAFLD_baseline_diagnosis"].value_counts(dropna = False)


# # Define the Cardiometabolic disease

# In[31]:


merge_df_2 # combine with T2DM
merge_df_CMD = pd.merge(merge_df_2, 
                        CGM[["participant_id", "T2DM_Diagnose"]],
                        on = "participant_id", how = "left")
merge_df_CMD
merge_df_CMD["T2DM_Diagnose"] = merge_df_CMD["T2DM_Diagnose"].map({"Yes": 1, "No":0})

print(merge_df_CMD["T2DM_Diagnose"].value_counts(dropna = False))


# In[32]:


merge_df_CMD["dylipidemia"].value_counts(dropna = False)


# In[33]:


# Make the crosstab table for the T2DM and dyslipidemia
pd.crosstab(merge_df_CMD["T2DM_Diagnose"], merge_df_CMD["dylipidemia"], margins = True, dropna = False)


# In[34]:


# to calculate both the dylipidemia and T2DM_Diagnose
# conditions = (merge_df_CMD["dylipidemia"] == 1) | (merge_df_CMD["T2DM_Diagnose"] == 1)
# has_na = merge_df_CMD["dylipidemia"].isna() | merge_df_CMD["T2DM_Diagnose"].isna()
# conditions.value_counts() # 1715 cases 
# merge_df_CMD.loc[:, "CMD_baseline_diagnosis"] = np.where(has_na, np.nan, conditions.map({True:1, False:0}))
# merge_df_CMD
import numpy as np
def determine_value(var1, var2):
    if var1 == 1 or var2 == 1:
        return 1
    elif var1 == 0 and var2 == 0:
        return 0
    elif (var1 == 1 and pd.isna(var2)) or (pd.isna(var1) and var2 == 1):
        return 1
    else:
        return np.nan

# apply function row-wise
merge_df_CMD["CMD_baseline_diagnosis"] = merge_df_CMD.apply(
    lambda row: determine_value(row["dylipidemia"], row["T2DM_Diagnose"]), 
    axis=1
)
merge_df_CMD["CMD_baseline_diagnosis"].value_counts(dropna = False)


# In[37]:


# Update: the NaN to 0
merge_df_CMD["CMD_baseline_diagnosis"] = merge_df_CMD["CMD_baseline_diagnosis"].replace(np.nan, 0)
merge_df_CMD["CMD_baseline_diagnosis"]


# In[38]:


merge_df_CMD["CMD_baseline_diagnosis"].value_counts()


# In[36]:


merge_df_CMD["age_all"].isnull().sum()


# ## The same function for three outcomes

# In[76]:


# change the level of factor variables
# merge_df_2_complete["dylipidemia"]= pd.Categorical(merge_df_2_complete['dylipidemia'], 
#                             categories=['0', '1'],
#                             ordered=True)


# In[39]:


covariate_sets_test_AHEI = {"model1": ["AHEI_2010_score_eadj_scaled"],
                      
                       "model2": ["AHEI_2010_score_eadj_scaled", "age_all", 
                             "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"],
                       
                       "model3": ["AHEI_2010_score_eadj_scaled", "age_all", "bmi",
                             "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"],
                      
                       "model4": ["AHEI_2010_score_eadj_scaled", "age_all",
                                 "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use", 
                                  'NSAID_use', 'CVD_f_history', 'T2D_f_history']}


# In[40]:


covariate_sets_test_AHEI


# In[41]:


## Define a function to modify the element in each list and return a new dictionary
## -- Then we get the covariates for the other dietary pattern scores
def modify_dict(original, index, new_value, new_element = None):
    new_dict = {}
    for key, value_list in original.items():
        new_list = value_list.copy()
        # change the elements in the list by new value
        new_list[index] = new_value
        
        if key not in ["model1"]:
            if new_element is not None:
                new_list.append(new_element)
                
        new_dict[key] = new_list
    return new_dict

covariate_sets_test_rDII = modify_dict(covariate_sets_test_AHEI, 0, "rDII_score_eadj_scaled")
covariate_sets_test_AMED = modify_dict(covariate_sets_test_AHEI, 0, "AMED_score_eadj_scaled")
covariate_sets_test_rEDIH = modify_dict(covariate_sets_test_AHEI, 0, "rEDIH_score_all_eadj_scaled")
covariate_sets_test_hPDI = modify_dict(covariate_sets_test_AHEI, 0, "hPDI_score_eadj_scaled", new_element= "alcohol_g")


# In[62]:


cov_check_missing = ["AHEI_2010_score_eadj_scaled", "age_all", "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use", "alcohol_g"]
merge_df_CMD_full = merge_df_CMD.dropna(subset = cov_check_missing)
merge_df_CMD_full.shape


# In[46]:


results_CMD = logist_model_df(data=merge_df_CMD, dependent_var="CMD_baseline_diagnosis")
results_CMD_all = pd.concat(results_CMD)
results_CMD_all.round(2)


# In[78]:


cat_order = ["AHEI_2010_score_eadj_scaled", "AMED_score_eadj_scaled", "hPDI_score_eadj_scaled", "rDII_score_eadj_scaled", "rEDIH_score_all_eadj_scaled"]
results_CMD_all["Exposure"] = pd.Categorical(results_CMD_all["Exposure"], categories= cat_order, ordered = True)

# now sort value based on the order
results_CMD_all_sorted = results_CMD_all.sort_values(by = ["Exposure", "Model"])
results_CMD_all_sorted.round(2)


# In[ ]:





# In[63]:


merge_df_CMD_full = merge_df_CMD.dropna(subset = "CMD_baseline_diagnosis")
merge_df_CMD_full


# In[86]:


# # Make the Tableone
# !pip install --quiet tableone
# from tableone import TableOne


# In[48]:


# print(merge_df_CMD_full["age_all"].describe().round(0))
# print(merge_df_CMD_full["sex_x"].value_counts(dropna = False))
# merge_df_CMD_full["sex_x"].value_counts(dropna = False, normalize=True)


# In[49]:


# print(merge_df_CMD_full["smoking_status"].value_counts(dropna = False))
# merge_df_CMD_full["smoking_status"].value_counts(dropna = False, normalize=True)


# In[50]:


# merge_df_CMD_full["edu_status"].value_counts(dropna = False)


# In[51]:


# merge_df_CMD_full["edu_status"].value_counts(dropna = False, normalize=True)


# In[52]:


# merge_df_CMD_full["bmi"].describe().round(1)


# In[53]:


# merge_df_CMD_full["sleep_hours_daily"].describe().round(1)


# In[54]:


# replace those value < 0 as NA
# merge_df_CMD_full.loc[merge_df_CMD_full["alcohol_g"] < 0, "alcohol_g"] = np.nan
# merge_df_CMD_full["alcohol_g"].describe().round(1)


# In[55]:


# print(merge_df_CMD_full["MET_hour"].describe().round(1))
# print(merge_df_CMD_full["Vit_use"].value_counts(dropna = False))
# merge_df_CMD_full["Vit_use"].value_counts(dropna = False, normalize = True)


# In[56]:


# print(merge_df_CMD_full["Hormone_use"].value_counts(dropna = False))
# merge_df_CMD_full["Hormone_use"].value_counts(dropna = False, normalize = True)


# In[57]:


display(merge_df_CMD_full[['AHEI_2010_score_eadj', 'AMED_score_eadj', 'hPDI_score_eadj', "rDII_score_eadj",'rEDIH_score_all_eadj']].describe().round(1))


# In[58]:


# print(merge_df_CMD_full.columns.tolist())


# In[59]:


# Function to calculate the median and IQR in the data frame
def calculate_summary(df, columns):
    results = []

    for col in columns:
        N = len(df[col])
        N_missing = df[col].isna().sum()
        N_valid = N - N_missing
        Missing_pct = (N_missing / N) * 100

        # calculate the summaries
        mean = df[col].mean()
        sd = df[col].std()
        median = df[col].median()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR = q3 - q1

        # results append
        results.append({
            "Variable": col,
            "N_total": N,
            "N_valid" : N_valid,
            "N_missing" : N_missing,
            "Pct_missing" : Missing_pct,
            "Mean" : mean,
            "SD" : sd,
            "Mean(SD)" : f"{mean:.2f}({sd:.2f})",
            "Median" : median,
            "Q1": q1,
            "Q3": q3,
            "IQR": IQR,
            "Median(IQR)": f"{median:.1f}({q1:.1f} - {q3:.1f})"
        }
            
        )

    return pd.DataFrame(results)


# In[60]:


columns_analyze = ["age_all", "bmi", "sleep_hours_daily", "MET_hour", 
                   "alcohol_g", 'AHEI_2010_score_eadj', 'hPDI_score_eadj', 
                   'rDII_score_eadj', 'AMED_score_eadj', 'rEDIH_score_all_eadj',]
calculate_summary(merge_df_CMD_full, columns= columns_analyze)


# In[64]:


# A function to run the analysis for the categorical variables
def calculate_categorical_stats(df, columns):
    
    results = []
    for col in columns:
        Total = len(df[col])
        N_missing = df[col].isna().sum()
        N_valid = Total - N_missing
        Pct_missing = (N_missing / Total) * 100

        # get the value counts
        value_counts = df[col].value_counts(dropna = False)

        for category, count in value_counts.items():
            pct_to_valid = (count/N_valid) * 100 if N_valid > 0 else 0
            pct_to_total = (count/Total) * 100

            results.append({
                "Variable": col,
                "Category": category,
                "N": Total,
                "Pct_of_Valid": round(pct_to_valid, 2),
                "Pct_of_Total": round(pct_to_total, 2),
                "N (%)" : f"{count}({pct_to_total:.1f}%)",
                "N_Total" : Total,
                "N_valid" : N_valid,
                "N_missing" : N_missing,
                "Pct_missing": Pct_missing  
            })

    return(pd.DataFrame(results))


# In[65]:


cat_columns = ["sex_x", 'Vit_use', 'Hormone_use', 'smoking_status', "edu_status"]
calculate_categorical_stats(merge_df_CMD_full, columns=cat_columns)


# In[66]:


col_select = ['AMED_score_eadj', 'AHEI_2010_score_eadj', 'rEDIH_score_all_eadj', "rDII_score_eadj", 'hPDI_score_eadj']
merge_df_CMD_select = merge_df_CMD[col_select]


# In[67]:


# function to calculate the pvalue_matrix
def calculate_pvalue_matrix(data):
    
    cols = data.columns
    n_cols = len(cols)
    #
    correlation_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), columns = cols, index = cols)
    p_value_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), columns = cols, index = cols)
    
    # compute the correlation and pvalue
    for i in range(n_cols):
        for j in range(n_cols):
            correlation, pvalue = stats.spearmanr(data[cols[i]], data[cols[j]])
            correlation_matrix.iloc[i, j] = correlation
            p_value_matrix.iloc[i, j] = round(pvalue, 3)
            
    combined_matrix = pd.DataFrame(
        index = correlation_matrix.index,
        columns = correlation_matrix.columns,
        data = [[f"{correlation_matrix.iloc[i,j]:.3f}\n(p={p_value_matrix.iloc[i,j]:.3f})"
                for j in range(n_cols)]
               for i in range(n_cols)]
    )
    
    return combined_matrix, correlation_matrix, p_value_matrix
    
    # visualization
#   fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))
    # plt.figure(figsize=(20,8))
    
    # # correlation heatmap
    # plt.subplot(1, 2, 1)
    # sns.heatmap(correlation_matrix, 
    #             annot = True, 
    #             fmt = ".2f", # format values into 2 decimal places
    #             # cmap = "coolwarm", 
    #             cmap = "reds",
    #             square = True,
    #             linewidths = 1, 
    #             linecolor = "white",
    #             vmin = -1, 
    #             vmax = 1, 
    #             center = 0, ax = ax1)
    
    # ax1.set_title('Correlation Coefficients')
    
    # # P-value heatmap
    # plt.subplot(1, 2, 2)
    # sns.heatmap(p_value_matrix,
    #             annot=True,
    #             cmap='YlOrRd',
    #             vmin=0, vmax=0.1,
    #             ax=ax2)
    
    # ax2.set_title('P-values')
    
    # plt.tight_layout()
    # plt.show()
                


# In[74]:


from scipy import stats
merge_df_CMD_select.shape


# In[69]:


combined_matrix_r, correlation_matrix_r, p_value_matrix_r = calculate_pvalue_matrix(merge_df_CMD_select)


# In[70]:


correlation_matrix_r = correlation_matrix_r.rename(
    index ={'AMED_score_eadj': 'AMED', 
            "AHEI_2010_score_eadj" : "AHEI",
            "rEDIH_score_all_eadj" : "rEDIH",
            'rDII_score_eadj': 'rDII',
            'hPDI_score_eadj': 'hPDI'},
    columns ={'AMED_score_eadj': 'AMED', 
            "AHEI_2010_score_eadj" : "AHEI",
            "rEDIH_score_all_eadj" : "rEDIH",
            'rDII_score_eadj': 'rDII',
            'hPDI_score_eadj': 'hPDI'})


# In[71]:


correlation_matrix_r.round(2)


# In[72]:


import seaborn as sns
import matplotlib.pyplot as plt

# keep the lower triangle
mask = np.triu(np.ones_like(correlation_matrix_r), k = 1)
mask


# In[73]:


ax = sns.heatmap(correlation_matrix_r, 
                annot = True, 
                fmt = ".2f", # format values into 2 decimal places
                cmap = "coolwarm", 
                # cmap = "Reds",
                square = True,
                # linewidths = 1, 
                # linecolor = "white",
                vmin = -1, 
                vmax = 1, 
                center = 0,
                mask = mask,
                cbar_kws = {"orientation":"horizontal", "pad": 0.1,
                            "shrink":0.65,
                            "aspect": 30})

# plt.tight_layout()

ax.set_xticklabels([])
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)


# ## Run the logistic model

# In[ ]:


def logist_model_df(data, dependent_var):
    dfs = []
    for covariates in [covariate_sets_test_AHEI, 
                       covariate_sets_test_rDII, 
                       covariate_sets_test_AMED, 
                       covariate_sets_test_rEDIH,
                  covariate_sets_test_hPDI]:
        models_test = multiple_glm_model(data = data, 
                                         dependent_var = dependent_var, 
                                         covariate_sets = covariates)
        Exposure = covariates.get("model1", [])[0]
        
        # Also combine the other Dataframes: model1, model2, model3...

        for model in ["model1", "model2", "model3", "model4"]:
            coef = models_test[model].params[Exposure]
            pvalue = models_test[model].pvalues[Exposure]
            conf_int = models_test[model].conf_int(alpha = 0.05).loc[Exposure]
            lower_ci, upper_ci = conf_int
            
           # Create a new Pandas Dataframe
            result_all = pd.DataFrame({
                                "Model": [model],
                                "Exposure" : [Exposure],
                                "Outcome" : [dependent_var],
                                "Coef": [coef],
                                "pval" : [pvalue],
                                "lower_ci" : [lower_ci],
                                "upper_ci" : [upper_ci],
                                "OR": [np.exp(coef)],
                                "lower_OR_ci" : [np.exp(lower_ci)],
                                "upper_OR_ci" : [np.exp(upper_ci)]
                            })
            # dfs.append(result_all)
            dfs = dfs + [result_all]
    return dfs


# In[ ]:


merge_df_NAFLD["NAFLD_baseline_diagnosis"].value_counts(dropna = False)


# In[ ]:


import numpy as np
# for Dyslipidemia
results_dyslipidemia = logist_model_df(data=merge_df_2, dependent_var="dylipidemia")
results_hypertension = logist_model_df(data=merge_df_HP, dependent_var="hypertension_diagnosed") # merge_df_HP["hypertension_diagnosed"] 
results_diabetes = logist_model_df(data=merge_df_CGM, dependent_var="T2DM_Diagnose")


# In[ ]:


results_CMD = logist_model_df(data=merge_df_CMD, dependent_var="CMD_baseline_diagnosis")
results_CMD_all = pd.concat(results_CMD)
results_CMD_all


# In[ ]:


results_CMD_all.round(2)


# In[ ]:


# merge_df_NAFLD = merge_df_NAFLD.dropna(subset = "hypertension_diagnosed")
results_NAFLD = logist_model_df(data=merge_df_NAFLD, dependent_var="NAFLD_baseline_diagnosis")
results_MAFLD = logist_model_df(data=merge_df_MAFLD, dependent_var="MAFLD_baseline_diagnosis")


# In[ ]:


# change the decimal to three

result1 = pd.concat(results_dyslipidemia)
result2 = pd.concat(results_diabetes)
result3 = pd.concat(results_hypertension)
result4 = pd.concat(results_NAFLD)
result5 = pd.concat(results_MAFLD)

results_all = pd.concat([result1, result2, result3, result4, result5])

results_all['OR'] = results_all['OR'].apply(lambda x: float("{:.3f}".format(x)))
results_all['lower_OR_ci'] = results_all['lower_OR_ci'].apply(lambda x: float("{:.3f}".format(x)))
results_all['upper_OR_ci'] = results_all['upper_OR_ci'].apply(lambda x: float("{:.3f}".format(x)))

results_all["Outcome"] = results_all["Outcome"].replace({
    "dylipidemia" : "Dyslipidemia",
    "T2DM_Diagnose" : "Type 2 Diabetes",
    "hypertension_diagnosed" : "Hypertension",
    "NAFLD_baseline_diagnosis" : "NAFLD",
    "MAFLD_baseline_diagnosis" : "MAFLD"
})


# In[ ]:


# Capitalized the first column
results_all["Model"] = results_all["Model"].str.capitalize()
results_all.head()
# results_all.to_csv("Results_for_all.csv", index = False)


# ## Plot the results

# In[ ]:


results_all
results_all.to_csv("./Results/Logistical_Regression.csv", index = False)


# In[ ]:


enumerate(results_all.groupby("Outcome"))


# In[ ]:


results_all["Exposure"].value_counts()


# In[ ]:


results_all["Exposure"] = results_all["Exposure"].replace({
    "AHEI_2010_score_eadj_scaled": "AHEI",
    "rDII_score_eadj_scaled" : "rDII",
    "AMED_score_eadj_scaled" : "AMED",
    "rEDIH_score_all_eadj_scaled" : "rEDIH",
    "hPDI_score_eadj_scaled" : "hPDI"
})


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl


# In[ ]:


fig, axes = plt.subplots(2, 3, figsize = (12,6))
axes_flat = axes.flatten()  # Flatten the 2D array to make indexing easier
# colormap = cm["tab20"]

# my_colors = ["#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
# my_cmap = mlt.colors.ListedColormap(my_colors, name="my_cmap")
my_cmap = mpl.colormaps['Paired'].resampled(4)

for i, (group_name, group_data) in enumerate(results_all.groupby("Outcome")):
    # choose the suplot
    # row, col = i // 3, i % 3  # For positioning in a 2×3 grid
    if i < 5:
        ax = axes_flat[i]
    #ax = axes[row,col]
        all_exposure = group_data["Exposure"].unique()
    # print(all_exposure)

    # name the subplot
        ax.set_title(f"{group_name}", fontsize=12)
    
        exposure_position = {exp: idx for idx, exp in enumerate(all_exposure)}

        models = group_data["Model"].unique()
    
        colors = {model: my_cmap(i/len(models)) for i, model in enumerate(models)}
    # print(colors)

    # group by model to allow for dodging
        for j, (model_name, model_data) in enumerate(group_data.groupby("Model")):

            dodge_positions = j * 0.2 - 0.3
            model_color = colors[model_name]

        # convert string exposure to numeric positions
            x_positions = np.array([exposure_position[exp] for exp in model_data["Exposure"]]) + dodge_positions
        # print(x_positions)

        # plot each point with error bars
            ax.errorbar(
                x = x_positions,
                y = model_data["OR"],
                yerr = [model_data["OR"] - model_data["lower_OR_ci"],
                   model_data["upper_OR_ci"]- model_data["OR"]],
                fmt = "o",
                capsize = 4,
                capthick = 1.2,
                linewidth = 1.2,
                color = model_color
        )
            ax.scatter(x_positions, model_data["OR"], color = model_color, label = model_name)
        # add reference line
            ax.axhline(y = 1, linestyle = "--", color = "grey")
    
    # set the ylim
       
        ax.set_ylim(bottom = 0.6, top = 1.4)

    # set new label for the exposure
        ax.set_xticks(np.arange(len(all_exposure)))
        ax.set_xticklabels(all_exposure)
        ax.set_ylabel("Odds Ratio with 95%CI")

# add legend
# ax.legend()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')

plt.tight_layout()
# Make room for the legend
plt.subplots_adjust(bottom = 0.2, wspace = 0.3)  # wspace means width space between subplots
plt.savefig("./Results/Logistic_regression_DPs_V2.pdf")
plt.show()


# In[ ]:


##### Another Diet index
# import numpy as np
# models_test = multiple_glm_model(data = merge_df_2, dependent_var = "dylipidemia", covariate_sets = covariate_sets_test_AMED)

# # Also combine the other Dataframes: model1, model2, model3...
# dfs = []
# for model in ["model1", "model2", "model3", "model4"]:
#     coef = models_test[model].params["AMED_score_eadj_scaled"]
#     pvalue = models_test[model].pvalues["AMED_score_eadj_scaled"]
#     conf_int = models_test[model].conf_int(alpha = 0.05).loc["AMED_score_eadj_scaled"]
#     lower_ci, upper_ci = conf_int
    
#     # Create a new Pandas Dataframe
#     result_all = pd.DataFrame({
#         "Model": [model],
#         "Exposure" : ["AMED_score_eadj_scaled"],
#         "Coef": [coef],
#         "pval" : [pvalue],
#         "lower_ci" : [lower_ci],
#         "upper_ci" : [upper_ci],
#         "OR": [np.exp(coef)],
#         "lower_OR_ci" : [np.exp(lower_ci)],
#         "upper_OR_ci" : [np.exp(upper_ci)]
#     })
#     dfs.append(result_all)

# result_all = pd.concat(dfs, ignore_index=True)
# result_all


# In[ ]:


# res_m2 = models_test["model2"].summary().tables[1]
# res_m2.loc["AHEI_2010_score_eadj_scaled", ["coef", "P>|z|"]]


# In[ ]:


# display(models_test["model2"].summary2().tables[1].loc["AHEI_2010_score_eadj_scaled"])
# print(np.exp(models_test["model2"].summary2().tables[1].loc["AHEI_2010_score_eadj_scaled"][0]))
# np.exp(models_test["model2"].conf_int().loc["AHEI_2010_score_eadj_scaled"])


# In[ ]:


# def summarize_models(models, variable_interest = None, alpha = 0.05):
#     summary = pd.DataFrame()

#     # Basic model status
#     for model_name, result in models.items():
#         model_stats = {
#             "Model": model_name,
#             "AIC": result.aic,
#             "BIC": result.bic,
#             "N": result.nobs
#         }

#         if variable_interest:
#             for var in variable_interest:
#                 var_params = [param for param in result.param.index if var in param]

#                 for param in var_params:
#                     coef = result.param.get(param, None)
#                     if coef is not None:
#                         # add coefficient
#                         model_stats[f"{param}_coef"] = coef

#                         # add pvalue
#                         p_value = result.pvalues.get(param, None)
#                         model_stats[f"{param}_pvalue"] = p_value

#                         # add CI
#                         conf_int = result.conf_int(alpha = alpha)
#                         if param in conf_int.index:
#                             model_stats[f"{param}_CI_low"] = conf_int.loc[param, 0]
#                             model_stats[f"{param}_CI_up"] = conf_int.loc[param, 1]

#         summary = pd.concat([summary, pd.DataFrame(model_stats, index = [0])], ignore_index = True)

#     return summary


# In[ ]:






# ============================================================
Source: 04_Logistic_regression_MASLD.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ### Logistic regression model-MASLD
# #### Keyong Deng, Clinical Epidemiology,LUMC
# Fit a logistic regression model containing additive effects for all covariates.  
# The `GLM` function fits many types of generalized linear models (GLMs).  
# Choosing the `Binomial` family makes it a logistic regression.

# In[1]:


def std_columns(df, start_col = 2, end_col = 10, lastfix = "_scaled"):
    result_df = df.copy()

    cols_select = df.columns[start_col:end_col+1]

    for cols in cols_select:
        mean = df[cols].mean()
        std = df[cols].std()
        result_df[f"{cols}{lastfix}"] = (df[cols] - mean)/ std

    return result_df


# In[6]:


import pandas as pd

# Dyslipidemia = pd.read_csv("Dylipidemia.csv")
# Dyslipidemia.head()

# Read in the DPs
DPs = pd.read_csv("../01_DPs_Diet_patterns/Adj/DPs_Final_score_outlieradj_v2.csv")
DPs.head()

# Read in the alcohol intake
Nutrients = pd.read_csv("/home/ec2-user/studies/ruifang/Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
Nutrients_2.shape

lifestyles = pd.read_csv("../02_Lifestyle factors/lifestyle_factor_all_disease_V2.csv")

lifestyles = pd.merge(lifestyles, Nutrients_2, on = "participant_id", how = "left")
lifestyles.head()


# In[8]:


DPs_scale = std_columns(DPs, start_col=1, end_col=5)
DPs_scale.to_csv("./Data/DPs_scale_v2.csv", index = False)


# In[9]:


merge_df = pd.merge(DPs_scale, lifestyles, on = "participant_id", how = "left")
merge_df


# In[10]:


merge_df["alcohol_g"].describe()


# In[11]:


merge_df["bmi"].describe()

merge_df["bmi_cat"] = merge_df["bmi"].apply(
    lambda x: 0 if x < 25 else 1
).astype("category")
merge_df["bmi_cat"].value_counts()


# In[12]:


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


# In[13]:


# Make another function to include interactions
def multiple_glm_model_2 (data, dependent_var, covariate_sets, 
                          vars_of_interest = None, 
                          include_interactions = None,
                          family = sm.families.Binomial()):
    models = {}

    for model_name, covariates in covariate_sets.items():
        # start with base covariates
        formula_term = covariates.copy()

        # add only specific interaction
        if vars_of_interest and include_interactions:
            for interaction in include_interactions:
                if interaction not in formula_term:
                    formula_term.append(interaction)

        formula = f"{dependent_var} ~ {'+'.join(formula_term)}"
        print(f'Fitting {model_name} with formula : {formula}')

        # fit the model (remove missing)
        model = sm.GLM.from_formula(formula, family = family, data = data, missing = "drop")
        result = model.fit()
        models[model_name] = result
    return models


# In[15]:


# covariate_sets_test_AHEI_model2 = covariate_sets_test_AHEI["model2"]
# covariate_sets_test_AHEI_model2 = {"model2" : covariate_sets_test_AHEI["model2"]}


# In[16]:


# covariate_sets_test_AHEI_model2


# In[17]:


# temp =  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_AHEI_model2,
#                    vars_of_interest = ["AHEI_2010_score_eadj_scaled"],
#                    include_interactions = ["AHEI_2010_score_eadj_scaled:sex_x"])
# temp


# In[18]:


# merge_df_NAFLD["bmi"].describe()

# merge_df_NAFLD["bmi_cat"] = merge_df_NAFLD["bmi"].apply(
#     lambda x: 0 if x < 25 else 1
# ).astype("category")
# print(merge_df_NAFLD["bmi_cat"].value_counts())

# covariate_sets_test_rDII_model2 = {"model2" : covariate_sets_test_rDII["model2"]}

# temp2 =  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_rDII_model2,
#                    vars_of_interest = ["rDII_score_eadj_scaled"],
#                    include_interactions = ["rDII_score_eadj_scaled:bmi_cat"])
# print(temp2["model2"].params)
# temp2["model2"].pvalues


# In[19]:


# print(temp["model2"].params)
# temp["model2"].pvalues


# In[20]:


merge_df.columns[merge_df.columns.str.contains("sex")]
print(merge_df.sex_x.value_counts()) # select this variable
print(merge_df.sex_y.value_counts())
merge_df.sex.value_counts()


# In[21]:


import pandas as pd

# Based on the medication diagnosis (self-reported of medication intake)
df_NAFLD = pd.read_csv("../05_Medication_status/Medication_baseline.csv")
merge_df_NAFLD = pd.merge(merge_df, df_NAFLD[["participant_id", "NAFLD_baseline_diagnosis"]],
                          on = "participant_id", how = "left")

merge_df_NAFLD["NAFLD_baseline_diagnosis"] = merge_df_NAFLD["NAFLD_baseline_diagnosis"].map({"Yes":1, "No": 0})
display(merge_df_NAFLD["NAFLD_baseline_diagnosis"].value_counts(dropna = False))

# Based on another definition of MAFLD(speed of sound)
df_MAFLD = pd.read_csv("../05_Medication_status/MAFLD_baseline.csv")
merge_df_MAFLD = pd.merge(merge_df, df_MAFLD[["participant_id", "mafld_diagnosed"]],
                          on = "participant_id", how = "left")

merge_df_MAFLD["MAFLD_baseline_diagnosis"] = merge_df_MAFLD["mafld_diagnosed"].map({"Yes":1, "No": 0})
merge_df_MAFLD["MAFLD_baseline_diagnosis"].value_counts(dropna = False)


# In[22]:


covariate_sets_test_AHEI = {"model1": ["AHEI_2010_score_eadj_scaled"],
                      
                       "model2": ["AHEI_2010_score_eadj_scaled", "age", 
                             "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"],
                       
                       "model3": ["AHEI_2010_score_eadj_scaled", "age", "bmi",
                             "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"],
                      
                       "model4": ["AHEI_2010_score_eadj_scaled", "age",
                                 "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use", 
                                  'NSAID_use', 'CVD_f_history', 'T2D_f_history']}

## Define a function to modify the element in each list and return a new dictionary
## -- Then we get the covariates for the other dietary pattern scores
def modify_dict(original, index, new_value, new_element = None):
    new_dict = {}
    for key, value_list in original.items():
        new_list = value_list.copy()
        # change the elements in the list by new value
        new_list[index] = new_value
        
        if key not in ["model1"]:
            if new_element is not None:
                new_list.append(new_element)
                
        new_dict[key] = new_list
    return new_dict

covariate_sets_test_rDII = modify_dict(covariate_sets_test_AHEI, 0, "rDII_score_eadj_scaled")
covariate_sets_test_AMED = modify_dict(covariate_sets_test_AHEI, 0, "AMED_score_eadj_scaled")
covariate_sets_test_rEDIH = modify_dict(covariate_sets_test_AHEI, 0, "rEDIH_score_all_eadj_scaled")
covariate_sets_test_hPDI = modify_dict(covariate_sets_test_AHEI, 0, "hPDI_score_eadj_scaled", new_element= "alcohol_g")


# In[23]:


def logist_model_df(data, dependent_var):
    dfs = []
    for covariates in [covariate_sets_test_AHEI, 
                       covariate_sets_test_rDII, 
                       covariate_sets_test_AMED, 
                       covariate_sets_test_rEDIH,
                  covariate_sets_test_hPDI]:
        models_test = multiple_glm_model(data = data, 
                                         dependent_var = dependent_var, 
                                         covariate_sets = covariates)
        Exposure = covariates.get("model1", [])[0]
        
        # Also combine the other Dataframes: model1, model2, model3...
        for model in ["model1", "model2", "model3", "model4"]:
            coef = models_test[model].params[Exposure]
            pvalue = models_test[model].pvalues[Exposure]
            conf_int = models_test[model].conf_int(alpha = 0.05).loc[Exposure]
            lower_ci, upper_ci = conf_int
            
           # Create a new Pandas Dataframe
            result_all = pd.DataFrame({
                                "Model": [model],
                                "Exposure" : [Exposure],
                                "Outcome" : [dependent_var],
                                "Coef": [coef],
                                "pval" : [pvalue],
                                "lower_ci" : [lower_ci],
                                "upper_ci" : [upper_ci],
                                "OR": [np.exp(coef)],
                                "lower_OR_ci" : [np.exp(lower_ci)],
                                "upper_OR_ci" : [np.exp(upper_ci)]
                            })
            # dfs.append(result_all)
            dfs = dfs + [result_all]
            
    return dfs


# In[24]:


merge_df_NAFLD.columns


# In[25]:


import numpy as np

# for NAFLD
results_NAFLD = logist_model_df(data = merge_df_NAFLD, dependent_var = "NAFLD_baseline_diagnosis")

results_MAFLD = logist_model_df(data = merge_df_MAFLD, dependent_var = "MAFLD_baseline_diagnosis")


# In[26]:


result1 = pd.concat(results_NAFLD)
result2 = pd.concat(results_MAFLD)


# In[27]:


results_all = pd.concat([result1, result2])

# Format the results
results_all['OR'] = results_all['OR'].apply(lambda x: float("{:.3f}".format(x)))
results_all['lower_OR_ci'] = results_all['lower_OR_ci'].apply(lambda x: float("{:.3f}".format(x)))
results_all['upper_OR_ci'] = results_all['upper_OR_ci'].apply(lambda x: float("{:.3f}".format(x)))

results_all["Outcome"] = results_all["Outcome"].replace({
    "NAFLD_baseline_diagnosis" : "MAFLD_Based_on_medication",
    "MAFLD_baseline_diagnosis" : "MAFLD_Based_on_speed_of_sound"
})


# In[30]:


# Capitalized the first column
results_all["Model"] = results_all["Model"].str.capitalize()
results_all


# In[31]:


results_all
results_all.to_csv("./Results/Logistical_Regression_MASLD.csv", index = False)


# In[32]:


print(results_all["Exposure"].value_counts())

results_all["Exposure"] = results_all["Exposure"].replace({
    "AHEI_2010_score_eadj_scaled": "AHEI",
    "rDII_score_eadj_scaled" : "rDII",
    "AMED_score_eadj_scaled" : "AMED",
    "rEDIH_score_all_eadj_scaled" : "rEDIH",
    "hPDI_score_eadj_scaled" : "hPDI"
})


# In[33]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

fig, axes = plt.subplots(2, 1, figsize = (10,8))
axes_flat = axes.flatten()  # Flatten the 2D array to make indexing easier
# colormap = cm["tab20"]

# my_colors = ["#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
# my_cmap = mlt.colors.ListedColormap(my_colors, name="my_cmap")
my_cmap = mpl.colormaps['Paired'].resampled(4)

for i, (group_name, group_data) in enumerate(results_all.groupby("Outcome")):
    # choose the suplot
    # row, col = i // 3, i % 3  # For positioning in a 2×3 grid
    if i < 5:
        ax = axes_flat[i]
    #ax = axes[row,col]
        all_exposure = group_data["Exposure"].unique()
    # print(all_exposure)

    # name the subplot
        ax.set_title(f"{group_name}", fontsize=12)
    
        exposure_position = {exp: idx for idx, exp in enumerate(all_exposure)}

        models = group_data["Model"].unique()
    
        colors = {model: my_cmap(i/len(models)) for i, model in enumerate(models)}
    # print(colors)

    # group by model to allow for dodging
        for j, (model_name, model_data) in enumerate(group_data.groupby("Model")):

            dodge_positions = j * 0.2 - 0.3
            model_color = colors[model_name]

        # convert string exposure to numeric positions
            x_positions = np.array([exposure_position[exp] for exp in model_data["Exposure"]]) + dodge_positions
        # print(x_positions)

        # plot each point with error bars
            ax.errorbar(
                x = x_positions,
                y = model_data["OR"],
                yerr = [model_data["OR"] - model_data["lower_OR_ci"],
                   model_data["upper_OR_ci"]- model_data["OR"]],
                fmt = "o",
                capsize = 4,
                capthick = 1.2,
                linewidth = 1.2,
                color = model_color
        )
            ax.scatter(x_positions, model_data["OR"], color = model_color, label = model_name)
        # add reference line
            ax.axhline(y = 1, linestyle = "--", color = "grey")
    
    # set the ylim
       
        ax.set_ylim(bottom = 0.6, top = 1.4)

    # set new label for the exposure
        ax.set_xticks(np.arange(len(all_exposure)))
        ax.set_xticklabels(all_exposure)
        ax.set_ylabel("Odds Ratio with 95%CI")

# add legend
# ax.legend()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')

plt.tight_layout()

# Make room for the legend
plt.subplots_adjust(bottom = 0.2, wspace = 0.3)  # wspace means width space between subplots
plt.savefig("./Results/Logistic_regression_DPs_MASLD.pdf")
plt.show()


# ## Perform the subgroup analysis

# In[35]:


# merge_df_MAFLD.columns.tolist()
merge_df_MAFLD.shape


# In[34]:


merge_df_MAFLD["sex_x"].value_counts()


# In[36]:


# Male
results_MAFLD_Male = logist_model_df(data = merge_df_MAFLD[merge_df_MAFLD["sex_x"] == "Male"], 
                                dependent_var = "MAFLD_baseline_diagnosis")
results_MAFLD_Male = pd.concat(results_MAFLD_Male)

# Female
results_MAFLD_Female = logist_model_df(data = merge_df_MAFLD[merge_df_MAFLD["sex_x"] == "Female"], 
                                dependent_var = "MAFLD_baseline_diagnosis")
results_MAFLD_Female = pd.concat(results_MAFLD_Female)


# In[72]:


# Format the results
# Male
results_MAFLD_Male['OR'] = results_MAFLD_Male['OR'].apply(lambda x: float("{:.3f}".format(x)))
results_MAFLD_Male['lower_OR_ci'] = results_MAFLD_Male['lower_OR_ci'].apply(lambda x: float("{:.3f}".format(x)))
results_MAFLD_Male['upper_OR_ci'] = results_MAFLD_Male['upper_OR_ci'].apply(lambda x: float("{:.3f}".format(x)))

results_MAFLD_Male["Outcome"] = results_MAFLD_Male["Outcome"].replace({
    "NAFLD_baseline_diagnosis" : "MAFLD_Based_on_medication",
    "MAFLD_baseline_diagnosis" : "MAFLD_Based_on_speed_of_sound"
})

# Female
results_MAFLD_Female['OR'] = results_MAFLD_Female['OR'].apply(lambda x: float("{:.3f}".format(x)))
results_MAFLD_Female['lower_OR_ci'] = results_MAFLD_Female['lower_OR_ci'].apply(lambda x: float("{:.3f}".format(x)))
results_MAFLD_Female['upper_OR_ci'] = results_MAFLD_Female['upper_OR_ci'].apply(lambda x: float("{:.3f}".format(x)))
results_MAFLD_Female["Outcome"] = results_MAFLD_Female["Outcome"].replace({
    "NAFLD_baseline_diagnosis" : "MAFLD_Based_on_medication",
    "MAFLD_baseline_diagnosis" : "MAFLD_Based_on_speed_of_sound"
})


# In[38]:


results_MAFLD_Male


# In[39]:


results_MAFLD_Female



# ============================================================
Source: 05_Logistic_regression_MASLD_Categorical_DP.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ### Logistic regression model-MASLD
# #### Keyong Deng, Clinical Epidemiology,LUMC
# Fit a logistic regression model containing additive effects for all covariates.  
# The `GLM` function fits many types of generalized linear models (GLMs).  
# Choosing the `Binomial` family makes it a logistic regression.

# In[1]:


# import python modules
import pandas as pd


# In[2]:


# The function to perform the scaling of pandas columns
def std_columns(df, start_col = 2, end_col = 10, lastfix = "_scaled"):
    
    result_df = df.copy()
    cols_select = df.columns[start_col:end_col+1]

    for cols in cols_select:
        mean = df[cols].mean()
        std = df[cols].std()
        result_df[f"{cols}{lastfix}"] = (df[cols] - mean)/ std

    return result_df


# In[3]:


# Read in the DPs
DPs = pd.read_csv("../01_DPs_Diet_patterns/DPs_Final_Score_with_Category_v2.csv")
DPs.head()

# Read in the alcohol intake
Nutrients = pd.read_csv("/home/ec2-user/studies/ruifang/Food log.csv")
Nutrients['RegistrationCode'] = Nutrients.reset_index().RegistrationCode.str.split("_", expand = True).iloc[:,1]
Nutrients.rename(columns={"RegistrationCode": "participant_id"}, inplace=True)

Nutrients_2 = Nutrients[['participant_id', "alcohol_g"]].copy()
Nutrients_2['participant_id'] =  pd.to_numeric(Nutrients_2['participant_id'])
Nutrients_2.shape

# Age update in the V2
lifestyles = pd.read_csv("../02_Lifestyle factors/lifestyle_factor_all_disease_V2.csv")

lifestyles = pd.merge(lifestyles, Nutrients_2, on = "participant_id", how = "left")
lifestyles.head()


# In[4]:


DPs.head()


# In[5]:


DPs_scale = std_columns(DPs, start_col=1, end_col=5)
# DPs_scale.to_csv("./Data/DPs_scale_v2.csv", index = False)


# In[6]:


merge_df = pd.merge(DPs_scale, lifestyles, on = "participant_id", how = "left")
merge_df


# In[7]:


display(merge_df["bmi"].describe())

merge_df["bmi_cat"] = merge_df["bmi"].apply(
    lambda x: 0 if x < 25 else 1
).astype("category")
merge_df["bmi_cat"].value_counts()


# ## Logistic regression model without considering interaction

# In[8]:


get_ipython().system('pip install statsmodels')


# In[9]:


import statsmodels.api as sm

# Logistic regression model without interaction
def multiple_glm_model (data, 
                        dependent_var, 
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


# ## Logistic regression model with interaction

# In[10]:


# Make another function to include interactions in the model
def multiple_glm_model_2 (data, 
                          dependent_var, 
                          covariate_sets, 
                          vars_of_interest = None, 
                          include_interactions = None,
                          family = sm.families.Binomial()):
    models = {}

    for model_name, covariates in covariate_sets.items():
        # start with base covariates
        formula_term = covariates.copy()

        # add only specific interaction
        if vars_of_interest and include_interactions:
            for interaction in include_interactions:
                if interaction not in formula_term:
                    formula_term.append(interaction)

        formula = f"{dependent_var} ~ {'+'.join(formula_term)}"
        print(f'Fitting {model_name} with formula : {formula}')

        # fit the model (remove missing)
        model = sm.GLM.from_formula(formula, family = family, data = data, missing = "drop")
        result = model.fit()
        models[model_name] = result
    return models


# ## Non-linear logistic regression model
# - knots put at 10th, 50th and 90th percentiles of the dietary pattern scores

# In[11]:


# Consider the non-linear associations between exposure and outcome
# By adding Splines into the function
from patsy import dmatrix

def multiple_glm_model_nonlinear (data, dependent_var, 
                        covariate_sets, 
                        nonlinear_vars = None, 
                        family = sm.families.Binomial()):
    models = {}
    # covariate_sets (is a dictionary)

    for model_name, covariates in covariate_sets.items():

        # linear terms
        linear_vars = '+'.join(covariates)

        # Nonlinear term
        nonlinear_terms = ""
        if nonlinear_vars:
            nonlinear_part = [f"cc({var}, df = {df})" for var, df in nonlinear_vars.items()]
            nonlinear_terms = "+" + "+".join(nonlinear_part)
        
        # formula = f"{dependent_var} ~ {'+'.join(covariates)}"
        # formula = f'{dependent_var} ~ cc({exposure}, df = 4) + {' + '.join(covariates)}'
        formula = f"{dependent_var} ~ {linear_vars}{nonlinear_terms}"
        
        print(f"Fitting {model_name} with formula: {formula}")

        # fit the model (remove missing)
        model = sm.GLM.from_formula(formula, family = family, data = data, missing = "drop")

        # results
        result = model.fit()

        # store the results
        # models[model_name] = result.summary()
        models[model_name] = result
        
    return models


# ## Preparing the dataset for logisitic regression model

# In[12]:


# Based on the medication diagnosis (self-reported, ICD-11)
df_NAFLD = pd.read_csv("../05_Medication_status/Medication_baseline.csv")
merge_df_NAFLD = pd.merge(merge_df, df_NAFLD[["participant_id", 
                                              "NAFLD_baseline_diagnosis"]],
                          on = "participant_id", how = "left")

merge_df_NAFLD["NAFLD_baseline_diagnosis"] = merge_df_NAFLD["NAFLD_baseline_diagnosis"].map({"Yes":1, "No": 0})
display(merge_df_NAFLD["NAFLD_baseline_diagnosis"].value_counts(dropna = False))

# Based on another definition of MAFLD(speed of sound) + ICD-11
df_MAFLD = pd.read_csv("../05_Medication_status/MAFLD_baseline.csv")
merge_df_MAFLD = pd.merge(merge_df, df_MAFLD[["participant_id", "mafld_diagnosed"]],
                          on = "participant_id", how = "left")

merge_df_MAFLD["MAFLD_baseline_diagnosis"] = merge_df_MAFLD["mafld_diagnosed"].map({"Yes":1, "No": 0})
merge_df_MAFLD["MAFLD_baseline_diagnosis"].value_counts(dropna = False)


# In[13]:


# Reorder smokeing status
merge_df_NAFLD["smoking_status"] = pd.Categorical(
    merge_df_NAFLD["smoking_status"],
    categories =  ["Never", "Former", "Current"],
    ordered = True
)


# In[14]:


# Reorder smokeing status
merge_df_MAFLD["smoking_status"] = pd.Categorical(
    merge_df_MAFLD["smoking_status"],
    categories =  ["Never", "Former", "Current"],
    ordered = True
)


# In[15]:


print(merge_df_MAFLD.columns[merge_df_MAFLD.columns.str.contains(f'age')])
merge_df_MAFLD["age_all"].isnull().sum()


# In[16]:


# merge_df_MAFLD["age_at_research_stage"].isnull().sum()
# merge_df_MAFLD["age"].isnull().sum()


# In[17]:


covariate_sets_test_AHEI = {"model1": ["AHEI_2010_score_eadj_scaled"],
                    # model2
                       "model2": ["AHEI_2010_score_eadj_scaled", "age_all", 
                             "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"],
                    # model3  
                       "model3": ["AHEI_2010_score_eadj_scaled", "age_all", "bmi",
                             "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use"],
                    # model4 
                       "model4": ["AHEI_2010_score_eadj_scaled", "age_all",
                                 "sex_x",'smoking_status', 'edu_status', "sleep_hours_daily", "MET_hour", "Vit_use", "Hormone_use", 
                                  'NSAID_use', 'CVD_f_history', 'T2D_f_history']}

## Define a function to modify the element in each list and return a new dictionary
## -- Then we get the covariates for the other dietary pattern scores
def modify_dict(original, index, new_value, new_element = None):
    
    new_dict = {}
    for key, value_list in original.items():
        new_list = value_list.copy()
        
        # change the elements in the list by new value
        new_list[index] = new_value
        
        if key not in ["model1"]:
            if new_element is not None:
                new_list.append(new_element)
                
        # re-assign back       
        new_dict[key] = new_list
    return new_dict

covariate_sets_test_rDII = modify_dict(covariate_sets_test_AHEI, 0, "rDII_score_eadj_scaled")
covariate_sets_test_AMED = modify_dict(covariate_sets_test_AHEI, 0, "AMED_score_eadj_scaled")
covariate_sets_test_rEDIH = modify_dict(covariate_sets_test_AHEI, 0, "rEDIH_score_all_eadj_scaled")
covariate_sets_test_hPDI = modify_dict(covariate_sets_test_AHEI, 0, "hPDI_score_eadj_scaled", new_element= "alcohol_g")


# In[18]:


# Only select model2
covariate_sets_test_AHEI_model2 = {"model2" : covariate_sets_test_AHEI["model2"]}
covariate_sets_test_rDII_model2 = {"model2" : covariate_sets_test_rDII["model2"]}
covariate_sets_test_AMED_model2 = {"model2" : covariate_sets_test_AMED["model2"]}
covariate_sets_test_rEDIH_model2 = {"model2" : covariate_sets_test_rEDIH["model2"]}
covariate_sets_test_hPDI_model2 = {"model2" : covariate_sets_test_hPDI["model2"]}


# ## 1. Run the interaction for the model2 (Diet intereact with Sex)

# In[19]:


display(merge_df_MAFLD["MAFLD_baseline_diagnosis"].value_counts(dropna = False))
merge_df_NAFLD["NAFLD_baseline_diagnosis"].value_counts(dropna = False)


# In[20]:


# merge_df_NAFLD["sex_x"].value_counts()
# pd.crosstab(merge_df_NAFLD["sex_x"], merge_df_NAFLD["NAFLD_baseline_diagnosis"])
# pd.crosstab(merge_df_NAFLD["sex_x"], merge_df_NAFLD["smoking_status"])

display(pd.crosstab(merge_df_NAFLD["smoking_status"], merge_df_NAFLD["NAFLD_baseline_diagnosis"], dropna = False))
pd.crosstab(merge_df_MAFLD["smoking_status"], merge_df_MAFLD["MAFLD_baseline_diagnosis"], dropna = False)


# In[21]:


# Define diet quality indices and their parameters
diet_indices = {
    "AHEI" : {
        "covariate_set" : covariate_sets_test_AHEI_model2,
        "score_var" : "AHEI_2010_score_eadj_scaled"
    },
    "rDII" : {
        "covariate_set" : covariate_sets_test_rDII_model2,
        "score_var" : "rDII_score_eadj_scaled"
    },
    "AMED" : {
        "covariate_set" : covariate_sets_test_AMED_model2,
        "score_var" : "AMED_score_eadj_scaled"
    },
    "rEDIH" : {
        "covariate_set" : covariate_sets_test_rEDIH_model2,
        "score_var" : "rEDIH_score_all_eadj_scaled"
    },
    "hPDI" : {
        "covariate_set" : covariate_sets_test_hPDI_model2,
        "score_var" : "hPDI_score_eadj_scaled"
    } 
}

# Run all the method and create results (MAFLD)
results_dfs_MAFLD = {}

for name, params in diet_indices.items():
    model = multiple_glm_model_2(
        data = merge_df_MAFLD,
        dependent_var = "MAFLD_baseline_diagnosis",
        covariate_sets = params["covariate_set"],
        vars_of_interest= [params["score_var"]],
        include_interactions= [f'{params["score_var"]}:sex_x'])

    results_dfs_MAFLD[name] = pd.DataFrame(
        {
        "Coefficient": model["model2"].params,
        "Con_interval_low": model["model2"].conf_int()[0],
        "Con_interval_upper": model["model2"].conf_int()[1],
        "p_value": model["model2"].pvalues   
        }
    )

# Combine all the result into one data frame
results_combine_MAFLD = pd.concat(
    [df for name, df in results_dfs_MAFLD.items()],
    ignore_index= False
)

# results_combine
display(results_combine_MAFLD[results_combine_MAFLD.index.str.contains(f"scaled:sex_x")])


# In[22]:


# Run all the method and create results (NAFLD)
results_dfs_NAFLD = {}

for name, params in diet_indices.items():
    model = multiple_glm_model_2(
        data = merge_df_NAFLD,
        dependent_var = "NAFLD_baseline_diagnosis",
        covariate_sets = params["covariate_set"],
        vars_of_interest= [params["score_var"]],
        include_interactions= [f'{params["score_var"]}:sex_x'])

    results_dfs_NAFLD[name] = pd.DataFrame(
        {
        "Coefficient": model["model2"].params,
        "Con_interval_low": model["model2"].conf_int()[0],
        "Con_interval_upper": model["model2"].conf_int()[1],
        "p_value": model["model2"].pvalues   
        }
    )

# Combine all the result into one data frame
results_combine_NAFLD = pd.concat(
    [df for name, df in results_dfs_NAFLD.items()],
    ignore_index= False
)

# results_combine
display(results_combine_NAFLD[results_combine_NAFLD.index.str.contains(f"scaled:sex_x")])


# ## 2. BMI interaction with score analysis
# 
# > Add the bmi_cat into the adjusted list

# In[23]:


covariate_sets_test_AHEI_model2["model2"] = ['AHEI_2010_score_eadj_scaled', 'age_all',
                                             'sex_x','smoking_status','edu_status',
                                             'sleep_hours_daily','MET_hour','Vit_use',
                                             'Hormone_use', 'bmi_cat']

covariate_sets_test_rDII_model2["model2"] = ['rDII_score_eadj_scaled', 'age_all',
                                             'sex_x','smoking_status','edu_status',
                                             'sleep_hours_daily','MET_hour','Vit_use',
                                             'Hormone_use', 'bmi_cat']

covariate_sets_test_AMED_model2["model2"] = ['AMED_score_eadj_scaled', 'age_all',
                                             'sex_x','smoking_status','edu_status',
                                             'sleep_hours_daily','MET_hour','Vit_use',
                                             'Hormone_use', 'bmi_cat']

covariate_sets_test_rEDIH_model2["model2"] = ['rEDIH_score_all_eadj_scaled', 'age_all',
                                             'sex_x','smoking_status','edu_status',
                                             'sleep_hours_daily','MET_hour','Vit_use',
                                             'Hormone_use', 'bmi_cat']

covariate_sets_test_hPDI_model2["model2"] = ['hPDI_score_eadj_scaled', 'age_all',
                                             'sex_x','smoking_status','edu_status',
                                             'sleep_hours_daily','MET_hour','Vit_use',
                                             'Hormone_use', 'bmi_cat', "alcohol_g"]


# In[24]:


merge_df_MAFLD["bmi_cat"] = merge_df_MAFLD["bmi"].apply(
    lambda x: 0 if x < 25 else 1
).astype("category")

# print(merge_df_MAFLD["bmi_cat"].value_counts())
# temp1=  multiple_glm_model_2(data = merge_df_MAFLD, 
#                    dependent_var = "MAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_AHEI_model2,
#                    vars_of_interest = ["AHEI_2010_score_eadj_scaled"],
#                    include_interactions = ["AHEI_2010_score_eadj_scaled:bmi_cat"])

# Define diet quality indices and their parameters
diet_indices = {
    "AHEI" : {
        "covariate_set" : covariate_sets_test_AHEI_model2,
        "score_var" : "AHEI_2010_score_eadj_scaled"
    },
    "rDII" : {
        "covariate_set" : covariate_sets_test_rDII_model2,
        "score_var" : "rDII_score_eadj_scaled"
    },
    "AMED" : {
        "covariate_set" : covariate_sets_test_AMED_model2,
        "score_var" : "AMED_score_eadj_scaled"
    },
    "rEDIH" : {
        "covariate_set" : covariate_sets_test_rEDIH_model2,
        "score_var" : "rEDIH_score_all_eadj_scaled"
    },
    "hPDI" : {
        "covariate_set" : covariate_sets_test_hPDI_model2,
        "score_var" : "hPDI_score_eadj_scaled"
    } 
}

# Run all the method and create results (with BMI_cat interaction)
results_dfs_MAFLD = {}
for name, params in diet_indices.items():
    model = multiple_glm_model_2(
        data = merge_df_MAFLD,
        dependent_var = "MAFLD_baseline_diagnosis",
        covariate_sets = params["covariate_set"],
        vars_of_interest= [params["score_var"]],
        include_interactions= [f'{params["score_var"]}:bmi_cat'])

    results_dfs_MAFLD[name] = pd.DataFrame(
        {
        "Coefficient": model["model2"].params,
        "Con_interval_low": model["model2"].conf_int()[0],
        "Con_interval_upper": model["model2"].conf_int()[1],
        "p_value": model["model2"].pvalues   
        }
    )

# Combine all the result into one data frame
results_combine_MAFLD_BMI = pd.concat(
    [df for name, df in results_dfs_MAFLD.items()],
    ignore_index= False
)

# results_combine
display(results_combine_MAFLD_BMI[results_combine_MAFLD_BMI.index.str.contains(f"scaled:bmi_cat")])

# Case 2
# Run all the method and create results
results_dfs_NAFLD_BMI = {}
for name, params in diet_indices.items():
    model = multiple_glm_model_2(
        data = merge_df_NAFLD,
        dependent_var = "NAFLD_baseline_diagnosis",
        covariate_sets = params["covariate_set"],
        vars_of_interest= [params["score_var"]],
        include_interactions= [f'{params["score_var"]}:bmi_cat'])

    results_dfs_NAFLD_BMI[name] = pd.DataFrame(
        {
        "Coefficient": model["model2"].params,
        "Con_interval_low": model["model2"].conf_int()[0],
        "Con_interval_upper": model["model2"].conf_int()[1],
        "p_value": model["model2"].pvalues   
        }
    )

# Combine all the result into one data frame
results_combine_NAFLD_BMI = pd.concat(
    [df for name, df in results_dfs_NAFLD_BMI.items()],
    ignore_index= False
)
results_combine_NAFLD_BMI[results_combine_NAFLD_BMI.index.str.contains(f"scaled:bmi_cat")]


# > For rDII, there are modified effect varied by BMI or Sex

# # 3. Explore analysis 
# ## The non-linear association between dietary pattern scores and MAFLD risk

# In[25]:


covariate_sets_test_AHEI_model2["model2"] = ['AHEI_2010_score_eadj_scaled', 'age_all',
                                             'sex_x','smoking_status','edu_status',
                                             'sleep_hours_daily','MET_hour','Vit_use',
                                             'Hormone_use']

covariate_sets_test_rDII_model2["model2"] = ['rDII_score_eadj_scaled', 'age_all',
                                             'sex_x','smoking_status','edu_status',
                                             'sleep_hours_daily','MET_hour','Vit_use',
                                             'Hormone_use']

covariate_sets_test_AMED_model2["model2"] = ['AMED_score_eadj_scaled', 'age_all',
                                             'sex_x','smoking_status','edu_status',
                                             'sleep_hours_daily','MET_hour','Vit_use',
                                             'Hormone_use']

covariate_sets_test_rEDIH_model2["model2"] = ['rEDIH_score_all_eadj_scaled', 'age_all',
                                             'sex_x','smoking_status','edu_status',
                                             'sleep_hours_daily','MET_hour','Vit_use',
                                             'Hormone_use']

covariate_sets_test_hPDI_model2["model2"] = ['hPDI_score_eadj_scaled', 'age_all',
                                             'sex_x','smoking_status','edu_status',
                                             'sleep_hours_daily','MET_hour','Vit_use',
                                             'Hormone_use', "alcohol_g"]


# In[26]:


# from statsmodels.stats.anova import anova_glm
# rDII
# temp3_1 = multiple_glm_model(data = merge_df_MAFLD,
#                             dependent_var="MAFLD_baseline_diagnosis",
#                             covariate_sets= covariate_sets_test_rDII_model2)
temp1_2 =  multiple_glm_model_nonlinear(data = merge_df_MAFLD, 
                   dependent_var = "MAFLD_baseline_diagnosis",
                   covariate_sets = covariate_sets_test_AHEI_model2,
                   nonlinear_vars = {"AHEI_2010_score_eadj_scaled" : 3}            
                   )

temp2_2 =  multiple_glm_model_nonlinear(data = merge_df_MAFLD, 
                   dependent_var = "MAFLD_baseline_diagnosis",
                   covariate_sets = covariate_sets_test_rDII_model2,
                   nonlinear_vars = {"rDII_score_eadj_scaled" : 3}            
                   )

temp3_2 =  multiple_glm_model_nonlinear(data = merge_df_MAFLD, 
                   dependent_var = "MAFLD_baseline_diagnosis",
                   covariate_sets = covariate_sets_test_AMED_model2,
                   nonlinear_vars = {"AMED_score_eadj_scaled" : 3}            
                   )

temp4_2 =  multiple_glm_model_nonlinear(data = merge_df_MAFLD, 
                   dependent_var = "MAFLD_baseline_diagnosis",
                   covariate_sets = covariate_sets_test_rEDIH_model2,
                   nonlinear_vars = {"rEDIH_score_all_eadj_scaled" : 3}            
                   )

temp5_2 =  multiple_glm_model_nonlinear(data = merge_df_MAFLD, 
                   dependent_var = "MAFLD_baseline_diagnosis",
                   covariate_sets = covariate_sets_test_hPDI_model2,
                   nonlinear_vars = {"hPDI_score_eadj_scaled" : 3}            
                   )


# In[27]:


temp5_1 = multiple_glm_model(data = merge_df_MAFLD,
                            dependent_var="MAFLD_baseline_diagnosis",
                            covariate_sets= covariate_sets_test_hPDI_model2)

# Likelihood ratio test for the non-linearity
from scipy.stats import chi2
# lr_stat = 2 * (temp5_2["model2"].llf - temp5_1["model2"].llf)
# print(lr_stat)
# degree_freedom = temp5_2["model2"].df_resid - temp5_1["model2"].df_resid
# print(degree_freedom)
# p_nonlinear = 1 - chi2.cdf(lr_stat, degree_freedom)

lr_stat = -2 * (temp5_1["model2"].llf - temp5_2["model2"].llf)
degree_freedom = len( temp5_2["model2"].params) - len(temp5_1["model2"].params)
print(degree_freedom)
p_nonlinear = 1 - chi2.cdf(lr_stat, degree_freedom)
print(f"P for non-linearity: {p_nonlinear: .3f}")


# In[28]:


# rEDIH
temp4_1 = multiple_glm_model(data = merge_df_MAFLD,
                            dependent_var="MAFLD_baseline_diagnosis",
                            covariate_sets= covariate_sets_test_rEDIH_model2)

lr_stat = -2 * (temp4_1["model2"].llf - temp4_2["model2"].llf)
degree_freedom = len( temp4_2["model2"].params) - len(temp4_1["model2"].params)
print(degree_freedom)
p_nonlinear = 1 - chi2.cdf(lr_stat, degree_freedom)
print(f"P for non-linearity: {p_nonlinear: .3f}")


# In[29]:


# AMED
temp3_1 = multiple_glm_model(data = merge_df_MAFLD,
                            dependent_var="MAFLD_baseline_diagnosis",
                            covariate_sets= covariate_sets_test_AMED_model2)

lr_stat = -2 * (temp3_1["model2"].llf - temp3_2["model2"].llf)
degree_freedom = len( temp3_2["model2"].params) - len(temp3_1["model2"].params)
print(degree_freedom)
p_nonlinear = 1 - chi2.cdf(lr_stat, degree_freedom)
print(f"P for non-linearity: {p_nonlinear: .3f}")


# In[30]:


# rDII
temp2_1 = multiple_glm_model(data = merge_df_MAFLD,
                            dependent_var="MAFLD_baseline_diagnosis",
                            covariate_sets= covariate_sets_test_rDII_model2)

lr_stat = -2 * (temp2_1["model2"].llf - temp2_2["model2"].llf)
degree_freedom = len( temp2_2["model2"].params) - len(temp2_1["model2"].params)
print(degree_freedom)
p_nonlinear = 1 - chi2.cdf(lr_stat, degree_freedom)
print(f"P for non-linearity: {p_nonlinear: .3f}")


# In[31]:


# AHEI
temp1_1 = multiple_glm_model(data = merge_df_MAFLD,
                            dependent_var="MAFLD_baseline_diagnosis",
                            covariate_sets= covariate_sets_test_AHEI_model2)

lr_stat = -2 * (temp1_1["model2"].llf - temp1_2["model2"].llf)
degree_freedom = len( temp1_2["model2"].params) - len(temp1_1["model2"].params)
print(degree_freedom)
p_nonlinear = 1 - chi2.cdf(lr_stat, degree_freedom)
print(f"P for non-linearity: {p_nonlinear: .3f}")


# ## Plot the RCS figure

# In[32]:


import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize = (18, 10))
axes = axes.flatten()

AHEI_range = np.linspace(
    merge_df_MAFLD["AHEI_2010_score_eadj_scaled"].min(),
    merge_df_MAFLD["AH

# ============================================================
Source: Function_Logistic_Model.py
# ============================================================

import statsmodels.api as sm

# Logistic regression model without interaction
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

# Make another function to include interactions in the model
def multiple_glm_model_2 (data, dependent_var, covariate_sets, 
                          vars_of_interest = None, 
                          include_interactions = None,
                          family = sm.families.Binomial()):
    models = {}

    for model_name, covariates in covariate_sets.items():
        # start with base covariates
        formula_term = covariates.copy()

        # add only specific interaction
        if vars_of_interest and include_interactions:
            for interaction in include_interactions:
                if interaction not in formula_term:
                    formula_term.append(interaction)

        formula = f"{dependent_var} ~ {'+'.join(formula_term)}"
        print(f'Fitting {model_name} with formula : {formula}')

        # fit the model (remove missing)
        model = sm.GLM.from_formula(formula, family = family, data = data, missing = "drop")
        result = model.fit()
        models[model_name] = result
    return models

# Consider the non-linear associations between exposure and outcome
# By adding Splines into the function
from patsy import dmatrix

def multiple_glm_model_nonlinear (data, dependent_var, 
                        covariate_sets, 
                        nonlinear_vars = None, 
                        family = sm.families.Binomial()):
    models = {}
    # covariate_sets (is a dictionary)

    for model_name, covariates in covariate_sets.items():

        # linear terms
        linear_vars = '+'.join(covariates)

        # Nonlinear term
        nonlinear_terms = ""
        if nonlinear_vars:
            nonlinear_part = [f"cc({var}, df = {df})" for var, df in nonlinear_vars.items()]
            nonlinear_terms = "+" + "+".join(nonlinear_part)
        
        # formula = f"{dependent_var} ~ {'+'.join(covariates)}"
        # formula = f'{dependent_var} ~ cc({exposure}, df = 4) + {' + '.join(covariates)}'
        formula = f"{dependent_var} ~ {linear_vars}{nonlinear_terms}"
        
        print(f"Fitting {model_name} with formula: {formula}")

        # fit the model (remove missing)
        model = sm.GLM.from_formula(formula, family = family, data = data, missing = "drop")

        # results
        result = model.fit()

        # store the results
        # models[model_name] = result.summary()
        models[model_name] = result
        
    return models

def logist_model_df(data, dependent_var):
    dfs = []
    for covariates in [covariate_sets_test_AHEI, 
                       covariate_sets_test_rDII, 
                       covariate_sets_test_AMED, 
                       covariate_sets_test_rEDIH,
                  covariate_sets_test_hPDI]:
        
        models_test = multiple_glm_model(data = data, 
                                         dependent_var = dependent_var, 
                                         covariate_sets = covariates)
        
        Exposure = covariates.get("model1", [])[0]
        
        # Also combine the other Dataframes: model1, model2, model3...
        for model in ["model1", "model2", "model3", "model4"]:
            coef = models_test[model].params[Exposure]
            pvalue = models_test[model].pvalues[Exposure]
            conf_int = models_test[model].conf_int(alpha = 0.05).loc[Exposure]
            lower_ci, upper_ci = conf_int
            
           # Create a new Pandas Dataframe
            result_all = pd.DataFrame({
                                "Model": [model],
                                "Exposure" : [Exposure],
                                "Outcome" : [dependent_var],
                                "Coef": [coef],
                                "pval" : [pvalue],
                                "lower_ci" : [lower_ci],
                                "upper_ci" : [upper_ci],
                                "OR": [np.exp(coef)],
                                "lower_OR_ci" : [np.exp(lower_ci)],
                                "upper_OR_ci" : [np.exp(upper_ci)]
                            })
            # dfs.append(result_all)
            dfs = dfs + [result_all]
            
    return dfs

# ## Back up code
# temp1=  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_AHEI_model2,
#                    vars_of_interest = ["AHEI_2010_score_eadj_scaled"],
#                    include_interactions = ["AHEI_2010_score_eadj_scaled:sex_x"])
# display(temp1["model2"].params)

# temp2=  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_rDII_model2,
#                    vars_of_interest = ["rDII_score_eadj_scaled"],
#                    include_interactions = ["rDII_score_eadj_scaled:sex_x"])

# display(temp2["model2"].params)

# temp3 =  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_AMED_model2,
#                    vars_of_interest = ["AMED_score_eadj_scaled"],
#                    include_interactions = ["AMED_score_eadj_scaled:sex_x"])
# display(temp3["model2"].params)

# temp4 =  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_rEDIH_model2,
#                    vars_of_interest = ["rEDIH_score_all_eadj_scaled"],
#                    include_interactions = ["rEDIH_score_all_eadj_scaled:sex_x"])
# display(temp4["model2"].params)

# temp5 =  multiple_glm_model_2(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                     covariate_sets = covariate_sets_test_hPDI_model2,
#                    vars_of_interest = ["hPDI_score_eadj_scaled"],
#                    include_interactions = ["hPDI_score_eadj_scaled:sex_x"])
# display(temp5["model2"].params)

# results_df1_NAFLD = pd.DataFrame(
#     {
#         "Coefficient": temp1["model2"].params,
#         "Con_interval_low": temp1["model2"].conf_int()[0],
#         "Con_interval_upper": temp1["model2"].conf_int()[1],
#         "p_value": temp1["model2"].pvalues
#     }
# )

# results_df2_NAFLD = pd.DataFrame(
#     {
#         "Coefficient": temp2["model2"].params,
#         "Con_interval_low": temp2["model2"].conf_int()[0],
#         "Con_interval_upper": temp2["model2"].conf_int()[1],
#         "p_value": temp2["model2"].pvalues
#     }
# )

# results_df3_NAFLD = pd.DataFrame(
#     {
#         "Coefficient": temp3["model2"].params,
#         "Con_interval_low": temp3["model2"].conf_int()[0],
#         "Con_interval_upper": temp3["model2"].conf_int()[1],
#         "p_value": temp3["model2"].pvalues
#     }
# )

# results_df4_NAFLD = pd.DataFrame(
#     {
#         "Coefficient": temp4["model2"].params,
#         "Con_interval_low": temp4["model2"].conf_int()[0],
#         "Con_interval_upper": temp4["model2"].conf_int()[1],
#         "p_value": temp4["model2"].pvalues
#     }
# )

# results_df5_NAFLD = pd.DataFrame(
#     {
#         "Coefficient": temp5["model2"].params,
#         "Con_interval_low": temp5["model2"].conf_int()[0],
#         "Con_interval_upper": temp5["model2"].conf_int()[1],
#         "p_value": temp5["model2"].pvalues
#     }
# )

# combined_results_NAFLD = pd.concat([results_df1_NAFLD, results_df2_NAFLD, results_df3_NAFLD, results_df4_NAFLD, results_df5_NAFLD], ignore_index=False)
# combined_results_NAFLD

# # Check the interaction p value 
# combined_results_NAFLD[combined_results_NAFLD.index.str.contains(f"scaled:sex_x")]

# # Backup code
# temp4_1 = multiple_glm_model(data = merge_df_NAFLD,
#                             dependent_var="NAFLD_baseline_diagnosis",
#                             covariate_sets= covariate_sets_test_AHEI_model2)

# temp4_2 =  multiple_glm_model_nonlinear(data = merge_df_NAFLD, 
#                    dependent_var = "NAFLD_baseline_diagnosis",
#                    covariate_sets = covariate_sets_test_AHEI_model2,
#                    nonlinear_vars = {"AHEI_2010_score_eadj_scaled" : 4}            
#                    )

# print(temp4_1["model2"].params)
# print(temp4_2["model2"].pvalues)

# # Compare the two models
# from scipy.stats import chi2

# # Likelihood ratio test
# lr_statistic = 2 * (temp3_1["model2"].llf - temp3_2["model2"].llf)
# lr_statistic
# p_value = 1 - chi2.cdf(lr_statistic, df = 3)
# p_value

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