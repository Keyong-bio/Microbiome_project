
# ============================================================
Source: 01_AHEI_2010_v2.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Calculate the AHEI in 10K
# -----------
# Keyong Deng,
# Leiden University Medical Center 
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# - Change of metabolites and protein and disease outcome (another paper)
# - Baseline of the metabolites and proteins (one paper)
# (2024-11-11)

# In[18]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from pheno_utils import PhenoLoader


# In[19]:


Nutrients = pd.read_csv("./studies/ruifang/Food log.csv")
display(Nutrients.columns)


# In[20]:


# Define the columns we want to subtract 
fats_to_subtract = [
    'Total daily MUFA (g)',
    'Total daily PUFA (g)', 
    'totalsaturatedfattyacids_g'
]

# Calculate trans fat in one operation
Nutrients['trans_fat'] = Nutrients['totallipid_g'] - Nutrients[fats_to_subtract].sum(axis=1)
Nutrients['trans_fat%'] = ((Nutrients['trans_fat']*9)/Nutrients['energy_kcal']) * 100
Nutrients['PUFA%'] = ((Nutrients['Total daily PUFA (g)']*9)/Nutrients['energy_kcal']) * 100


# In[21]:


# kJ = kcal × 4.184
import time

print(
    f'{time.strftime("%Y-%m-%d %H:%M:%S")}'
)  


# In[22]:


vars_AHEI = [
    'RegistrationCode',
    'energy_kcal',
    'sodium_mg',
    # 'Total daily PUFA (g)',
    'PUFA%',
    # 'trans_fat',
    'trans_fat%',
    'alcohol_g'
]

df_AHEI1 = Nutrients[vars_AHEI]
df_AHEI1

# split the RegistrationCode, expand = True gives dataframe
df_AHEI1[['Cohort','Participand_ID']] = df_AHEI1['RegistrationCode'].str.split('_', n=1, expand=True)
df_AHEI1 = df_AHEI1.drop(columns = ['RegistrationCode','Cohort'],
                         # inplace = True
                        )
df_AHEI1

# Move last column to first position
df_AHEI1.insert(0, 'Participand_ID', df_AHEI1.pop('Participand_ID'))
df_AHEI1


# In[23]:


df_AHEI1.loc[(df_AHEI1["energy_kcal"] > 5000) | (df_AHEI1["energy_kcal"] < 600)]


# In[24]:


# including other elements
# Vegetables (expect potatoes and legume); fruit, grain, Nuts/legumes/vege protein; ssb+friut juice; Red/proessed meat
import os
from pheno_utils.config import (
    DATASETS_PATH, 
    )

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


# In[25]:


temp1 = events_df_base_noNA1['participant_id'].unique()
temp1

# All ID in df_AHEI1 also in the diet logging data
sum(df_AHEI1['Participand_ID'].isin(temp1))


# ### Food names

# In[26]:


food_name = events_df_base_noNA1["short_food_name"]


# ## Recode the food groups

# In[27]:


# create the condition 
condition = (events_df_base_noNA1['food_category'] == "Fruits") & (events_df_base_noNA1['short_food_name'].isna())
# condition.value_counts() # 1760

print(condition.dtype)

# Fill NA value with Fruit_item when condition is True
events_df_base_noNA1.loc[condition, 'short_food_name'] = "Fruit_item"


# In[28]:


# 1. Fruit
Fruit_item = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Fruits"]["short_food_name"].unique().tolist()

juice_items = [item for item in Fruit_item if pd.notna(item) and 'juice' in item.lower()]
print(juice_items)

def is_berry(item):
    if pd.notna(item):
        item_lower = item.lower()
        return 'berry' in item_lower or 'berries' in item_lower
    return False

# exclude the juice items
Fruit_item2 = pd.Series(Fruit_item)[~ pd.Series(Fruit_item).isin(juice_items)]
Fruit_item2

# including the canned veg
Canned_fruit = [
    'Prune',
    'Apricot', 'Dried Fruit', 'Cooked Raisins', 'Fruit salad', 'Cherries', 'Dried apple', 'Dried pineapple', 'Dried cranberries', 'Dried fig',
    'Goji berry', 'Dried Mango', 'Dried blueberries', 'Dates', 'Palm', 'Applesauce'
]

overlap = [x for x in Fruit_item2 if x in Canned_fruit]
overlap

# len(list(set(Fruit_item2) | set(Canned_fruit))) # 54

Fruit_item3 = list(set(Fruit_item2) | set(Canned_fruit))
Fruit_item3

Berries_items = [item for item in Fruit_item3 if is_berry(item)]
Berries_items

Fruit_item4 = [item for item in Fruit_item3 if not is_berry(item)]
Fruit_item4


# In[29]:


# create the condition 
condition2 = (events_df_base_noNA1['food_category'] == "Vegetables") & (events_df_base_noNA1['short_food_name'].isna())
# condition.value_counts() # 1760

display(events_df_base_noNA1[condition2])

# Fill NA value with Fruit_item when condition is True
events_df_base_noNA1.loc[condition2, 'short_food_name'] = "Vegetables_item"


# In[30]:


# 2. Vegetables
Vegetable_item = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Vegetables"]["short_food_name"].dropna().unique()
Vegetable_item.tolist()


# In[31]:


# excluding Legumes
Legume_item = food_name[food_name.str.contains(r'\bbeans?|Lentil?|Peas|pea\b|tofu\b', case=False).fillna(False)].unique()      
Legume_item

# exclude the legumes item in the vegetable item
Vegetable_item2 = Vegetable_item[~ Vegetable_item.isin(Legume_item)] # exclude the legumes
Vegetable_item3 = Vegetable_item2[~ Vegetable_item2.isin(Fruit_item)] # exclude the fruits (miclassification into the vegetable group)

# Vegetable_item2[Vegetable_item2.isin(Fruit_item)] # cranberries, Mango, Lemon are fruit, not vegetables

# exclude the potatoes (including french fries)
potatoes = pd.Series(food_name[(food_name.str.contains(r'potato(es)?\b', case=False).fillna(False))].unique())

fried_foods = pd.Series(['Sweet Potato Fries', 'French fries'])

Combined_potatoes = pd.concat([potatoes, fried_foods]).drop_duplicates()
Combined_potatoes

# exclude the potatoes 
Vegetable_item4 = Vegetable_item3[~Vegetable_item3.isin(Combined_potatoes)] 
Vegetable_item4

Canned_veg_fruit = events_df_base_noNA1[(events_df_base_noNA1['food_category'] == "Canned veg and fruits")]["short_food_name"].dropna().unique()

Canned_vegetables = events_df_base_noNA1[
    (events_df_base_noNA1['food_category'] == "Canned veg and fruits") & 
    (~ events_df_base_noNA1['short_food_name'].isin(Canned_fruit))
]["short_food_name"].dropna().unique()

print(Canned_vegetables)

# Convert to pandas Series first
Canned_vegetables = pd.Series(Canned_vegetables)

# Now we can use drop to remove the values
Canned_vegetables = Canned_vegetables[~ Canned_vegetables.isin(['Jam', 'Lemon juice', 'Tea'])]
Canned_vegetables = Canned_vegetables.tolist()
Canned_vegetables

Vegetable_item5 = pd.concat([pd.Series(Vegetable_item4), pd.Series(Canned_vegetables)])
Vegetable_item5


# In[32]:


# 3. Whole grains
whole_grains = food_name[(food_name.str.contains(r'\bOatmeal Cereal|Cereals|dark bread|brown rice|grain\b|bran\b|wheat?\b|popcorn\b', case=False).fillna(False)) & 
                (~ food_name.str.contains(r'\bcrackers\b', case=False).fillna(False))]
whole_grains_item = whole_grains.unique()

# fill in whole_grain item for the NA in Bread_wholewheat
condition3 = (events_df_base_noNA1['food_category'] == "Bread_wholewheat") & (events_df_base_noNA1['short_food_name'].isna())
# events_df_base_noNA1[condition3]['product_name'].iloc[0] # A yellow cheese sandwich with a full-flavored flour bite bun

events_df_base_noNA1.loc[condition3, 'short_food_name'] = "whole_grains"
Bread_wholewheat = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Bread_wholewheat"]["short_food_name"].dropna().unique()

# get the union whole grain items
whole_grains_item2 = list(set(whole_grains_item) | set(Bread_wholewheat))
whole_grains_item2


# In[33]:


# 4. Nuts/Legume/vege protein
condition4 = (events_df_base_noNA1['food_category'] == "Nuts, seeds, and products") & (events_df_base_noNA1['short_food_name'].isna())
# events_df_base_noNA1[condition3]['product_name'].iloc[0] # A yellow cheese sandwich with a full-flavored flour bite bun

condition4.value_counts()
events_df_base_noNA1.loc[condition4, 'short_food_name'] = "Nuts"


# In[34]:


Nut_item = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Nuts, seeds, and products"]["short_food_name"].dropna().unique()
Legume = food_name[food_name.str.contains(r'\bbeans?|Lentil?|Peas|pea\b|tofu\b', case=False).fillna(False)].unique()

# print(Nut_item)
# print(Legume)
Nut_legume = list(set(Nut_item) | set(Legume))
Nut_legume


# In[35]:


## important notation: change tea in the Proccessed  meat products category into tea_processed meat
events_df_base_noNA1.loc[(events_df_base_noNA1['short_food_name']=="Tea") & (events_df_base_noNA1['food_category'] == "Proccessed  meat products"), 'short_food_name'] = "Tea_processed_meat"
events_df_base_noNA1.loc[(events_df_base_noNA1['short_food_name']=="Tea_processed_meat") & (events_df_base_noNA1['food_category'] == "Proccessed  meat products")]


# In[36]:


# processed meat
condition6 = (events_df_base_noNA1['food_category'] == "Proccessed  meat products") & (events_df_base_noNA1['short_food_name'].isna())
print(condition6.value_counts()) # 2524 True

events_df_base_noNA1.loc[condition6, 'short_food_name'] = "Processed_meat"

Proccessed_meat = events_df_base_noNA1[events_df_base_noNA1['food_category']=="Proccessed  meat products"]['short_food_name'].unique()
print(Proccessed_meat) # Tea is good chicken tea sausage here 


# In[37]:


Proccessed_meat = ['Proecessed_meat', 'Pastrami', 'Sausages', 'Cold cut', 'Meatballs', 'Hamburger', "Tea_processed_meat"]
Proccessed_meat


# In[38]:


# 5. Red meat
condition5 = (events_df_base_noNA1['food_category'] == "Beef, veal, lamb, and other meat products") & (events_df_base_noNA1['short_food_name'].isna())
print(condition5.value_counts())

events_df_base_noNA1.loc[condition5, 'short_food_name'] = "Red_meat_product"

red_meat_product_item = events_df_base_noNA1[
    (events_df_base_noNA1['food_category'] == "Beef, veal, lamb, and other meat products") 
    # | (events_df_base_noNA1['food_category'] == "Processed meat products")
]["short_food_name"].unique()

red_meat_product_item = [items for items in red_meat_product_item if items not in Proccessed_meat]
red_meat_product_item


# In[39]:


# 6. Sugar-sweets Beves + fruit juices
ssb_item = food_name[food_name.str.contains(r'\bcoke\b|Lemonade|Smoothies|Sweetened Cocoa Powder|Flavored Waters|Fruit Drink|Fruit syrup|Salep|Raspberry syrup|juice', case=False).fillna(False)].unique()
ssb_item


# In[40]:


# 7. Fish
condition7 = (events_df_base_noNA1['food_category'] == "Fish and seafood") & (events_df_base_noNA1['short_food_name'].isna())
print(condition7.value_counts()) # 7081 True

events_df_base_noNA1.loc[condition7, 'short_food_name'] = "Fish_seafood"

fish_item = events_df_base_noNA1[(events_df_base_noNA1['food_category'] == "Fish and seafood")]["short_food_name"].dropna().unique()
fish_item


# In[41]:


def categorize_food_item(df, category, items):
    
    mapping_df = pd.DataFrame({
        'short_food_name': items,
        'category': [category] * len(items)
    })
    
    map_dict = mapping_df.set_index('short_food_name')['category'].to_dict()
    
    # another way to do
    # map_dict = dict(zip(mapping_df['short_food_name'], mapping_df['category']))
    
    df[category] = df['short_food_name'].map(map_dict)
    
    return df


# In[42]:


def process_food_categories(events_df: pd.DataFrame) -> pd.DataFrame:
    
    # Define all category mappings in a dictionary for easier maintenance
    food_categories = {
        
        'Vegetables_item': Vegetable_item5,
        'Fruits_item': Fruit_item4,
        'Whole_grains_item': whole_grains_item2,
        'Nut_legume_item': Nut_legume,
        'red_meat_product_item': red_meat_product_item,
        'sugar_sweetened_beverages_item': ssb_item,
        'fish_item': fish_item,
        'berries_item': Berries_items,
        "processed_meat_item": Proccessed_meat
        
    }
    
    # Process each category
    for category, items in food_categories.items():
        events_df = categorize_food_item(events_df, category, items)
    
    return events_df

events_df_base_noNA1 = process_food_categories(events_df_base_noNA1)


# In[43]:


# classify into beer(12ozÍ), wine(5oz), spirits (1.5oz)
# wine: Wine, Sweet wine, Dessert Wine, Shandy, other wines
# Beer: Beer, Light beer, 
# spirits: Vodka or Arak, Whiskey, Brandy, Gin and tonic

# Transforming Data by using a function or mapping
alcohol_to_cate = {
    "Wine": "wine",
    "Beer" : "beer",
    "Whiskey":"spirits",
    "Campari":"wine",
    "Vodka or Arak":"spirits",
    "Sweet wine":"wine",
    "Dessert Wine":"wine",
    "Light Beer":"beer",
    "Ouzo":"wine",
    "Aperol":"wine",
    "Brandy":"spirits",
    "Gin and tonic":"spirits",
    "Fruit Drink":"wine",
    "Cocktail":"wine",
    "Shandy":"wine",
    "Sangria":"wine"
}

def get_alcohol_cat(x):
    return alcohol_to_cate.get(x, np.nan)

events_df_base_noNA1['wine_cate_item'] = events_df_base_noNA1['short_food_name'].map(get_alcohol_cat)


# In[44]:


temp = events_df_base_noNA1.groupby(["participant_id", "wine_cate_item"])["weight_g"].sum().reset_index()
print(temp['participant_id'].nunique()) # 6152 individuals

# long to wide format
temp_wide = temp.pivot(index = "participant_id", columns = "wine_cate_item", values = "weight_g")


# In[45]:


def calculate_food_consumption(events_df):
    
    # Define food categories
    food_categories = {
        
        'fruits': 'Fruits_item',
        'vegetables': 'Vegetables_item',
        'meat': 'red_meat_product_item',
        'fish': 'fish_item',
        'nuts': 'Nut_legume_item',
        'grains': 'Whole_grains_item',
        'beverages': 'sugar_sweetened_beverages_item',
        'berry': 'berries_item',
        'processed_meat': 'processed_meat_item'
    }
    
    # Initialize dictionary to store results
    consumption_totals = {}
    
    # Calculate totals for each food category
    for category_key, category_name in food_categories.items():
        
        # filtered_df = events_df[events_df['food_category'] == category_name]
        consumption_totals[category_key] = (
            events_df
            .groupby(['participant_id', f'{category_name}'])['weight_g']
            .sum()
            .reset_index(name = f'total_{category_key}_g')
        )
    
    return consumption_totals

consumption_data = calculate_food_consumption(events_df_base_noNA1)
consumption_data

consumption_data_df = list(consumption_data.values())
consumption_data_df


# In[46]:


result = consumption_data_df[0]

# merge with each dataframe in the dictionary
for df in consumption_data_df[1:]:
        result = pd.merge(result, df, on="participant_id", how = "outer") # how = outer, include all observations


# In[47]:


result


# In[48]:


result2 = result.filter(regex='participant_id|_g').drop('Whole_grains_item', axis=1)
# results = result.filter(regex='^(participant_id|.*_g)$') # using the regex expression
result2


# In[49]:


# merge with alcohol consumption
alcohol_intake = temp_wide.reset_index()

result3 = pd.merge(result2, alcohol_intake, on="participant_id", how = "outer") # how = outer, include all observations
result3


# In[50]:


df_AHEI1.columns


# In[51]:


# combine two dataframe based on participant ID
# df_AHEI1 and result2
df_AHEI1.shape # (11922, 6)

df_AHEI1['Participand_ID'] = df_AHEI1['Participand_ID'].astype('int64')
result3['participant_id'] = result3['participant_id'].astype('int64')

df_AHEI2 = pd.merge(result3, df_AHEI1, left_on = "participant_id", right_on = "Participand_ID", how = "left").drop(['Participand_ID'], axis = 1)
df_AHEI2


# In[52]:


df_AHEI2.to_csv("AHEI3_2024_1125.csv", index=False) # export 


# ### Combining the logging days

# In[53]:


log_d = pd.read_csv('logging_days.csv')
log_d


# In[54]:


df_AHEI3 = pd.merge(df_AHEI2, log_d, left_on= "participant_id", right_on= "ID", how = "left").drop(['ID'], axis = 1)
df_AHEI3.to_csv("AHEI3_2024_1125_02.csv", index = False)


# ### Calculate daily intake

# In[55]:


# calculate the daily intake
# divide by the same column
# result = df.iloc[:, :3].div(df[3], axis=0)
import pandas as pd
df_AHEI3 = pd.read_csv("AHEI3_2024_1125_02.csv")

# Set the participant id as the index
df_AHEI3.set_index('participant_id', inplace=True)

df_AHEI3 = df_AHEI3.fillna(0)
df_AHEI3


# In[56]:


# change from grams into servings
# 4 oz of wine, 12 oz of beer, or 1.5 oz of liquor (1 oz = 28.35 g) -> 1 serving
servings_dict = {
    
    "total_vegetables_g": 236.95/2, # 0.5 cup green lefy 
    "total_fruits_g": 236.95, # 1 cup 
    "total_berry_g":236.95 * 0.5, # 0.5 cup
    "total_beverages_g":28.35 * 8, # 8 oz
    "total_nuts_g": 28.35, # 1 oz
    "total_processed_meat_g": 28.35 *1.5, # processed meat 
    "total_meat_g":28.35 * 4, # red meat
    # "total_fish_g":24*28.35/7 # is optimal intake 97.2g/d 
    "beer": 12*28.35, # 12 oz beer 
    "wine": 4*28.35, # 4 oz wine
    "spirits": 1.5*28.35, # 1.5 oz spirits
    "total_grains_g": 15 # 15g/serving (different use in the GNHS cohort)
    
}
servings_dict.items()


# In[57]:


# change from grams to servings
df_result2 = df_AHEI3.copy()
for col, divisor in servings_dict.items():
    if col in df_AHEI3.columns:  # Check if column exists
        df_result2[col] = df_AHEI3[col] / divisor
df_result2     


# In[58]:


col_select = ['total_fruits_g', "total_vegetables_g", "total_meat_g", "total_fish_g", "total_nuts_g", "total_grains_g", "total_beverages_g","total_berry_g", "total_processed_meat_g",
              "beer", "spirits", "wine"] # the unit sodium_mg is grams/day

# sodium should not be divide by log_day? (2024-12-03)??


# In[59]:


df_AHEI4 = df_result2.copy()
df_AHEI4.loc[:,col_select] = df_AHEI4.loc[:,col_select].div(df_result2['log_days'], axis = 0).fillna(0)
# df_AHEI4
df_AHEI4


# In[60]:


# sum specific groups into one variable
df_AHEI4['total_fruits_full_g'] = df_AHEI4['total_fruits_g'] + df_AHEI4['total_berry_g']
df_AHEI4['total_alcohol_full_g'] = df_AHEI4['beer'] + df_AHEI4['spirits'] + df_AHEI4['wine']
df_AHEI4['total_meat_full_g'] = df_AHEI4['total_meat_g'] + df_AHEI4['total_processed_meat_g']


# In[61]:


df_AHEI4.to_csv("AHEI3_2024_1125_final.csv")


# ### Assign scores

# In[62]:


import pandas as pd
import numpy as np
df_AHEI4 = pd.read_csv("AHEI3_2024_1125_final.csv")
df_AHEI4


# In[63]:


df_AHEI4.describe()


# In[64]:


col_selected = ['participant_id', 'total_fruits_full_g', 'total_vegetables_g', 'total_meat_g', 'total_fish_g', 'total_nuts_g', 'total_grains_g',
               'total_beverages_g', 'total_processed_meat_g', 'total_alcohol_full_g', 'total_meat_full_g', 'sodium_mg', 'PUFA%', 'trans_fat%']
df_AHEI5 = df_AHEI4[col_selected]
df_AHEI5


# In[65]:


# thinking about the outlier for each food component consumption
def replace_outlier(df):
    df_copy = df.copy()
    
    # columns 
    cols = df_copy.columns[1:]
    for col in cols:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        lower = Q1 - 3*(Q3-Q1)
        upper = Q3 + 3*(Q3-Q1)
        df_copy.loc[df_copy[col] > upper, col] = upper
        df_copy.loc[df_copy[col] < lower, col] = lower
        
    return df_copy

# df_AHEI5 = replace_outlier(df = df_AHEI5)


# In[66]:


# def score_healthy(actual_serv, min_serv, max_serv, min_score, max_score):
#     if  (actual_serv <= min_serv).all():
#         return min_score
#     elif (actual_serv >= max_serv).all():
#         return max_score
#     else:
#         return min_score + (actual_serv - min_serv) * max_score / (max_serv - min_serv)

# def score_unhealthy(actual_serv, min_serv, max_serv, min_score, max_score):
#     if (actual_serv >= min_serv).all():
#         return min_score
#     elif (actual_serv <= max_serv).all():
#         return max_score
#     else:
# #         return min_score + (actual_serv - min_serv) * max_score / (max_serv - min_serv)


# In[67]:


df_AHEI5.columns


# In[68]:


df_AHEI5['total_fruits_full_g'].describe()


# In[69]:


# healthy food
def score_healthy(actual_serv, min_serv, max_serv, min_score, max_score):
    
    # Convert input to numpy array for vectorized operations
    actual_serv = np.array(actual_serv)
    
    # Initialize scores array with the default calculation
    scores = min_score + (actual_serv - min_serv) * max_score / (max_serv - min_serv)
    
    # Apply threshold conditions
    scores = np.where(actual_serv <= min_serv, min_score, scores)
    scores = np.where(actual_serv >= max_serv, max_score, scores)
    
    return scores

df_AHEI5['Fruit_score'] = score_healthy(df_AHEI5['total_fruits_full_g'], min_serv=0, max_serv=4, min_score=0, max_score=10)
df_AHEI5['vegetable_score'] = score_healthy(df_AHEI5['total_vegetables_g'], min_serv=0, max_serv=5, min_score=0, max_score=10)
df_AHEI5['nuts_score'] = score_healthy(df_AHEI5['total_nuts_g'], min_serv=0, max_serv=1, min_score=0, max_score=10)
df_AHEI5['fish_score'] = score_healthy(df_AHEI5['total_fish_g'], min_serv=0, max_serv=24*28.35/7, min_score=0, max_score=10)
df_AHEI5['PUFA_score'] = score_healthy(df_AHEI5['PUFA%'], min_serv=0, max_serv=10, min_score=0, max_score=10)


# In[70]:


# healthy food
def score_unhealthy(actual_serv, min_serv, max_serv, min_score, max_score):
    
    # Convert input to numpy array for vectorized operations
    actual_serv = np.array(actual_serv)
    
    # Initialize scores array with the default calculation
    scores = min_score + (actual_serv - min_serv) * max_score / (max_serv - min_serv)
    
    # Apply threshold conditions
    scores = np.where(actual_serv >= min_serv, min_score, scores)
    scores = np.where(actual_serv <= max_serv, max_score, scores)
    
    return scores

# unhealthy food
df_AHEI5['ssb_score'] = score_unhealthy(df_AHEI5['total_beverages_g'], min_serv=1, max_serv=0, min_score=0, max_score=10)
# display(df_AHEI4['total_beverages_g'])
# df_AHEI4['ssb_score']
df_AHEI5['meat_score'] = score_unhealthy(df_AHEI5['total_meat_full_g'], min_serv=1.5, max_serv=0, min_score=0, max_score=10)
# trans-fat
df_AHEI5['trans_score'] = score_unhealthy(df_AHEI5['trans_fat%'], min_serv=4, max_serv=0.5, min_score=0, max_score=10)


# In[71]:


df_AHEI5.columns


# In[73]:


# merge with sex information
age_sex_bmi = pd.read_csv("age_sex_bmi.csv")
age_sex_bmi


# In[74]:


age_sex_bmi = age_sex_bmi[['participant_id', 'sex_x', 'bmi', 'age', 'sex']] # 1: male, 0: female
age_sex_bmi


# In[75]:


df_AHEI5 = pd.merge(df_AHEI5, age_sex_bmi, on= "participant_id", how = "left")
df_AHEI5


# In[76]:


df_AHEI5['sex_x'].value_counts()

df_AHEI5['sex'] = np.where(df_AHEI5['sex_x'] == "Male", 1, 0)
print(df_AHEI5['sex'].value_counts())


# In[77]:


# grains (beneficial)
# women 75g/d (5 servings), men 90g/d (6 servings)
df_AHEI5.columns

df_AHEI5.loc[df_AHEI5["sex"] == 0, 'grains_score'] = score_healthy(df_AHEI5.loc[df_AHEI5["sex"] == 0, 'total_grains_g'], 
                                                                   min_serv=0, max_serv=5, min_score=0, max_score=10)
df_AHEI5.loc[df_AHEI5["sex"] == 1, 'grains_score'] = score_healthy(df_AHEI5.loc[df_AHEI5["sex"] == 1, 'total_grains_g'], 
                                                                   min_serv=0, max_serv=6, min_score=0, max_score=10)


# In[78]:


df_AHEI5['grains_score'].describe()


# In[79]:


# sodium
# calculate the decile and give the score (the lowest quantile will have 10, the highest one will have 0)
df_AHEI5['sodium_Decile'] = pd.qcut(df_AHEI5['sodium_mg'], 11, labels=range(10, -1, -1))
df_AHEI5['sodium_Decile'].value_counts()


# In[80]:


# Alcohol score
def calculate_ahei_alcohol(sex, total_alcohol_full_g):
    # Female (gender = 0)
    if sex == 0:
        if total_alcohol_full_g >= 2.5:
            return 0
        elif 1.5 < total_alcohol_full_g < 2.5:
            return (total_alcohol_full_g - 2.5) * 10 / (1.5 - 2.5)
        elif 0.5 <= total_alcohol_full_g <= 1.5:
            return 10
        #elif 0.125 < total_alcohol_full_g < 0.5:
        #    return (total_alcohol_full_g - 0) * 10 / (0.5 - 0)
        elif total_alcohol_full_g < 0.5:
            return 2.5
    
    # Male (gender = 1)
    elif sex == 1:
        if total_alcohol_full_g >= 3.5:
            return 0
        elif 2.0 < total_alcohol_full_g < 3.5:
            return (total_alcohol_full_g - 3.5) * 10 / (2.0 - 3.5)
        elif 0.5 <= total_alcohol_full_g <= 2.0:
            return 10
#         elif 0.125 < total_alcohol_full_g < 0.5:
#             return (total_alcohol_full_g - 0) * 10 / (0.5 - 0)
        elif total_alcohol_full_g < 0.5:
            return 2.5
    
    return None

df_AHEI5['AHEI_ALCOHOL'] = df_AHEI5.apply(lambda row: calculate_ahei_alcohol(row['sex'], row['total_alcohol_full_g']), axis=1)
df_AHEI5['AHEI_ALCOHOL'].describe() 


# In[81]:


df_AHEI5.describe()


# In[82]:


## rowSum of the food components
col_sum = ['Fruit_score', 'ssb_score', 'meat_score', 'trans_score', 'vegetable_score', 'nuts_score', 'fish_score', 'PUFA_score',
          'grains_score', 'sodium_Decile', 'AHEI_ALCOHOL']
print(len(col_sum))

# calculate the row sum
df_AHEI5['AHEI_2010_score'] = df_AHEI5[col_sum].sum(axis = 1)


# In[83]:


# df_AHEI5.to_csv("AHEI_2010_noadj_2.csv", index = False)
df_AHEI5.to_csv("AHEI_2010_noadj_nooutlieradj.csv", index = False)


# In[84]:


import pandas as pd
df_AHEI5 = pd.read_csv("AHEI_2010_noadj_nooutlieradj.csv")
df_AHEI5.describe()


# In[ ]:


# import numpy as np

# cols = df_AHEI5.columns[1:18]

# # calculate the 99.5 percentile of specific columns
# df_AHEI5_2 = df_AHEI5.copy()
# for col in cols:
#     upper = np.percentile(df_AHEI5[col], 99.5)
#     df_AHEI5_2[col] = upper


# In[ ]:


# pd.set_option('display.max_columns', None)


# ## Distribution of the AHEI score

# In[85]:


import seaborn as sns
# sns.set_style('whitegrid')

sns.displot(df_AHEI5, x = "AHEI_2010_score", binwidth=3)



# ============================================================
Source: 02_AMED_score_v2.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Calculate the AMED score
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# ### Ref

# In[1]:


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


# In[2]:


pl = PhenoLoader('diet_logging', age_sex_dataset = None)

events_df_path = os.path.join(DATASETS_PATH, pl.dataset, 'diet_logging_events.parquet')
events_df = pd.read_parquet(events_df_path)

# select the baseline visit 
events_df_base = events_df[events_df.index.get_level_values('research_stage') == "00_00_visit"]

## remove those null rows in specific columns
events_df_base_noNA = events_df_base.dropna(axis = 0, subset = ["short_food_name", "product_name", "food_category"] , how = "all")
events_df_base_noNA1 = events_df_base_noNA.reset_index()
events_df_base_noNA1.shape


# In[3]:


food_name = events_df_base_noNA1["short_food_name"]
food_name.head(5)


# ## AMED food components

# In[4]:


# define a function to categorize food
def categorize_food_item(df, category, items):
    mapping_df = pd.DataFrame({
        'short_food_name': items,
        'category': [category] * len(items)
    })
    
    map_dict = mapping_df.set_index('short_food_name')['category'].to_dict()
    df[category] = df['short_food_name'].map(map_dict)
    
    return df


# In[5]:


# 1. Fruit
Fruit_condition = (events_df_base_noNA1['food_category'] == "Fruits") & (events_df_base_noNA1['short_food_name'].isna())
# Fill NA value with Fruit_item when condition is True
events_df_base_noNA1.loc[Fruit_condition, 'short_food_name'] = "Fruit_item"
Fruit_item = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Fruits"]["short_food_name"].unique().tolist()


# In[6]:


# including the canned veg
Canned_fruit = [
    'Prune',
    'Apricot', 'Dried Fruit', 'Cooked Raisins', 'Fruit salad', 'Cherries', 'Dried apple', 'Dried pineapple', 'Dried cranberries', 'Dried fig',
    'Goji berry', 'Dried Mango', 'Dried blueberries', 'Dates', 'Palm', 'Applesauce'
]
print(len(Canned_fruit))

overlap = [x for x in Fruit_item if x in Canned_fruit]
overlap

Fruit_item1 = list(set(Fruit_item) | set(Canned_fruit))
print(Fruit_item1)


# In[7]:


# 2. Vegetables
condition_veg = (events_df_base_noNA1['food_category'] == "Vegetables") & (events_df_base_noNA1['short_food_name'].isna())
# condition.value_counts() # 1760

# display(events_df_base_noNA1[condition_veg])

# Fill NA value with Fruit_item when condition is True
events_df_base_noNA1.loc[condition_veg, 'short_food_name'] = "Vegetables_item"
Vegetable_item = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Vegetables"]["short_food_name"].unique().tolist()

# ## remove pd.NA
# Vegetable_item = [x for x in Vegetable_item if x is not pd.NA]
print(len(Vegetable_item))

# exclude the fruit in the vegetables
Vegetable_item = [items for items in Vegetable_item if items not in Fruit_item1]
print(len(Vegetable_item))# 163 fruit in total


# In[8]:


# 3. Whole grains
whole_grains = food_name[(food_name.str.contains(r'\bOatmeal Cereal|Cereals|dark bread|brown rice|grain\b|bran\b|wheat?\b|popcorn\b', case=False).fillna(False)) & 
                (~ food_name.str.contains(r'\bcrackers\b', case=False).fillna(False))]
whole_grains_item = whole_grains.unique()

# fill in whole_grain item for the NA in Bread_wholewheat
condition3 = (events_df_base_noNA1['food_category'] == "Bread_wholewheat") & (events_df_base_noNA1['short_food_name'].isna())
# events_df_base_noNA1[condition3]['product_name'].iloc[0] # A yellow cheese sandwich with a full-flavored flour bite bun

events_df_base_noNA1.loc[condition3, 'short_food_name'] = "whole_grains"
Bread_wholewheat = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Bread_wholewheat"]["short_food_name"].dropna().unique()

# get the union whole grain items
whole_grains_item2 = list(set(whole_grains_item) | set(Bread_wholewheat))
print(whole_grains_item2)


# In[9]:


# 4. Nuts
condition_Nuts = (events_df_base_noNA1['food_category'] == "Nuts, seeds, and products") & (events_df_base_noNA1['short_food_name'].isna())
# events_df_base_noNA1[condition3]['product_name'].iloc[0] # A yellow cheese sandwich with a full-flavored flour bite bun
print(condition_Nuts.value_counts())

events_df_base_noNA1.loc[condition_Nuts, 'short_food_name'] = "Nuts"
Nut_item = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Nuts, seeds, and products"]["short_food_name"].unique().tolist()
# ## remove pd.NA
# Nut_item = [x for x in Nut_item if x is not pd.NA]
print(Nut_item) # vegan cheese made of nuts, so included as nuts here


# In[10]:


# 5. Legumes
Legume_item = food_name[food_name.str.contains(r'\bbeans?|Lentil?|Peas|pea\b|tofu\b', case=False).fillna(False)].unique()
display(Legume_item)


# In[11]:


## important notation: change tea in the Proccessed  meat products category into tea_processed meat
events_df_base_noNA1.loc[(events_df_base_noNA1['short_food_name']=="Tea") & (events_df_base_noNA1['food_category'] == "Proccessed  meat products"), 'short_food_name'] = "Tea_processed_meat"
events_df_base_noNA1.loc[(events_df_base_noNA1['short_food_name']=="Tea_processed_meat") & (events_df_base_noNA1['food_category'] == "Proccessed  meat products")]


# In[12]:


# 6. Red and processed meat
condition_red_meat_product = ((events_df_base_noNA1['food_category'] == "Beef, veal, lamb, and other meat products")
                              |(events_df_base_noNA1['food_category'] == "Proccessed  meat products"))& (events_df_base_noNA1['short_food_name'].isna())

print(condition_red_meat_product.value_counts(dropna = False)) #7390 individuals

events_df_base_noNA1.loc[condition_red_meat_product, 'short_food_name'] = "Red_and_processed_meat"

red_meat_product_item = events_df_base_noNA1[
    (events_df_base_noNA1['food_category'] == "Beef, veal, lamb, and other meat products") |
    (events_df_base_noNA1['food_category'] == "Proccessed  meat products")
]["short_food_name"].unique()

# ## remove pd.NA
# red_meat_product_item = [x for x in red_meat_product_item if x is not pd.NA]

red_meat_product_item


# In[13]:


events_df_base_noNA1[
    ((events_df_base_noNA1['food_category'] == "Beef, veal, lamb, and other meat products") | (events_df_base_noNA1['food_category'] == "Proccessed  meat products"))
].food_category.value_counts()


# In[14]:


# 7. Fish
condition_fish = (events_df_base_noNA1['food_category'] == "Fish and seafood") & (events_df_base_noNA1['short_food_name'].isna())
print(condition_fish.value_counts()) # 7081 True

events_df_base_noNA1.loc[condition_fish, 'short_food_name'] = "Fish_seafood"

fish_item = events_df_base_noNA1[(events_df_base_noNA1['food_category'] == "Fish and seafood")]["short_food_name"].dropna().unique()
fish_item


# In[15]:


events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Fruit_item', Fruit_item1)
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Vegetable_item', Vegetable_item)
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'whole_grains_item', whole_grains_item2)
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Nut_item', Nut_item)
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Legume_item', Legume_item)
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'red_meat_product_item', red_meat_product_item)
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'fish_item', fish_item)


# In[16]:


events_df_base_noNA1.columns


# In[17]:


events_df_base_noNA1_count = pd.read_csv('logging_days.csv', index_col=None)


# In[18]:


events_df_base_noNA1_count


# In[19]:


X1 = events_df_base_noNA1.groupby(["participant_id", "Fruit_item"])["weight_g"].sum().reset_index()
X1


# In[20]:


X2 = events_df_base_noNA1.groupby(["participant_id", "Vegetable_item"])["weight_g"].sum().reset_index()
X3 = events_df_base_noNA1.groupby(["participant_id", "whole_grains_item"])["weight_g"].sum().reset_index()
X4 = events_df_base_noNA1.groupby(["participant_id", "Nut_item"])["weight_g"].sum().reset_index()
X5 = events_df_base_noNA1.groupby(["participant_id", "Legume_item"])["weight_g"].sum().reset_index()
X6 = events_df_base_noNA1.groupby(["participant_id", "red_meat_product_item"])["weight_g"].sum().reset_index() # red meat and processed meat
X7 = events_df_base_noNA1.groupby(["participant_id", "fish_item"])["weight_g"].sum().reset_index()
# X8 = events_df_base_noNA1.groupby(["participant_id", "alcohol_g"])["weight_g"].sum().reset_index()


# In[21]:


# define the mapping dictionary for weight column renames
weight_mappings = {
    'X1': 'Fruit_item_g',
    'X2': 'Vegetable_item_g',
    'X3': 'whole_grains_g',
    'X4': 'Nut_g',
    'X5': 'Legumes_g',
    'X6': 'red_meat_product_g',
    'X7': 'fish_g'
}

# define a function to merge and rename
def merge_and_rename(base_df, dataframes_dict):
    
    result_df = base_df.copy()
    for df_name, df in dataframes_dict.items():
        result_df = pd.merge(
            result_df,
            df,
            left_on='ID', right_on='participant_id',  how="left"
        ).drop('participant_id', axis=1)
        
        result_df = result_df.rename(
            columns = {'weight_g': weight_mappings[df_name]},
            inplace = False
        )
        
    return result_df

x_dataframes = {
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'X4': X4,
    'X5': X5,
    'X6': X6,
    'X7': X7
}

AMED_df = merge_and_rename(events_df_base_noNA1_count, x_dataframes)


# In[22]:


AMED_df


# # Fat intake information

# In[23]:


directory2 = '/home/ec2-user/studies/ruifang/Food log.csv'
Food_log = pd.read_csv(directory2)
Food_log['Participant_ID'] = Food_log['RegistrationCode'].str.split("_").str[1]

# Manually define the column order
Food_log = Food_log[['Participant_ID'] + [col for col in Food_log.columns if col != 'Participant_ID']]
Food_log


# In[24]:


Food_log.columns


# In[25]:


Food_log[['Total daily MUFA (g)', 'Total daily PUFA (g)', 'totalsaturatedfattyacids_g']]


# # Food_log['MUFA_SF_ratio'] = Food_log['Total daily MUFA (g)']/Food_log['totalsaturatedfattyacids_g']
# Food_log['MUFA_SF_ratio']

# In[27]:


Food_log['Participant_ID'] = pd.to_numeric(Food_log['Participant_ID'])


# ### Merge food_Log data with AMED info

# In[28]:


AMED_df2 = pd.merge(AMED_df, Food_log[['Participant_ID', 'MUFA_SF_ratio', 'alcohol_g']], left_on = "ID", right_on = "Participant_ID", how= "left").drop('Participant_ID', axis = 1)


# In[29]:


AMED_df2


# In[30]:


AMED_df3 = AMED_df2.filter(regex='ID|days|_g|ratio').drop('whole_grains_item', axis = 1).fillna(0)
AMED_df3


# In[31]:


# divide by log_days (exclude the MUFA_SF_ratio, it is already divided by days)
col_select = ["Fruit_item_g", "Vegetable_item_g", "whole_grains_g", "Nut_g", "Legumes_g", "red_meat_product_g", "fish_g"]

AMED_df3.loc[:, col_select] = AMED_df3.loc[:, col_select].div(AMED_df3["log_days"], axis = 0)
AMED_df3


# In[32]:


def replace_outlier(df):
    df_copy = df.copy()
    
    # columns 
    cols = df_copy.columns[2:]
    for col in cols:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        lower = Q1 - 3*(Q3-Q1)
        upper = Q3 + 3*(Q3-Q1)
        df_copy.loc[df_copy[col] > upper, col] = upper
        df_copy.loc[df_copy[col] < lower, col] = lower
        
    return df_copy

# AMED_df3 = replace_outlier(df = AMED_df3)


# In[33]:


# assign score based on the median value
import numpy as np

var1 = ['Fruit_item_g', 'Vegetable_item_g', 'whole_grains_g', 'Nut_g', 'Legumes_g', 'fish_g', 'MUFA_SF_ratio']

for variable in var1:
    
    # Calculate median
    median = AMED_df3[variable].median()
    
    # Assign scores based on quintiles
    AMED_df3[variable + '_score'] = np.where(AMED_df3[variable] >= median, 1, 0)


# In[34]:


AMED_df3


# In[35]:


# read in the baseline data related with sex
baseline_df = pd.read_csv('age_sex_bmi.csv')
baseline_df


# In[36]:


# select columns and rename
baseline_df_1 = baseline_df[['participant_id', 'sex_x']].rename(columns = {'sex_x': 'sex'})
baseline_df_1

AMED_df4 = pd.merge(AMED_df3, baseline_df_1, left_on = "ID", right_on = "participant_id", how = "left").drop('participant_id', axis = 1)


# In[37]:


AMED_df4


# In[38]:


AMED_df4['sex'].value_counts()


# In[39]:


## sex = 1 means male, 0 means female

AMED_df4_male = AMED_df4[AMED_df4["sex"] == "Male"]
AMED_df4_male.shape # 4641
AMED_df4_male['Alcohol_score'] = np.where((AMED_df4_male['alcohol_g'] < 10) | (AMED_df4_male['alcohol_g'] > 25), 
                                             0, 1) # Alcohol
AMED_df4_female = AMED_df4[AMED_df4["sex"] == "Female"]
AMED_df4_female.shape # 5096
AMED_df4_female['Alcohol_score'] = np.where((AMED_df4_female['alcohol_g'] < 5) | (AMED_df4_female['alcohol_g'] > 15), 
                                                   0, 1) # Alcohol

res_sum_combine = pd.concat([AMED_df4_male, AMED_df4_female]).drop('sex', axis = 1)
res_sum_combine 


# In[40]:


res_sum_combine['red_meat_product_g' + '_score'] = np.where(res_sum_combine['red_meat_product_g'] >= res_sum_combine['red_meat_product_g'].median(), 0, 1)


# In[41]:


res_sum_combine.columns


# In[42]:


res_sum_combine['AMED_score'] = res_sum_combine.loc[: , "Fruit_item_g_score":"red_meat_product_g_score"].sum(axis=1)


# In[43]:


res_sum_combine.shape


# In[44]:


import seaborn as sns


# In[45]:


sns.histplot(data = res_sum_combine, x = "AMED_score")


# In[46]:


res_sum_combine.to_csv('AMED_score_nooutlieradj.csv', index = False)


# In[47]:


import pandas as pd

AMED_score = pd.read_csv("AMED_score_nooutlieradj.csv")
AMED_score


# In[48]:


AMED_score.columns
AMED_score.describe() # without outlier detection


# In[49]:


AMED_score.columns
AMED_score.describe()



# ============================================================
Source: 03_hPDI_score_v2.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Calculate the hPDI score
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[1]:


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


# In[2]:


pl = PhenoLoader('diet_logging', age_sex_dataset = None)
pl


# In[3]:


pl.dict.head()


# In[4]:


df_log = pl.dfs['diet_logging']
df_log.head()


# In[5]:


events_df_path = os.path.join(DATASETS_PATH, pl.dataset, 'diet_logging_events.parquet')
events_df = pd.read_parquet(events_df_path)
events_df_base = events_df[events_df.index.get_level_values('research_stage') == "00_00_visit"]
events_df_base.shape


# In[6]:


events_df = events_df.reset_index()
events_df.columns
## Select baseline persons
events_df_00visit = events_df[events_df["research_stage"] == "00_00_visit"]
events_df_00visit.shape


# In[7]:


# events_df_base_missing = events_df_base[events_df_base["food_category"].isnull()]

## remove those null rows in specific columns
events_df_base_noNA = events_df_base.dropna(axis = 0, subset = ["short_food_name", "product_name", "food_category"] , how = "all")


# In[8]:


events_df_base_noNA1 = events_df_base_noNA.reset_index()

# display(events_df_base_noNA1["food_category"].unique())
display(events_df_base_noNA1["participant_id"].nunique()) # 9737 obs
events_df_base_noNA1.shape


# In[9]:


# events_df_base_noNA1.to_csv("Diet_00visit.csv", index = False)


# In[10]:


# # calculate the logging days
# events_df_base_noNA1.head()
# ID_var = events_df_base_noNA1["participant_id"].unique()
# ID_var.shape # 9759 > 9737

# d = []
# for p in ID_var:
#     d.append({
#         'ID': p,
#         'log_days': events_df_base_noNA1.loc[events_df_base_noNA1['participant_id'] == p]["logging_day"].value_counts().sort_index().rename_axis("Days").to_frame("counts").shape[0]
#     })
# events_df_base_noNA1_count = pd.DataFrame(d)


# In[11]:


# events_df_base_noNA1_count

# display(events_df_base_noNA1_count["log_days"].max())# 14 days
# display(events_df_base_noNA1_count["log_days"].min()) #5 days


# In[12]:


## save the count file
# events_df_base_noNA1_count.to_csv("logging_days.csv", index = False) # 9737 obs


# In[13]:


events_df_base_noNA1_count = pd.read_csv("logging_days.csv")
events_df_base_noNA1_count


# In[14]:


events_df_base_noNA1.groupby(["participant_id", "food_category"], as_index = False)["weight_g"].mean()


# In[15]:


# events_df_base_noNA1[events_df_base_noNA1['food_category'] == "fruit juices and soft drinks"]["short_food_name"].unique()
# events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Bread"]["short_food_name"].unique()

# # Fruit juice is also coded in the Fruits category (removed from Fruits in my case)
# events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Fruits"]["short_food_name"].unique() 
# events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Vegetables"]["short_food_name"].unique()
# events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Fish and seafood"]["short_food_name"].unique()

# print(events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Med Oil and fats"]["short_food_name"].unique()) # Vegetabable oil
# print(events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Oils and fats"]["short_food_name"].unique()) 
# print(events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Nuts, seeds, and products"]["short_food_name"].unique()) 


# In[16]:


food_name = events_df_base_noNA1["short_food_name"]
# food_name.dtype # string type
food_name.head(10)


# In[17]:


### Exclude friut juice from the Fruit
condition = (events_df_base_noNA1['food_category'] == "Fruits") & (events_df_base_noNA1['short_food_name'].isna())
# condition.value_counts() # 1760

# Fill NA value with Fruit_item when condition is True
events_df_base_noNA1.loc[condition, 'short_food_name'] = "Fruit_item"

Fruit_item = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Fruits"]["short_food_name"].unique().tolist()
# display(Fruit_index)
print(Fruit_item)


# In[18]:


import numpy as np

print([i for i, j in enumerate(Fruit_item) if pd.notna(j) and j == 'Fruit juice']) # 41
print([i for i, j in enumerate(Fruit_item) if pd.notna(j) and j == 'Lemon juice']) # 45

Fruit_items = np.delete(Fruit_item, [41, 45]).tolist()
print(Fruit_items)


# ### whole grains

# In[19]:


## Whole grains
print(food_name[(food_name.str.contains(r'\bOatmeal Cereal|Cereals|dark bread|brown rice|grain\b|bran\b|wheat?\b|popcorn\b', case=False).fillna(False)) & 
                (~ food_name.str.contains(r'\bcrackers\b', case=False).fillna(False))].value_counts())


# In[20]:


whole_grains = food_name[(food_name.str.contains(r'\bOatmeal Cereal|Cereals|dark bread|brown rice|grain\b|bran\b|wheat?\b|popcorn\b', case=False).fillna(False)) & 
                (~ food_name.str.contains(r'\bcrackers\b', case=False).fillna(False))]
whole_grains_item = whole_grains.unique()
whole_grains_item

# fill in whole_grain item for the NA in Bread_wholewheat
condition1 = (events_df_base_noNA1['food_category'] == "Bread_wholewheat") & (events_df_base_noNA1['short_food_name'].isna())
# events_df_base_noNA1[condition3]['product_name'].iloc[0] # A yellow cheese sandwich with a full-flavored flour bite bun

events_df_base_noNA1.loc[condition1, 'short_food_name'] = "whole_grains"
Bread_wholewheat = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Bread_wholewheat"]["short_food_name"].dropna().unique()

# get the union whole grain items
whole_grains_item2 = list(set(whole_grains_item) | set(Bread_wholewheat))
whole_grains_item2


# ## Vegetables

# In[21]:


## vegetables
Canned_fruit = [
    'Prune',
    'Apricot', 'Dried Fruit', 'Cooked Raisins', 'Fruit salad', 'Cherries', 'Dried apple', 'Dried pineapple', 'Dried cranberries', 'Dried fig',
    'Goji berry', 'Dried Mango', 'Dried blueberries', 'Dates', 'Palm', 'Applesauce'
]

condition2 = (events_df_base_noNA1['food_category'] == "Vegetables") & (events_df_base_noNA1['short_food_name'].isna())
display(events_df_base_noNA1[condition2])


# In[22]:


# Fill NA value with Fruit_item when condition is True
events_df_base_noNA1.loc[condition2, 'short_food_name'] = "Vegetables_item"

Vegetable_item = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Vegetables"]["short_food_name"].dropna().unique()
Vegetable_item

# excluding Legumes
Legume_item = food_name[food_name.str.contains(r'\bbeans?|Lentil?|Peas|pea\b|tofu\b', case=False).fillna(False)].unique()      
Legume_item

# exclude the legumes item in the vegetable item
Vegetable_item2 = Vegetable_item[~ Vegetable_item.isin(Legume_item)] # exclude the legumes
Vegetable_item3 = Vegetable_item2[~ Vegetable_item2.isin(Fruit_items)] # exclude the fruits

# Vegetable_item2[Vegetable_item2.isin(Fruit_item)] # cranberries, Mango, Lemon are fruit, not vegetables

# exclude the potatoes (including french fries, but including sweet potatoes)
potatoes = pd.Series(food_name[((food_name.str.contains(r'potato(es)?\b', case=False).fillna(False))) &
                               (~(food_name.str.contains(r'Sweet\b', case=False).fillna(False)))
                              ].unique())
print(potatoes)

fried_foods = pd.Series(['Sweet Potato Fries', 'French fries'])
Combined_potatoes = pd.concat([potatoes, fried_foods]).drop_duplicates()
Combined_potatoes
print(Combined_potatoes)

# exclude the potatoes 
Vegetable_item4 = Vegetable_item3[~Vegetable_item3.isin(Combined_potatoes)] 
Vegetable_item4

Canned_veg_fruit = events_df_base_noNA1[(events_df_base_noNA1['food_category'] == "Canned veg and fruits")]["short_food_name"].dropna().unique()

Canned_vegetables = events_df_base_noNA1[
    (events_df_base_noNA1['food_category'] == "Canned veg and fruits") & 
    (~ events_df_base_noNA1['short_food_name'].isin(Canned_fruit))
]["short_food_name"].dropna().unique()

print(Canned_vegetables)

# Convert to pandas Series first
Canned_vegetables = pd.Series(Canned_vegetables)

# Now we can use drop to remove the values
Canned_vegetables = Canned_vegetables[~ Canned_vegetables.isin(['Jam', 'Lemon juice', 'Tea'])]
Canned_vegetables = Canned_vegetables.tolist()
Canned_vegetables

Vegetable_item5 = pd.concat([pd.Series(Vegetable_item4), pd.Series(Canned_vegetables)])
Vegetable_item5

## remove pd.NA
# Vegetable_item = [x for x in Vegetable_item if x is not pd.NA]


# ### Nuts

# In[23]:


## nuts
condition_nut = (events_df_base_noNA1['food_category'] == "Nuts, seeds, and products") & (events_df_base_noNA1['short_food_name'].isna())

print(condition_nut.value_counts())

events_df_base_noNA1.loc[condition_nut, 'short_food_name'] = "Nuts"


# In[24]:


Nut_item = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Nuts, seeds, and products"]["short_food_name"].unique().tolist()

# ## remove pd.NA
# Nut_item = [x for x in Nut_item if x is not pd.NA]
# Nut_item


# ### Legumes

# In[25]:


## Legumes
food_name[food_name.str.contains(r'\bbeans?|Lentil?|Peas|pea\b|tofu\b', case=False).fillna(False)].value_counts()
Legume_item = food_name[food_name.str.contains(r'\bbeans?|Lentil?|Peas|pea\b|tofu\b', case=False).fillna(False)].unique()
Legume_item                   


# ### Vegetable oils

# In[26]:


## Vegetable oils
# print(food_name[food_name.str.contains(r'\boil\b', case=False).fillna(False)].value_counts()) 
vegetable_oil = food_name[food_name.str.contains(r'\boil\b', case=False).fillna(False)].unique()
vegetable_oil


# In[27]:


## important notation: change tea in the Proccessed  meat products category into tea_processed meat
events_df_base_noNA1.loc[(events_df_base_noNA1['short_food_name']=="Tea") & (events_df_base_noNA1['food_category'] == "Proccessed  meat products"), 'short_food_name'] = "Tea_processed_meat"
events_df_base_noNA1.loc[(events_df_base_noNA1['short_food_name']=="Tea_processed_meat") & (events_df_base_noNA1['food_category'] == "Proccessed  meat products")]


# ### Tea & Coffee

# In[28]:


## Tea and Coffee
# print(food_name[food_name.str.contains(r'\btea\b|\bcoffee\b', case=False).fillna(False)].value_counts()) 
tea_coffee = food_name[food_name.str.contains(r'\btea\b|\bcoffee\b', case=False).fillna(False)].unique()
print(tea_coffee)
tea_coffee = ['Coffee', 'Green Tea', 'Tea', 'Diet Ice Coffee', 'Coffee Substitute']


# In[29]:


## all juice exclued vegetable juice
print(food_name[(food_name.str.contains(r'\bjuice|cider\b', case=False).fillna(False)) &
               (~ food_name.str.contains(r'\bCarrot|Beet|Wheatgrass|Celery\b', case=False).fillna(False))].value_counts())
Fruit_juice = food_name[(food_name.str.contains(r'\bjuice|cider\b', case=False).fillna(False)) &
               (~ food_name.str.contains(r'\bCarrot|Beet|Wheatgrass|Celery\b', case=False).fillna(False))].unique()
Fruit_juice


# ### Refined grains

# In[30]:


refined_grains = food_name[food_name.str.contains(r'\brefine\b|\bwhite bread\b|muffins|bagel|roll|biscuit|pancakes|waffle|cracker|pasta', case=False).fillna(False)&
               (~ food_name.str.contains(r'\begg roll\b', case=False).fillna(False))].unique()
refined_grains


# ### Potatoes

# In[31]:


## Potatoes (exclude all the sweet potatoes food products, but not sweet potatoes fries)
# print(food_name[(food_name.str.contains(r'potato(es)?\b', case=False).fillna(False)) &
#               (~ food_name.str.contains(r'Sweet\b', case=False).fillna(False))].value_counts())

potatoes = food_name[(food_name.str.contains(r'potato(es)?\b', case=False).fillna(False)) &
               (~ food_name.str.contains(r'Sweet\b', case=False).fillna(False))].unique()
potatoes


# In[32]:


potatoes_item = potatoes.tolist() + ['Sweet Potato Fries']
potatoes_item


# ### Sugar sweetened beverages

# In[33]:


## Sugar sweetened beverages (Tirosh is one kind of grape juice)
# print(food_name[food_name.str.contains(r'\bcoke\b|Lemonade|Smoothies|Sweetened Cocoa Powder|Flavored Waters|Fruit Drink|Fruit syrup|Salep|Raspberry syrup', case=False).fillna(False)].value_counts()) 
sugar_swee = food_name[food_name.str.contains(r'\bcoke\b|Lemonade|Smoothies|Sweetened Cocoa Powder|Flavored Waters|Fruit Drink|Fruit syrup|Salep|Raspberry syrup', case=False).fillna(False)].unique()
sugar_swee


# ### Sweets and Desserts

# In[34]:


## Sweets and Desserts
food_name[food_name.str.contains(r'\bsugar?|chocolate?\b|candy|pie\b|cake|jam?\b|jelly|jellies|syrup|honey|cookie?\b|brownies|doughnuts|sweet roll\b', case=False).fillna(False)].value_counts()
sweets = food_name[food_name.str.contains(r'\bsugar?|chocolate?\b|candy|pie\b|cake|jam?\b|jelly|jellies|syrup|honey|cookie?\b|brownies|doughnuts|sweet roll\b', case=False).fillna(False)].unique()
sweets


# In[35]:


# Animal food group
# Animal fat
# print(food_name[food_name.str.contains(r'\bButter|lard\b', case=False).fillna(False)].value_counts()) 
Animal_fat = food_name[food_name.str.contains(r'\bButter|lard\b', case=False).fillna(False)].unique()
Animal_fat

## Dairy
# print(food_name[food_name.str.contains(r'\bmilk|cream|yogurt|sherbet|cheese|cottage\b', case=False).fillna(False)].value_counts()) 
Dairy_item = food_name[food_name.str.contains(r'\bmilk|cream|yogurt|sherbet|cheese|cottage\b', case=False).fillna(False)].unique()
Dairy_item


# In[36]:


## Egg
# print(food_name[food_name.str.contains(r'\begg\b', case=False).fillna(False)].value_counts())
egg_item = food_name[food_name.str.contains(r'\begg\b', case=False).fillna(False)].unique()

## Fish or Seafood
# print(food_name[food_name.str.contains(r'\bfish?\b|tuna\b|shrimp|lobster\b|scallop?\b', case=False).fillna(False)].value_counts()) ## not include salmon or Herring as Fish -> other fish
## Use food category "Fish and Seafood" directly
condition_fish = (events_df_base_noNA1['food_category'] == "Fish and seafood") & (events_df_base_noNA1['short_food_name'].isna())
print(condition_fish.value_counts()) 

events_df_base_noNA1.loc[condition_fish, 'short_food_name'] = "Fish_seafood"
fish_item = events_df_base_noNA1[(events_df_base_noNA1['food_category'] == "Fish and seafood")]["short_food_name"].unique()

## Meat
# print(food_name[food_name.str.contains(r'\bmeat|chicken\b|turkey|bacon|hot dogs|liver|hamburger|pork|lamb|beef\b', case=False).fillna(False)].value_counts()) 
meat_item = food_name[food_name.str.contains(r'\bmeat|chicken\b|turkey|bacon|hotdogs|liver|hamburger|pork|lamb|beef\b', case=False).fillna(False)].unique()

## Micellaneous animal-based foods
# print(food_name[food_name.str.contains(r'\bPizza|chowder\b|\bcream soup\b|\bcreamy salad dressing\b', case=False).fillna(False)].value_counts()) ##  Miscellaneous-based foods
micellaneous = food_name[food_name.str.contains(r'\bPizza|chowder\b|\bcream soup\b|\bcreamy salad dressing\b', case=False).fillna(False)].unique()


# In[37]:


def categorize_food_item(df, category, items):
    
    mapping_df = pd.DataFrame({
        'short_food_name': items,
        'category': [category] * len(items)
    })
    
    map_dict = mapping_df.set_index('short_food_name')['category'].to_dict()
    
    # another way to do
    # map_dict = dict(zip(mapping_df['short_food_name'], mapping_df['category']))
    
    df[category] = df['short_food_name'].map(map_dict)
    
    return df


# In[38]:


events_df_base_noNA1.columns


# In[39]:


import pandas as pd
def process_food_categories(events_df: pd.DataFrame) -> pd.DataFrame:
    
    # Define all category mappings in a dictionary for easier maintenance
    food_categories = {
        
        'Vegetables_item': Vegetable_item5,
        "Whole_grains_item": whole_grains_item2,
        'Fruits_item': Fruit_items,
        'Fruit_juice_item': Fruit_juice,
        'Legume_item': Legume_item,
        'Nut_item': Nut_item,
        'Veg_oil_item': vegetable_oil,
        'Tea_coffee_item': tea_coffee,
        "Refined_item": refined_grains,
        "Potatoes_item": potatoes_item,
        "Sugar_swee_item": sugar_swee,
        "Sweets_item":sweets,
        "Animal_item":Animal_fat,
        "Dairy_item": Dairy_item,
        "egg_item": egg_item,
        "fish_item": fish_item,
        "meat_item": meat_item,
        "micellaneous":micellaneous
    }
    
    # Process each category
    for category, items in food_categories.items():
        events_df = categorize_food_item(events_df, category, items)
    
    return events_df


# In[40]:


# events_df_base_noNA1 = process_food_categories(events_df_base_noNA1)


# In[41]:


# Vegetable_item5
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Vegetables_item', Vegetable_item5)

# whole_grains_item2
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Whole_grains_item', whole_grains_item2)

# Fruit_items
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Fruits_item', Fruit_items)

# Fruit_juice
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Fruit_juice_item', Fruit_juice)

# Legume_item
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Legume_item', Legume_item)

# Nut_item
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Nut_item', Nut_item)

# vegetable_oil
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Veg_oil_item', vegetable_oil)

# tea_coffee
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Tea_coffee_item', tea_coffee)

# refined_grains
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Refined_item', refined_grains)

# potatoes_item
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Potatoes_item', potatoes_item)

# sugar_swee
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Sugar_swee_item', sugar_swee)

# sweets
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Sweets_item', sweets)

# Animal_fat
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Animal_item', Animal_fat)

# Dairy_item
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'Dairy_item', Dairy_item)

# egg_item
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'egg_item', egg_item)

# fish_item
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'fish_item', fish_item)


# In[42]:


# meat_item
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'meat_item', meat_item)

# micellaneous
events_df_base_noNA1 = categorize_food_item(events_df_base_noNA1, 'micellaneous', micellaneous)


# In[43]:


events_df_base_noNA1.columns


# In[44]:


events_df_base_noNA1['micellaneous'].value_counts(dropna = False)


# In[45]:


# calculate the participants consumption (sum)
X1 = events_df_base_noNA1.groupby(["participant_id", "egg_item"])["weight_g"].sum().reset_index()
X2 = events_df_base_noNA1.groupby(["participant_id", "Dairy_item"])["weight_g"].sum().reset_index()
X3 = events_df_base_noNA1.groupby(["participant_id", "Animal_item"])["weight_g"].sum().reset_index()
X4 = events_df_base_noNA1.groupby(["participant_id", "Sugar_swee_item"])["weight_g"].sum().reset_index()
X5 = events_df_base_noNA1.groupby(["participant_id", "Fruits_item"])["weight_g"].sum().reset_index()
X6 = events_df_base_noNA1.groupby(["participant_id", "micellaneous"])["weight_g"].sum().reset_index()
X7 = events_df_base_noNA1.groupby(["participant_id", "meat_item"])["weight_g"].sum().reset_index()
X8 = events_df_base_noNA1.groupby(["participant_id", "fish_item"])["weight_g"].sum().reset_index()
X9 = events_df_base_noNA1.groupby(["participant_id", "Nut_item"])["weight_g"].sum().reset_index()
X10 = events_df_base_noNA1.groupby(["participant_id", "Whole_grains_item"])["weight_g"].sum().reset_index()
X11 = events_df_base_noNA1.groupby(["participant_id", "Sweets_item"])["weight_g"].sum().reset_index()
X12 = events_df_base_noNA1.groupby(["participant_id", "Potatoes_item"])["weight_g"].sum().reset_index()
X13 = events_df_base_noNA1.groupby(["participant_id", "Refined_item"])["weight_g"].sum().reset_index()
X14 = events_df_base_noNA1.groupby(["participant_id", "Fruit_juice_item"])["weight_g"].sum().reset_index()
X15 = events_df_base_noNA1.groupby(["participant_id", "Tea_coffee_item"])["weight_g"].sum().reset_index()
X16 = events_df_base_noNA1.groupby(["participant_id", "Veg_oil_item"])["weight_g"].sum().reset_index()
X17 = events_df_base_noNA1.groupby(["participant_id", "Legume_item"])["weight_g"].sum().reset_index()
X18 = events_df_base_noNA1.groupby(["participant_id", "Vegetables_item"])["weight_g"].sum().reset_index()


# In[46]:


events_df_base_noNA1_count = pd.read_csv("logging_days.csv")
events_df_base_noNA1_count


# In[47]:


hPDI_df = pd.merge(events_df_base_noNA1_count, X1, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_egg'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X2, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_dairy'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X3, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_Animal_fat'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X4, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_sugar_swee'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X5, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_Fruit_items'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X6, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_micellaneous'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X7, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_meat_item'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X8, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_fish_item'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X9, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_Nut_item'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X10, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_whole_grain'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X11, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_sweets'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X12, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_potatoes'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X13, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_refined_grains'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X14, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_Fruit_juice'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X15, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_tea_coffee'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X16, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_vegetable_oil'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X17, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_Legume_group'}, inplace = True)

hPDI_df = pd.merge(hPDI_df, X18, left_on='ID', right_on='participant_id',  how="left").drop('participant_id', axis=1)
hPDI_df.rename(columns={'weight_g': 'weight_g_vegetables_group'}, inplace = True)


# In[48]:


hPDI_df


# In[49]:


hPDI_df_2 = hPDI_df.filter(regex='ID|days|weight')


# In[50]:


hPDI_df_2


# In[51]:


hPDI_df_2.columns


# In[52]:


hPDI_df_3 = hPDI_df_2.fillna(0)
hPDI_df_3


# In[53]:


# how to cap or truncate the food intake for each component
def replace_outlier(df):
    df_copy = df.copy()
    
    # columns 
    cols = df_copy.columns[2:]
    for col in cols:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        lower = Q1 - 3*(Q3-Q1)
        upper = Q3 + 3*(Q3-Q1)
        df_copy.loc[df_copy[col] > upper, col] = upper
        df_copy.loc[df_copy[col] < lower, col] = lower
        
    return df_copy

# hPDI_df_3 = replace_outlier(df = hPDI_df_3)


# In[54]:


# divide by log_days
hPDI_df_3.loc[:, "weight_g_egg":"weight_g_vegetables_group"] = hPDI_df_3.loc[:,  "weight_g_egg":"weight_g_vegetables_group"].div(hPDI_df_3["log_days"], axis = 0)
hPDI_df_3


# In[55]:


hPDI_df_3.columns


# In[56]:


# calculate the quintile for each variables
healthfood = ['weight_g_whole_grain', 'weight_g_Fruit_items', 'weight_g_vegetables_group', 'weight_g_Nut_item','weight_g_vegetable_oil',
             'weight_g_Legume_group', 'weight_g_tea_coffee']

unhealthyfood = ['weight_g_Fruit_juice', 'weight_g_refined_grains', 'weight_g_potatoes', 'weight_g_sugar_swee', 
                'weight_g_sweets'] 

Animalfood = ['weight_g_Animal_fat', 'weight_g_dairy', 'weight_g_egg', 'weight_g_fish_item', 
            'weight_g_meat_item', 'weight_g_micellaneous']

unhealthy_all = unhealthyfood + Animalfood


# In[57]:


for variable in healthfood:
    # Calculate quintiles
    quintiles = np.quantile(hPDI_df_3[variable].dropna(), q = np.linspace(0, 1, 6))
    
    # Assign scores based on quintiles
    hPDI_df_3[variable + '_score'] = np.where(hPDI_df_3[variable] >= quintiles[4], 5, 
                                 np.where(hPDI_df_3[variable] >= quintiles[3], 4,
                                 np.where(hPDI_df_3[variable] >= quintiles[2], 3,
                                 np.where(hPDI_df_3[variable] >= quintiles[1], 2, 1))))


# In[58]:


for variable in unhealthy_all:
    # Calculate quintiles
    quintiles = np.quantile(hPDI_df_3[variable].dropna(), q=np.linspace(0, 1, 6))
    
    # Assign scores based on quintiles
    hPDI_df_3[variable + '_score'] = np.where(hPDI_df_3[variable] >= quintiles[4], 1, 
                                 np.where(hPDI_df_3[variable] >= quintiles[3], 2,
                                 np.where(hPDI_df_3[variable] >= quintiles[2], 3,
                                 np.where(hPDI_df_3[variable] >= quintiles[1], 4, 5))))


# In[59]:


# hPDI_df_3.to_csv('hPDI.csv', index = False)
hPDI_df_3.to_csv('hPDI_nooutlieradj.csv', index = False)


# In[60]:


hPDI_df_3


# In[61]:


hPDI_df_3 = hPDI_df_3.set_index('ID')


# In[62]:


hPDI_df_3


# In[63]:


hPDI_df_4 = hPDI_df_3.filter(regex='score')
display(hPDI_df_4)
hPDI_df_4['hPDI_score'] = hPDI_df_4.sum(axis = 1)


# In[64]:


hPDI_df_4.columns


# In[65]:


sns.histplot(data=hPDI_df_4, x="hPDI_score")


# In[66]:


hPDI_df_4['hPDI_score'].describe()


# In[67]:


# hPDI_df_4.reset_index().to_csv('hPDI_sum.csv', index = False)
hPDI_df_4.reset_index().to_csv('hPDI_sum_nooutlieradj.csv', index = False)



# ============================================================
Source: 04_EDIH_score_v2.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Calculate the EDIH
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# Ref: 'Empirical Dietary Index for Hyperinsulinemia (EDIH). Reference: Br J Nutr. 2016;116:1787–1798.'
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[1]:


# load the modules
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
import seaborn as sns
# sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import numpy as np
import os

from pheno_utils import PhenoLoader


# In[2]:


# !pip install pheno_utils


# In[3]:


# baseline data
import os
from pheno_utils.config import (
    DATASETS_PATH, 
    )

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


# In[4]:


food_name = events_df_base_noNA1["short_food_name"]


# In[5]:


## important notation: change tea in the Proccessed  meat products category into tea_processed meat
events_df_base_noNA1.loc[(events_df_base_noNA1['short_food_name']=="Tea") & (events_df_base_noNA1['food_category'] == "Proccessed  meat products"), 'short_food_name'] = "Tea_processed_meat"
events_df_base_noNA1.loc[(events_df_base_noNA1['short_food_name']=="Tea_processed_meat") & (events_df_base_noNA1['food_category'] == "Proccessed  meat products")]

# processed meat
condition6 = (events_df_base_noNA1['food_category'] == "Proccessed  meat products") & (events_df_base_noNA1['short_food_name'].isna())
print(condition6.value_counts()) # 2524 True

events_df_base_noNA1.loc[condition6, 'short_food_name'] = "Processed_meat"

Proccessed_meat = events_df_base_noNA1[events_df_base_noNA1['food_category']=="Proccessed  meat products"]['short_food_name'].unique()
print(Proccessed_meat) # Tea is good chicken tea sausage here 

Proccessed_meat = ['Proecessed_meat', 'Pastrami', 'Sausages', 'Cold cut', 'Meatballs', 'Hamburger', "Tea_processed_meat"]
Proccessed_meat


# In[6]:


# 1. Red meats
# Red_item = events_df_base_noNA1[events_df_base_noNA1['food_category'] == ""]["short_food_name"].unique().tolist()
condition = (events_df_base_noNA1['food_category'] == "Beef, veal, lamb, and other meat products") & (events_df_base_noNA1['short_food_name'].isna())
print(condition.value_counts())

events_df_base_noNA1.loc[condition, 'short_food_name'] = "Red_meat_product"

red_meat_product_item = events_df_base_noNA1[
    (events_df_base_noNA1['food_category'] == "Beef, veal, lamb, and other meat products") 
    # | events_df_base_noNA1['food_category'] == "Proccessed meat products"
]["short_food_name"].dropna().unique()

red_meat_product_item = [items for items in red_meat_product_item if items not in Proccessed_meat]
red_meat_product_item


# In[7]:


# 2.Low energy beverage
condition1 = (events_df_base_noNA1['food_category'] == "Low calories and diet drinks") & (events_df_base_noNA1['short_food_name'].isna())

# Fill NA value with low calories  when condition is True
events_df_base_noNA1.loc[condition1, 'short_food_name'] = "Low_energy_beverages_item"
Low_energy_beverages_item1 = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Low calories and diet drinks"]["short_food_name"].unique()
Low_energy_beverages_item1


# In[8]:


# 3.Cream soup
cream_soup_item = food_name[food_name.str.contains(r'\bsoup\b', case=False).fillna(False)].unique()      
cream_soup_item

cream_soup_item1 = ['Tomato soup and rice', 'Sweet Potato Soup', 'Tomato soup']


# In[9]:


# 4. Poultry
condition2 = (events_df_base_noNA1['food_category'] == "Poultry and its products") & (events_df_base_noNA1['short_food_name'].isna())
events_df_base_noNA1.loc[condition2, 'short_food_name'] = "Poultry_item"
Poultry_item = events_df_base_noNA1[events_df_base_noNA1['food_category'] == "Poultry and its products"]["short_food_name"].unique()
Poultry_item


# In[10]:


# 5. Butter item
butter_item = food_name[
    (food_name.str.contains(r'\bbutter\b', case=False)) & 
    (food_name != 'Butter Cookies')
].fillna(False).unique()  

butter_item


# In[11]:


# 7. French fries
French_fries = food_name[food_name.str.contains(r'\bfrie', case=False).fillna(False)].to_list()

items_to_remove = ['Fried eggplant', 'Fried cauliflower', 'Fried onions', 'Fried zucchini', 'Fried Bread']
French_fries = [item for item in French_fries if item not in items_to_remove]
French_fries = list(set(French_fries))
French_fries


# In[12]:


# 8. Fish
condition4 = (events_df_base_noNA1['food_category'] == "Fish and seafood") & (events_df_base_noNA1['short_food_name'].isna())
print(condition4.value_counts()) # 7081 True

events_df_base_noNA1.loc[condition4, 'short_food_name'] = "Fish_seafood"

fish_item = events_df_base_noNA1[(events_df_base_noNA1['food_category'] == "Fish and seafood")]["short_food_name"].dropna().unique()
fish_item


# In[13]:


# 9. High energy beverages
high_energy_item = food_name[(food_name.str.contains(r'\bsoda|coke', case=False).fillna(False)) & 
          (~food_name.str.contains('diet', case=False).fillna(False))].unique()
high_energy_item


# In[14]:


# 10. tomatoes
tomato_item = food_name[food_name.str.contains(r'\btomato', case=False).fillna(False)].unique()
tomato_item


# In[15]:


# 11. wine
wine_item = events_df_base_noNA1[(events_df_base_noNA1['food_category'] == "Alcoholic Drinks")]["short_food_name"].dropna().to_list()
unique_wine_items = list(set(wine_item))

# only select these contains wine
wine_item1 = [item for item in unique_wine_items if 'wine' in item.lower()]
wine_item1


# In[16]:


# 12. Coffee
coffee_item = food_name[
    (food_name.str.contains(r'\bcoffee', case=False).fillna(False)) & 
    (food_name != 'Coffee Substitute')
].unique()
coffee_item


# In[17]:


# 13. Fruit
fruit_item = food_name[food_name.str.contains(r'\braisin|grapes|avocado|bananas|melon|apple|pear|oranges|grapefruit|strawberries|blueberries|peach|plums|apricots|grape', case=False).fillna(False)].unique()
print(fruit_item)

# exclusion juice and other products
fruit_exclude = ['Apple Cake', 'Apple juice', 'Grapefruit juice', 'Applesauce', 'Avocado Sandwich', 'Apple Vinegar', 'Grape juice', 'Watermelon Seeds']
fruit_item1 = [x for x in fruit_item if x not in fruit_exclude]
len(fruit_item1)

fruit_item1


# In[18]:


fat_dairy = events_df_base_noNA1[(events_df_base_noNA1['food_category'] == "milk, cream cheese and yogurts")]["short_food_name"].unique()

# 14. high_fat 
high_fat_item = ['Ricotta Cheese', 'Cottage cheese', 'Heavy cream', 'Tzfatit Cheese', 'Salty Cheese',
                'White Cheese', 'Feta Cheese', 'Halloumi Cheese', 'Labneh Cheese', 'Leben', 'Eshel', 'Camembert or Brie', 'Soy cheese', 
                'Tzatziki Cheese', 'Mascarpone Cheese', 'Manchego Cheese', 'Ghee', 'Sweetened White Cheese', 'Buttermilk', 'Coconut milk',
                'Sheep milk Labaneh', 'Sheep Milk Yoghurt']

# 15. low_fat
low_fat_item = ['Natural Yogurt', 'Light Soymilk', 'Low fat Milk', 'Light Cream Cheese', 'Goat Cheese',
                'Goats Milk', 'Flavored milk drinks', 'Goat Milk Yogurt',
               'Almond Beverage', 'Oat Drink', 'Rice drink','Flavored Yogurt', 'Gil Leben', 'Soy Pudding']


# In[19]:


# 16. Green_leafy_Vegetables
# cabbage, kale, arugula, Celery, Chard, 
Green_leafy_Vegetables_item = food_name[(food_name.str.contains(r'\bspinach|turnip|collard|lettuce|romaine|cabbage|kale|arugula|celery|chard|parsley\b', case=False).fillna(False)) &
                                        (~food_name.str.contains(r'\bQuiche'))
                                       ].unique()
Green_leafy_Vegetables_item # need to remove the Spinach Quiche


# In[20]:


# 17. egg item
egg_item = food_name[food_name.str.contains(r'\begg\b', case=False).fillna(False)].unique()
egg_item1 = ['Egg']


# In[21]:


# 18, Margarine (not available)
food_name[food_name.str.contains(r'\bmargarine\b', case=False).fillna(False)].unique()


# In[22]:


# set the low_fat item
# df['your_column'] = df['your_column'].apply(lambda x: x if pd.notna(x) and x in high_fat_item else None)

events_df_base_noNA1['low_fat_itme_new'] = events_df_base_noNA1['short_food_name'].apply(lambda x: x if pd.notna(x) and x in low_fat_item else None)
events_df_base_noNA1['low_fat_itme_new'].value_counts()

events_df_base_noNA1['high_fat_itme_new'] = events_df_base_noNA1['short_food_name'].apply(lambda x: x if pd.notna(x) and x in high_fat_item else None)
events_df_base_noNA1['high_fat_itme_new'].value_counts()

events_df_base_noNA1['Fruit_new'] = events_df_base_noNA1['short_food_name'].apply(lambda x: x if pd.notna(x) and x in fruit_item1 else None)
events_df_base_noNA1['Fruit_new'].value_counts()


# In[23]:


# recode the item
def categorize_food_item(df, category, items):
    
    mapping_df = pd.DataFrame({
        'short_food_name': items,
        'category': [category] * len(items)
    })
    
    map_dict = mapping_df.set_index('short_food_name')['category'].to_dict()
    
    # another way to do
    # map_dict = dict(zip(mapping_df['short_food_name'], mapping_df['category']))
    
    df[category] = df['short_food_name'].map(map_dict)
    
    return df

def process_food_categories(events_df: pd.DataFrame) -> pd.DataFrame:
    
    # Define all category mappings in a dictionary for easier maintenance
    food_categories = {
        "red_meat": red_meat_product_item,
        "Low_energy_drinks": Low_energy_beverages_item1,
        "Cream_soup": cream_soup_item1,
        "Processed_meat": Proccessed_meat,
        "Poultry": Poultry_item,
        "Butter": butter_item,
        "French_fries": French_fries,
        "Fish": fish_item,
        "High_energy_drinks" : high_energy_item,
        "tomato": tomato_item,
        "Eggs": egg_item,
        "Wine": wine_item1,
        "Coffee": coffee_item,
        # "high_fat_diary": high_fat_item,
        "Green_veg": Green_leafy_Vegetables_item
    }
    
    # Process each category
    for category, items in food_categories.items():
        events_df = categorize_food_item(events_df, category, items)
    
    return events_df

events_df_base_noNA1 = process_food_categories(events_df_base_noNA1)
events_df_base_noNA1


# In[24]:


events_df_base_noNA1.columns


# In[25]:


def calculate_food_consumption(events_df):
    
    # Define food categories
    food_categories = {
        
        "red_meat": "red_meat",
        "Low_energy_drinks": "Low_energy_drinks",
        "Cream_soup": "Cream_soup",
        "Processed_meat": "Processed_meat",
        "Poultry": "Poultry",
        "Butter": "Butter",
        "French_fries": "French_fries",
        "Fish": "Fish",
        "High_energy_drinks" :  "High_energy_drinks",
        "tomato": "tomato",
        "Eggs": "Eggs",
        "Wine": "Wine",
        "Coffee": "Coffee",
        #"high_fat_diary": "high_fat_diary",
        "Green_veg":  "Green_veg"
        
    }
    
    # Initialize dictionary to store results
    consumption_totals = {}
    
    # Calculate totals for each food category
    for category_key, category_name in food_categories.items():
        
        # filtered_df = events_df[events_df['food_category'] == category_name]
        consumption_totals[category_key] = (
            events_df
            .groupby(['participant_id', f'{category_name}'])['weight_g']
            .sum()
            .reset_index(name = f'total_{category_key}_g')
        )
    
    return consumption_totals

consumption_data = calculate_food_consumption(events_df_base_noNA1)


# In[26]:


consumption_data_df = list(consumption_data.values())


# In[27]:


result = consumption_data_df[0]

# merge with each dataframe in the dictionary
for df in consumption_data_df[1:]:
        result = pd.merge(result, df, on="participant_id", how = "outer") # how = outer, include all observations
        
result


# In[28]:


temp_fruit = events_df_base_noNA1.groupby(['participant_id', 'Fruit_new'])['weight_g'].sum().reset_index()

# long to wide format
temp_fruit_wide = temp_fruit.pivot(index = "participant_id", columns = "Fruit_new", values = "weight_g").reset_index()
temp_fruit_wide


# In[29]:


temp_low_fat = events_df_base_noNA1.groupby(['participant_id', 'low_fat_itme_new'])['weight_g'].sum().reset_index()

temp_low_fat_wide = temp_low_fat.pivot(index = "participant_id", columns = "low_fat_itme_new", values = "weight_g")
temp_low_fat_wide


# In[30]:


temp_high_fat = events_df_base_noNA1.groupby(['participant_id', 'high_fat_itme_new'])['weight_g'].sum().reset_index()

temp_high_fat_wide = temp_high_fat.pivot(index = "participant_id", columns = "high_fat_itme_new", values = "weight_g")
temp_high_fat_wide


# In[31]:


temp_high_fat_wide1 = pd.DataFrame(temp_high_fat_wide.sum(axis = 1)).reset_index().rename(columns = {0:"high_fat_intake"})
temp_high_fat_wide1


# In[32]:


low_fat_yogurt = ['Flavored Yogurt', 'Gil Leben', 'Goat Milk Yogurt', 'Natural Yogurt']

# 'Almond Beverage', 'Flavored milk drinks', Gil Leben, Goat Cheese, 'Goats Milk', 'Light Cream Cheese', 'Light Soymilk', 'Low fat Milk','Oat Drink', 'Rice drink', 'Soy Pudding'
low_fat_Noyogurt_g = pd.DataFrame(temp_low_fat_wide.drop(columns = low_fat_yogurt, axis = 1).sum(axis = 1))
low_fat_yogurt_g = pd.DataFrame(temp_low_fat_wide[low_fat_yogurt].sum(axis = 1))

# low_fat_Noyogurt_g.rename(columns = {'0' : 'weights_low_fat_Noyogurt'}, inplace = True)
# low_fat_yogurt_g.rename(columns = {'0' : 'weights_low_fat_yogurt'}, inplace = True)
low_fat_Noyogurt_g = low_fat_Noyogurt_g.set_axis(['weights_low_fat_Noyogurt'], axis = 1).reset_index()
low_fat_yogurt_g = low_fat_yogurt_g.set_axis(['weights_low_fat_yogurt'], axis = 1).reset_index()


# In[33]:


low_fat_yogurt_g


# In[34]:


# meger the fruit and low fat item into results
result_fruit = pd.merge(result, temp_fruit_wide, on = "participant_id", how = "left")
result_fruit = pd.merge(result_fruit, low_fat_yogurt_g, on = "participant_id", how = "left")
result_all = pd.merge(result_fruit, low_fat_Noyogurt_g, on = "participant_id", how = "left")
result_all = pd.merge(result_all, temp_high_fat_wide1, on = "participant_id", how = "left")
result_all


# In[35]:


# select columns
result_all.columns

columns_select = ['participant_id', 'total_red_meat_g', 'total_Low_energy_drinks_g', 'total_Cream_soup_g', 'total_Processed_meat_g', 'total_Poultry_g', 'total_Butter_g',
                 'total_French_fries_g', 'total_Fish_g', 'total_High_energy_drinks_g', 'total_tomato_g', 'total_Eggs_g', 'total_Wine_g', 'total_Coffee_g', 'total_Green_veg_g',
                 'Apple', 'Avocado', 'Baked apple', 'Blueberries', 'Cooked Raisins', 'Dried apple', 'Dried blueberries', 'Dried pineapple','Grapefruit', 'Grapes', 'Melon', 'Peach', 'Pear', 'Pineapple',
                 'Prickly pears', 'Raisins', 'Watermelon', 'weights_low_fat_yogurt', 'weights_low_fat_Noyogurt', "high_fat_intake"]

result_all = result_all[columns_select]


# In[36]:


result_all


# In[37]:


result_all['total_Processed_meat_g'].describe()


# In[38]:


result_all.describe()


# In[39]:


# calculate the daily intake
servings_dict = {
    
    "total_red_meat_g": 5*28.35,
    "total_Low_energy_drinks_g": 240,
    "total_Cream_soup_g":248,
    "total_Processed_meat_g":45,
    "total_Poultry_g": 5*28.35,
    "total_Butter_g": 7,
    "total_French_fries_g": 4*28.35,
    "total_Fish_g": 4*28.35,
    "total_High_energy_drinks_g": 240,
    "total_tomato_g": 245/2,
    "total_Eggs_g": 50,
    "total_Wine_g": 354,
    "total_Coffee_g": 248,
    "high_fat_intake": 258,
    "total_Green_veg_g": 236.59/2,
    "Apple": 165,
    "Avocado": 304,
    "Baked apple":210,
    "Blueberries":75,
    "Cooked Raisins": 145,
    "Dried apple":86,
    "Dried blueberries": 43,
    "Dried pineapple": 40,
    "Grapefruit": 154,
    "Grapes": 151/2,
    "Melon": 204,
    "Peach": 125,
    "Pear": 125,
    "Pineapple": 165,
    "Prickly pears": 149,
    "Raisins": 145,
    "Watermelon": 325,
    "weights_low_fat_yogurt":245,
    "weights_low_fat_Noyogurt":8*28.35
}
servings_dict.items()


# In[40]:


# change from grams to servings
df_EDIH = result_all.copy()
for col, divisor in servings_dict.items():
    if col in result_all.columns:  # Check if column exists
        df_EDIH[col] = result_all[col] / divisor
df_EDIH     


# In[41]:


# sum all the fruits and low fat diary intake
df_EDIH.columns


# In[42]:


fruit_items = ['Apple', 'Avocado', 'Baked apple', 'Blueberries',
       'Cooked Raisins', 'Dried apple', 'Dried blueberries', 'Dried pineapple',
       'Grapefruit', 'Grapes', 'Melon', 'Peach', 'Pear', 'Pineapple',
       'Prickly pears', 'Raisins', 'Watermelon']

df_EDIH['Fruit_all'] = df_EDIH[fruit_items].fillna(0).sum(axis = 1)

# calculate the low_diary intake
df_EDIH['Low_diary_all'] = df_EDIH['weights_low_fat_yogurt'].fillna(0) + df_EDIH['weights_low_fat_Noyogurt'].fillna(0) 

df_EDIH2 = df_EDIH.drop(columns = fruit_items + ['weights_low_fat_yogurt', 'weights_low_fat_Noyogurt'], axis = 0)
df_EDIH2


# In[43]:


# calculate the missingness for the food component
missing = pd.DataFrame(df_EDIH2.iloc[:, 1:].isna().sum(axis = 1)/len(result_all.iloc[:, 1:].columns))
missing.describe()


# In[44]:


df_EDIH2 = df_EDIH2.fillna(0)


# In[45]:


log_d = pd.read_csv('logging_days.csv')
log_d

df_EDIH3 = pd.merge(df_EDIH2, log_d, left_on= "participant_id", right_on= "ID", how = "left").drop(['ID'], axis = 1).set_index('participant_id')
df_EDIH3


# In[46]:


df_EDIH4 = df_EDIH3.copy()

for variables in df_EDIH4.columns:
    if variables != "log_days":
        df_EDIH4[variables] = df_EDIH3[variables] / df_EDIH3['log_days']
        df_EDIH4


# In[47]:


df_EDIH4


# In[48]:


df_EDIH4['total_Processed_meat_g'].describe()


# In[49]:


df_EDIH4.describe()


# In[50]:


df_EDIH4.columns


# In[51]:


# how to cap or truncate the food intake for each component
def replace_outlier(df):
    df_copy = df.copy()
    
    # columns 
    cols = df_copy.columns[1:]
    for col in cols:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        lower = Q1 - 3*(Q3-Q1)
        upper = Q3 + 3*(Q3-Q1)
        df_copy.loc[df_copy[col] > upper, col] = upper
        df_copy.loc[df_copy[col] < lower, col] = lower
        
    return df_copy

# df_EDIH4 = replace_outlier(df = df_EDIH4)


# In[52]:


df_EDIH4.describe()


# In[53]:


# Calculate the EDIH
# including the variables
EDIH = ["total_red_meat_g", "total_Low_energy_drinks_g", "total_Cream_soup_g", "total_Processed_meat_g", "total_Poultry_g", 
        "total_Butter_g", "total_French_fries_g", "total_Fish_g", "total_High_energy_drinks_g",
       "total_tomato_g", "Low_diary_all", "total_Eggs_g", "total_Wine_g", "total_Coffee_g", "Fruit_all", "high_fat_intake", "total_Green_veg_g"]

# weights for each item
weights = [0.250, 0.053, 0.787, 0.199, 0.183, 0.094, 0.581, 0.172, 0.104, 0.095, 0.025, 0.124, -0.165, -0.035, -0.029, -0.046, -0.055]

EDIH_score_ref = pd.DataFrame(list(zip(EDIH, weights)),
              columns=['EDIH_variable','weights'])
EDIH_score_ref


# In[54]:


# calculate the EDIH score
EDIH_score = df_EDIH4.copy() 

for variable in df_EDIH4.drop(columns = "log_days", axis = 1).columns:
    
    # variable = df_EDIH4.columns[0]
    ref_row = EDIH_score_ref[EDIH_score_ref['EDIH_variable'] == variable]
    EDIH_score[variable] = EDIH_score[variable] * ref_row['weights'].values[0]


# In[55]:


EDIH_score


# In[56]:


# sum all the EDIH score
EDIH_score['EDIH_score_all'] = EDIH_score.loc[:, "total_red_meat_g":"Low_diary_all"].sum(axis = 1)
EDIH_score


# In[57]:


EDIH_score = EDIH_score.reset_index()


# In[58]:


# EDIH_score.to_csv('EDIH_score.csv', index = False)
EDIH_score.to_csv('EDIH_score_nooutlieradj.csv', index = False)


# In[59]:


import seaborn as sns
import pandas as pd

EDIH_score = pd.read_csv("EDIH_score_nooutlieradj.csv")
EDIH_score.columns


# In[60]:


EDIH_score['EDIH_score_all'].describe()


# In[61]:


sns.displot(EDIH_score, x = "EDIH_score_all")



# ============================================================
Source: 05_DII_scores_v2.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Calculate the DII score
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[141]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
import seaborn as sns

# sns.set_style('whitegrid')

import matplotlib.pyplot as plt
import numpy as np

from pheno_utils import PhenoLoader


# In[142]:


# pl = PhenoLoader('curated_phenotypes')
# pl


# In[143]:


# pl.dict[['feature_set', 'relative_location']].drop_duplicates()
# # List of all available feature sets
# list(pl.dict.feature_set.unique())


# In[144]:


# Load MASLD data
# df_mafld = pl.dfs['mafld']
# df_mafld.head(20)

# # select baseline
# df_mafld_bs = df_mafld[df_mafld.index.get_level_values('research_stage') == '00_00_visit']
# df_mafld_bs.head(20)


# In[145]:


# # Load MASLD data
# df_depression = pl.dfs['depression']
# df_depression.head(20)
# df_depression_bs = df_depression[df_depression.index.get_level_values('research_stage') == '00_00_visit']


# In[146]:


# df_depression_fl = df_depression[df_depression.index.get_level_values('research_stage') == '02_00_visit']
# df_depression_fl['depression__curated_phenotype'].value_counts()


# In[147]:


# df_depression_bs.head()
# df_depression_bs['depression__curated_phenotype'].value_counts()


# In[148]:


# df_mafld_bs['mafld__curated_phenotype'].value_counts(dropna = False)


# In[149]:


# df_mafld.index.get_level_values('research_stage').value_counts()


# In[150]:


# df_mafld_flw = df_mafld[df_mafld.index.get_level_values('research_stage') == '01_00_call']


# In[151]:


# df_mafld_bs.shape # 10950, 11
# df_mafld_bs['mafld__self_reporting'].value_counts() # 143 MAFLD


# In[152]:


Nutrients = pd.read_csv("../studies/ruifang/Food log.csv")
display(Nutrients.columns)
Nutrients.head()


# In[153]:


pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
Nutrients.describe()


# In[154]:


Nutrients.loc[Nutrients["alcohol_g"] < 0, "alcohol_g"] = np.nan


# In[155]:


Nutrients.columns


# In[156]:


# Nutrients.loc[Nutrients["Caffeine (g)"] != 0,].describe()


# In[157]:


Nutrients["Vitamin A (ug)"].isnull().sum()


# In[158]:


Nutrients.loc[Nutrients["Vitamin A (ug)"] > 10000,]


# In[159]:


Nutrients = Nutrients.dropna(subset = "alcohol_g")
Nutrients.shape


# In[160]:


# Nutrients["Zinc (mg)"].isnull().sum()
Nutrients["alcohol_g"].isnull().sum()


# In[161]:


Nutrients['energy_kcal'].describe()


# In[162]:


(Nutrients["totalfolate_ug"].describe())


# In[163]:


display(Nutrients['Caffeine (mg)'].describe())
Nutrients


# In[164]:


Nutrients['Caffeine (mg)'] = Nutrients['Caffeine (mg)']/1000
Nutrients['Caffeine (mg)'].describe()


# In[165]:


# rename the column 
Nutrients = Nutrients.rename(columns={'Caffeine (mg)': 'Caffeine (g)'})


# In[166]:


# DII component
DII_variable = ['alcohol_g', #'Caffeine (g)', 
                'carbohydrate_g', 'Cholesterol (mg)', 'energy_kcal', 'totallipid_g', 'totaldietaryfiber_g', 'Iron (mg)', 
                'Magnesium (mg)', 'Total daily MUFA (g)',  'Niacin (mg)', 'Total daily PUFA (g)', 'protein_g', 'Riboflavin (mg)',
                'totalsaturatedfattyacids_g', 'Thiamin (mg)', 'Vitamin A (ug)',
                'Vitamin B6 (mg)','Vitamin B12 (ug)', 'Vitamin C (mg)', 'Vitamin E (mg)', 'Zinc (mg)'
                # ,'totalfolate_ug'
               ]

# Inflammatory score weight
Inflammatory_score = [-0.278, #-0.110, 
                      0.097, 0.110, 0.180,0.298, -0.663, 0.032, -0.484, -0.009, -0.246, -0.337, 0.021, -0.068, 0.373, -0.098, -0.401, -0.365,
                     0.106, -0.424, -0.419, -0.313
                      #, -0.190
                     ]

# DII global mean
global_mean = [13.98, #8.05, 
               272.2, 279.4, 2056, 71.4, 18.8, 13.35, 310.1, 27, 25.9, 13.88, 79.4, 1.70, 28.6, 1.70, 983.9, 1.47, 5.15, 118.2, 8.73, 9.84
               # , 273
              ]
global_sd = [3.72, # 6.67, 
             40, 51.2, 338,19.4, 4.9, 3.71, 139.4, 6.1, 11.77, 3.76, 13.9, 0.79, 8, 0.66, 518.6, 0.74, 2.70, 43.46, 1.49, 2.19
             #, 70.7
            ]

# DII global sd
# zip function to generate the dictionary
DII_score = pd.DataFrame(list(zip(DII_variable, Inflammatory_score, global_mean, global_sd)),
              columns=['DII_variable','weights', "global_mean", "global_sd"])


# In[167]:


DII_score


# In[188]:


# Calculate mean and SD for each column
stats = Nutrients[DII_variable].agg(["mean", "std"]).T
stats = stats.rename(columns = {"mean": "Mean_10K", "std": "SD_10K"})
stats = stats.reset_index().rename(columns = {'index': "DII_variable"})
stats

temp_merge = pd.merge(DII_score, stats, on = "DII_variable", how = "left")
temp_merge


# In[168]:


# DII
Nutrients_select = ['RegistrationCode'] + DII_variable
Nutrients_select = Nutrients[Nutrients_select]
Nutrients_select.set_index('RegistrationCode')


# In[169]:


Nutrients_select_2 = Nutrients_select.copy()
Nutrients_select_2 = Nutrients_select_2.set_index('RegistrationCode')

for variable in Nutrients_select_2.columns:
    
    # extract the global mean and sd for DII variable
    ref_row = DII_score[DII_score['DII_variable'] == variable]
    global_mean = ref_row['global_mean'].values[0]
    global_sd = ref_row['global_sd'].values[0]
    Nutrients_select_2[variable] = (Nutrients_select_2[variable] - global_mean) / global_sd


# In[170]:


# Z score
Nutrients_select_2.head()


# In[171]:


# Correcting CPS calculation for each variable in DII 
import scipy.stats as stats

# Assuming DII2 is a pandas DataFrame
for variable in Nutrients_select_2.columns:
    
    # Calculate percentiles from z-scores
    percentiles = stats.norm.cdf(Nutrients_select_2[variable])
    
    # Calculate CPS from percentiles
    cps = (percentiles * 2) - 1
    
    # Replace the variable with CPS
    Nutrients_select_2[variable] = cps
    
Nutrients_select_2


# In[172]:


# Nutrients_select_2["totalfolate_ug"].nunique()


# In[173]:


# generate the new disctionary
Weights = dict(zip(DII_variable, Inflammatory_score))
Weights.items()


# In[174]:


# multiple by weights for each component
for column, weight in Weights.items():
    Nutrients_select_2[column] = Nutrients_select_2[column] * weight
Nutrients_select_2


# In[175]:


# DII score is calculated as the sum of each dietary parameter's IES multipled by the participants' central percentile of consumption
Nutrients_select_2['DII_score'] = Nutrients_select_2.sum(axis=1)


# In[176]:


Nutrients_select_2 = Nutrients_select_2.reset_index()


# In[177]:


Nutrients_select_2['RegistrationCode'] = Nutrients_select_2['RegistrationCode'].str.split("_", expand = True).iloc[:,1]


# In[178]:


Nutrients_select_2


# In[179]:


Nutrients_select_2.rename(columns={"RegistrationCode": "participant_ID"}, inplace=True)
Nutrients_select_2


# In[180]:


# Show the distribution of DII score
sns.displot(Nutrients_select_2, x = "DII_score")


# In[181]:


Nutrients_select_2.to_csv('DII_score_v2_20250624.csv', index = False)


# In[182]:


Nutrients_select_2["DII_score"].describe()


# In[183]:


Nutrients_select_2['DII_score'].describe()


# In[184]:


import pandas as pd
DII = pd.read_csv("DII_score_v2_20250624.csv")
DII


# In[189]:


DII["DII_score"].isnull().sum()


# In[ ]:






# ============================================================
Source: 06_Correlation_diet_scores.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Calculate the correlation between diet_scores
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# Ref: 'Empirical Dietary Index for Hyperinsulinemia (EDIH). Reference: Br J Nutr. 2016;116:1787–1798.'
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[36]:


## Check the correlation of calculated dietary index scores
# hPDI
# AHEI
# EDIH
# AMED
# DII


# In[37]:


import pandas as pd
import numpy as np

AMED = pd.read_csv("./Noadj/AMED_score_nooutlieradj.csv")
AMED = AMED[['ID', 'AMED_score']]
AMED


# In[38]:


hPDI = pd.read_csv("./Noadj/hPDI_sum_nooutlieradj.csv")
hPDI = hPDI[["ID", "hPDI_score"]]
hPDI


# In[39]:


AHEI = pd.read_csv("./Noadj/AHEI_2010_noadj_nooutlieradj.csv")
AHEI.columns

AHEI = AHEI[['participant_id', 'AHEI_2010_score']]
AHEI


# In[40]:


data1 = pd.merge(AHEI, hPDI, left_on="participant_id", right_on="ID", how = "outer").drop(columns = 'ID', axis = 0)
data1


# In[41]:


EDIH = pd.read_csv("./Noadj/EDIH_score_nooutlieradj.csv")
EDIH

EDIH["total_Butter_g"].describe()


# In[42]:


data1 = pd.merge(data1, EDIH[['participant_id', 'EDIH_score_all']], on = "participant_id", how = "left")

DII = pd.read_csv('DII_score_v2_20250624.csv')
display(DII.head())
data1 = pd.merge(data1, DII[['participant_ID', 'DII_score']], left_on= "participant_id", right_on= "participant_ID", how = "left")
data1


# In[43]:


data1 = data1.drop(columns = "participant_ID", axis = 1)
data1


# In[44]:


# Merge with AMED score
data1_1 = pd.merge(data1, AMED[['ID', 'AMED_score']], left_on= "participant_id", right_on = "ID", how = "left").drop(columns = "ID", axis = 1).set_index("participant_id")
data1_1.shape


# In[45]:


# Update the DII score after unit change
display(data1_1.corr("spearman"))
data1_1.corr("pearson")


# In[46]:


data1_1.corr("spearman")


# In[47]:


data1_1.corr("pearson")


# In[48]:


data1_1


# In[49]:


# Export five diet scores
data2 = data1_1.reset_index()


# In[50]:


data2


# In[51]:


data2['EDIH_score_all'].describe()


# In[52]:


data2['DII_score'].describe()


# In[53]:


data2.to_csv("./Noadj/Five_dietary_score_nooutlieradj_v3.csv", index = False)



# ============================================================
Source: 07_Energy_adjustment_dps.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Energy Adjustment
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

# from pheno_utils import PhenoLoader
# from pheno_utils.config import (
#     DATASETS_PATH, 
#     )


# In[2]:


# read in the five dietary score
five_scores = pd.read_csv('./Noadj/Five_dietary_score_nooutlieradj_v3.csv')
five_scores

energy_intake = pd.read_csv("./Archiv/AHEI3_2024_1125_02.csv")
energy_intake = energy_intake[['participant_id','energy_kcal']]
energy_intake


# In[37]:


df = pd.merge(five_scores, energy_intake, on = "participant_id", how = "left")
df

# merge with the lifestyle information
life_style = pd.read_csv('../02_Lifestyle factors/lifestyle_factor_all_disease_V2.csv')
life_style


# In[38]:


life_style.columns.tolist()


# In[39]:


df_bs = pd.merge(df, life_style, on = "participant_id", how = "left")
df_bs.shape


# In[40]:


df_bs.head()


# In[41]:


df_bs['sex'] = np.where(df_bs['sex_x'] == "Male", 1, 0)
df_bs['sex'].value_counts()


# In[42]:


# Remove rows where column1 or column2 have NA
df_bs = df_bs.dropna(subset=['energy_kcal'])
df_bs.shape


# In[43]:


# df_bs["Vit_use"].describe()
df_bs["Vit_use"].value_counts()


# In[44]:


df_bs[["AHEI_2010_score", "hPDI_score", "EDIH_score_all", "DII_score", "AMED_score"]].isnull().sum()


# In[123]:


# !pip install statsmodels
# !pip list | grep statsmodels


# In[45]:


df_bs = df_bs.dropna(subset = ["EDIH_score_all", "DII_score"])
df_bs.shape


# In[46]:


df_bs["sex"].value_counts()


# In[47]:


import statsmodels.api as sm

def perform_analysis(index_name, data):
    # Subset the data by sex, 1 = male, 0 = female
    male_data = data[data['sex'] == 1].copy()
    female_data = data[data['sex'] == 0].copy()

    # Perform linear regression for males
    X_male = sm.add_constant(male_data['energy_kcal'])
    y_male = male_data[index_name]
    lm_male = sm.OLS(y_male, X_male).fit()
    male_data['residuals'] = lm_male.resid
    male_constant = lm_male.params.iloc[0]
    male_coefficient = lm_male.params.iloc[1]
    male_energy_mean = male_data['energy_kcal'].mean()
    male_mean_index = male_constant + male_coefficient * male_energy_mean
    male_data[index_name + '_eadj'] = male_data['residuals'] + male_mean_index

    # Perform linear regression for females
    X_female = sm.add_constant(female_data['energy_kcal'])
    y_female = female_data[index_name]
    lm_female = sm.OLS(y_female, X_female).fit()
    female_data['residuals'] = lm_female.resid
    female_constant = lm_female.params.iloc[0]
    female_coefficient = lm_female.params.iloc[1]
    female_energy_mean = female_data['energy_kcal'].mean()
    female_mean_index = female_constant + female_coefficient * female_energy_mean
    female_data[index_name + '_eadj'] = female_data['residuals'] + female_mean_index

    # Merge adjusted index back into original dataset
    adjusted_data = pd.concat([male_data[['participant_id', index_name + '_eadj']], female_data[['participant_id', index_name + '_eadj']]])
    data = pd.merge(data, adjusted_data, on='participant_id', how='left')

    return data


# In[48]:


df_bs = perform_analysis('AHEI_2010_score', data = df_bs)
df_bs = perform_analysis('hPDI_score', data = df_bs)
# df_bs = perform_analysis('EDIH_score_all', data = df_bs)
df_bs = perform_analysis('DII_score', data = df_bs)
df_bs = perform_analysis('AMED_score', data = df_bs)


# In[49]:


df_bs


# In[15]:


# df_bs.to_csv("df_bs_AHEI_hPDI_AMED_DII_adj2.csv", index = False)


# In[50]:


df_bs1 = df_bs.dropna(subset = "EDIH_score_all")
df_bs1 = perform_analysis('EDIH_score_all', data = df_bs1)


# In[51]:


print(df_bs1.shape) # 9616 rows for EDIH
df_bs1.to_csv("./Adj/df_bs_EDIH_adj2.csv", index = False)


# In[52]:


df_bs1 = pd.read_csv("./Adj/df_bs_EDIH_adj2.csv")
df_bs1


# In[53]:


df_bs1.columns


# In[54]:


df_bs1 = df_bs1.set_index("participant_id")


# In[55]:


df_bs1['EDIH_score_all_eadj'].describe()


# In[60]:


df_bs1['DII_score_eadj'].describe()


# In[56]:


df_bs1['energy_kcal'].describe()


# In[57]:


## Calculated the correlation between five dietary pattern scores
col_select = ['AHEI_2010_score_eadj', 'hPDI_score_eadj', 'DII_score_eadj', 'AMED_score_eadj', 'EDIH_score_all_eadj']
df_bs1_select = df_bs1[col_select]
df_bs1_select.corr("spearman")


# In[58]:


df_bs1_select["DII_score_eadj"].isnull().sum()


# In[59]:


from scipy import stats

# also calculate the pvalue
corr = stats.spearmanr(df_bs1_select['AHEI_2010_score_eadj'], df_bs1_select['hPDI_score_eadj'])
corr


# In[61]:


# function to calculate the pvalue_matrix
def calculate_pvalue_matrix(data):
    
    cols = data.columns
    n_cols = len(cols)
    
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
    
#     # visualization
# #   fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))
#     plt.figure(figsize=(20,8))
    
#     # correlation heatmap
#     plt.subplot(1, 2, 1)
#     sns.heatmap(correlation_matrix, annot = True, 
#                cmap = "coolwarm", vmin = -1, vmax = 1, center = 0, ax = ax1)
    
#     ax1.set_title('Correlation Coefficients')
    
#     # P-value heatmap
#     plt.subplot(1, 2, 2)
#     sns.heatmap(p_value_matrix,
#                 annot=True,
#                 cmap='YlOrRd',
#                 vmin=0, vmax=0.1,
#                 ax=ax2)
    
#     ax2.set_title('P-values')
    
#     plt.tight_layout()
#     plt.show()
                


# In[62]:


combined_matrix, correlation_matrix, p_value_matrix = calculate_pvalue_matrix(df_bs1_select)


# In[63]:


combined_matrix


# In[64]:


correlation_matrix


# In[65]:


sns.heatmap(correlation_matrix, annot = True, 
               cmap = "coolwarm", vmin = -1, vmax = 1, center = 0)


# In[144]:


# calculate_pvalue_matrix(df_bs1_select)


# In[66]:


# reverse the DII and EDIH score
df_bs1_select_r = df_bs1_select.copy()
df_bs1_select_r['DII_score_eadj'] = (-1) * df_bs1_select_r['DII_score_eadj']
df_bs1_select_r['EDIH_score_all_eadj'] = (-1) * df_bs1_select_r['EDIH_score_all_eadj']
df_bs1_select_r


# In[68]:


# Using a dictionary to map old names to new names
df_bs1_select_r = df_bs1_select_r.rename(columns={'DII_score_eadj': 'rDII_score_eadj', 
                       'EDIH_score_all_eadj': 'rEDIH_score_all_eadj'})
df_bs1_select_r


# In[69]:


# how to cap or truncate the food intake for each component
def truncations(df):
    df_copy = df.copy()
    
    # columns 
    cols = df_copy.columns
    for col in cols:
        lower = df_copy[col].quantile(0.005)
        upper = df_copy[col].quantile(0.995)
        df_copy.loc[df_copy[col] > upper, col] = upper
        df_copy.loc[df_copy[col] < lower, col] = lower
        
    return df_copy

df_bs1_select_r = truncations(df_bs1_select_r)


# In[70]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create figure and axes with specific layout
fig, axes = plt.subplots(2, 3, figsize=(10, 5))

sns.histplot(df_bs1_select_r, x="AHEI_2010_score_eadj", ax = axes[0,0], color = "#3C5488") # Light purple
axes[0,0].set_xlabel('AHEI')
sns.histplot(df_bs1_select_r, x="hPDI_score_eadj", ax = axes[0,1], color = "#BE2C34") # Ruby red
axes[0,1].set_xlabel('hPDI')
sns.histplot(df_bs1_select_r, x="rDII_score_eadj", ax = axes[0,2], color = "#E5A733") #  Orange
axes[0,2].set_xlabel('rDII')
sns.histplot(df_bs1_select_r, x="AMED_score_eadj", ax = axes[1,0], color = "#279B48") # Forest green
axes[1,0].set_xlabel('AMED')
sns.histplot(df_bs1_select_r, x="rEDIH_score_all_eadj", ax = axes[1, 1], color = "#4DBBD5") # navy blue
axes[1,1].set_xlabel('rEDIH')

# Remove the empty subplot
fig.delaxes(axes[1,2])

# adjust the layout to prevent overlap
plt.tight_layout()
# plt.show()

# plt.savefig('Five_score_adj_distribution.png', dpi=300)

# plt.close()

# # show the plot
plt.savefig("./Results/Distribution_of_five_scores_v3.jpg")
plt.show()


# In[72]:


df_bs1_select_r["rDII_score_eadj"].describe()


# In[188]:


# df_bs1_select_r.to_csv("DPs_Final_score.csv")
# df_bs1_select_r.to_csv("./Adj/DPs_Final_score_outlieradj_v2.csv")


# In[73]:


combined_matrix_r, correlation_matrix_r, p_value_matrix_r = calculate_pvalue_matrix(df_bs1_select_r)

correlation_matrix_r = correlation_matrix_r.rename(
    index ={'AHEI_2010_score_eadj': 'AHEI', 
            'hPDI_score_eadj': 'hPDI', 
            'rDII_score_eadj': 'rDII',
           "AMED_score_eadj": 'AMED',
           "rEDIH_score_all_eadj": "rEDIH"},
    columns ={'AHEI_2010_score_eadj': 'AHEI', 
            'hPDI_score_eadj': 'hPDI', 
            'rDII_score_eadj': 'rDII',
           "AMED_score_eadj": 'AMED',
           "rEDIH_score_all_eadj": "rEDIH"})

plt.figure(figsize=(8, 6))

sns.heatmap(correlation_matrix_r, annot = True, 
               cmap = "coolwarm", vmin = -1, vmax = 1, center = 0)

# Adjust the layout to prevent label truncation
plt.tight_layout()

# plt.savefig('./Results/Correlation_five_dps_v2.png', bbox_inches='tight', pad_inches=0.5)


# In[76]:


# Getting the Upper Triangle of the co-relation matrix
import matplotlib.pyplot as plt
plt.figure(figsize = (6, 5))

correlation_matrix_r_half = np.triu(correlation_matrix_r, k = 1)
sns.heatmap(correlation_matrix_r, annot = True, 
               cmap = "coolwarm", vmin = -1, vmax = 1, center = 0, 
               mask = correlation_matrix_r_half)

# Adjust the layout to prevent label truncation
plt.tight_layout()

# plt.savefig('./Results/Correlation_five_dps_half_v2.png', bbox_inches='tight', pad_inches=0.5)
plt.savefig('./Results/Correlation_five_dps_half_v3.pdf', bbox_inches='tight')


# In[ ]:






# ============================================================
Source: 08_DPs_Category.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# ## Energy Adjustment
# -----------
# Keyong Deng,
# Leiden University Medical Center
# Please cite the reseach article related to the provided code. 
# For any questions, please contact k.deng@lumc.nl
# -----------

# In[2]:


import pandas as pd


# In[16]:


DPs = pd.read_csv(filepath_or_buffer= "./Adj/DPs_Final_score_outlieradj_v2.csv")
DPs


# In[4]:


DPs.columns


# In[17]:


# read in the logging days
Logging_days =  pd.read_csv("./Archiv/AHEI3_2024_1125_final.csv")

print(Logging_days.shape) # 9737 individuals

"log_days" in Logging_days.columns # False

Logging_days = Logging_days[["participant_id", "log_days"]]
Logging_days.to_csv("logging_days.csv", index = False)


# In[18]:


# combine with DPs
DPs_2 = pd.merge(DPs, Logging_days, on = "participant_id", how = "left")

display(DPs_2["log_days"].describe())

DPs_2["log_days"].value_counts()


# In[11]:


# assign quintiles
cols_process = ["AHEI_2010_score_eadj", "hPDI_score_eadj", "rDII_score_eadj", "AMED_score_eadj", "rEDIH_score_all_eadj"]

for col in cols_process:
    col_new = f'{col}_quntile'
    DPs[col_new] = pd.qcut(DPs[col], 5, labels = ["Q1", "Q2", "Q3", "Q4", "Q5"])


# In[12]:


DPs


# In[13]:


# plot the plot for different dietary scores
import seaborn as sns

sns.violinplot(data = DPs, 
               x = "AHEI_2010_score_eadj_quntile", 
               y = "AHEI_2010_score_eadj", 
               hue = "AHEI_2010_score_eadj_quntile")

# sns.violinplot(data=DPs, 
#                x="AHEI_2010_score_eadj_quntile", 
#                y="hPDI_score_eadj", 
#                hue="AHEI_2010_score_eadj_quntile")

# sns.violinplot(data=DPs, 
#                x="AHEI_2010_score_eadj_quntile", 
#                y="rDII_score_eadj", 
#                hue="AHEI_2010_score_eadj_quntile")


# In[14]:


sns.violinplot(data=DPs, 
               x="rDII_score_eadj_quntile", 
               y="rDII_score_eadj", 
               hue="rDII_score_eadj_quntile")


# In[15]:


DPs["rDII_score_eadj"].describe()


# In[16]:


DPs.shape


# In[17]:


DPs.to_csv("DPs_Final_Score_with_Category_v2.csv", index = False)



# ============================================================
Source: 09_Collect_Date.py
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

pl = PhenoLoader('diet_logging', age_sex_dataset = None)
pl


# In[2]:


df_log = pl.dfs['diet_logging']
df_log.head()


# In[8]:


df_log.loc[df_log.index.get_level_values('participant_id').isin([1321172995])]



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