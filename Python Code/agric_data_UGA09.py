# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:12:51 2018

@author: rodri
"""

#### Agriculture productivity Analysis Uganda 2009-10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col
#os.chdir('D:/Documents/Documents/IDEA/Research/python Albert')
#from data_functions_albert import remove_outliers

os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/Master tesi/Data & Code (Uganda)/data09')
pd.options.display.float_format = '{:,.2f}'.format

dollars = 2586.89



#%% AGRICULTURAL SEASON 1:

#Omit rents for evaluate agricultural productivity.


# =============================================================================
# Fertilizers & labor inputs
# =============================================================================

ag3a = pd.read_stata('agsec3a.dta')
ag3a = ag3a[["HHID", 'a3aq3', "a3aq8", "a3aq19", 'a3aq31', 'a3aq38', 'a3aq39', 'a3aq42a', 'a3aq42b','a3aq42c']]
ag3a['hhlabor'] = ag3a["a3aq39"].fillna(0) 
ag3a['hired_labor'] = ag3a["a3aq42a"].fillna(0)+ag3a["a3aq42b"].fillna(0)+ag3a["a3aq42c"].fillna(0) #Sum over hours men, women and kids. We assume all of them equally productive.

ag3a = ag3a[["HHID", 'a3aq3', "a3aq8", "a3aq19", "a3aq31",'hhlabor', 'hired_labor' ]]
ag3a.columns = ["HHID", 'pltid','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor']



# =============================================================================
# Crop choice and Seeds costs
# =============================================================================

ag4a = pd.read_stata('agsec4a.dta')
ag4a = ag4a[["HHID", 'a4aq2','a4aq5','a4aq6' , 'a4aq8',  "a4aq11"]]
ag4a.columns = ["HHID", 'pltid','cropID' , 'crop_code', 'area_planted', 'seed_cost']



# =============================================================================
# Output
# =============================================================================

ag5a = pd.read_stata('agsec5a.dta')
ag5a = ag5a[["HHID",'a5aq3',"a5aq4","a5aq6a","a5aq6c","a5aq6d","a5aq7a","a5aq7c","a5aq8","a5aq10","a5aq12","a5aq13","a5aq14a","a5aq14b","a5aq15","a5aq22"]]
ag5a.columns = ["HHID", 'pltid', "cropID", "total","unit", "tokg", "sell", "unit2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]



# Convert all quantitites to kilos:
#1.1 get median conversations (self-reported values)
conversion_kg = ag5a.groupby(by="unit")[["tokg"]].median()
conversion_kg.reset_index(inplace=True)
conversion_kg.loc[conversion_kg.unit==1, "tokg"] = 1
conversion_kg.columns = ["unit","kgconverter"]
ag5a = ag5a.merge(conversion_kg, on="unit", how="left")

# Convert to kg
ag5a[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]] = ag5a[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]].multiply(ag5a["kgconverter"], axis="index")


#1.2 Check reported quantities
ag5a["total"] = ag5a["total"].fillna(0)
ag5a["total2"] =  ag5a.loc[:,["sell","gift","cons","food_prod","animal", "seeds", "stored"]].sum(axis=1)
ag5a["diff_totals"] = ag5a.total -ag5a.total2
count_equal = len(ag5a.loc[ag5a.total==ag5a.total2])
count_bigger = len(ag5a.loc[ag5a.total>ag5a.total2])
count_smaller = len(ag5a.loc[ag5a.total<ag5a.total2])

#Prices
ag5a["prices"] = ag5a.value_sells.div(ag5a.sell, axis=0) 
prices = ag5a.groupby(by=["cropID"])[["prices"]].median()

#Check price values in Dollars: they make sense!!!!
prices_usd = ag5a.groupby(by=["cropID"])[["prices"]].median()/dollars
prices.reset_index(inplace=True)
prices.columns=["cropID","p_sell"]


ag5a = ag5a.merge(prices, on="cropID", how="left")

quant = ["total","total2","sell","gift","cons","food_prod","animal","seeds","stored"]
priceslist = ["p_sell"] 
#to check production value for the 3 type of prices uncomment:
#priceslist = ["p_sell", "p_c", "p_c_gate"] 
values_ag5a = ag5a[["HHID", 'pltid', 'cropID', "trans_cost"]]
#Generate values for each quantities and each type of price. Now I only use for sellings prices since the consumption ones where to big.
for q in quant:
    for p in priceslist:
        values_ag5a[q+"_value_"+p] = ag5a[q]*ag5a[p]


# Summarize results for crop
ag5acrop = values_ag5a.groupby(by=["HHID", 'cropID']).sum()
ag5acrop= ag5acrop.replace(0, np.nan)
ag5acrop = ag5acrop.dropna(subset=['total_value_p_sell'])
ag5acrop = ag5acrop.reset_index()
crop_count = pd.value_counts(ag5acrop['cropID'])/ len(ag5acrop)
crop_sum = ag5acrop.groupby(by=['cropID']).mean()/dollars
#Diversification of crops per household
hh_count = pd.value_counts(ag5acrop['HHID']).to_frame() 
hh_count = pd.value_counts(hh_count['HHID'])/len(hh_count)

ag5acrop = ag5acrop.reset_index()
sumag5acrop = ag5acrop.describe()/dollars


ag5a = values_ag5a.groupby(by=["HHID",  'pltid']).sum()
ag5a = ag5a.reset_index()
ag5a = ag5a.drop_duplicates(subset=['HHID','pltid'], keep=False)



#Only the plots with one crop




# =============================================================================
# Capital of the plots/households
# =============================================================================
'''
ag10 = pd.read_stata('agsec10.dta')
ag10 = ag10[['HHID', 'a10q2','a10q3','a10q7','a10q8']]
ag10 = ag10[['HHID', 'a10q2']]
ag10 = ag10.groupby(by="HHID").sum()
ag10 = ag10.reset_index()
ag10.columns = ['HHID','farm_capital']

ag11 = pd.read_stata('agsec11.dta')
ag11 = ag11[['HHID', 'AGroup_ID','a11q3']]
'''
#Non-informative. Only information about yes/no use of animals but not quantity or value.

# Merge datasets -------------------------------------------

agrica = pd.merge(ag3a, ag4a, on=['HHID','pltid'], how='outer')
agrica = pd.merge(agrica, ag5a, on=['HHID','pltid'], how='right')
#agrica = pd.merge(agrica, ag10, on='HHID', how='right')
agrica.set_index(['HHID','pltid'], inplace=True)
agrica = agrica.reset_index()
agrica = agrica.drop_duplicates(subset=['HHID','pltid'], keep=False)

del ag3a, ag4a, ag5a, ag5acrop, conversion_kg, count_bigger, count_equal, count_smaller, crop_count, crop_sum, p, prices, prices_usd, priceslist, q, quant, values_ag5a

#crop in production and planting coincide so we can eliminate one of them (in importing 2agsec4 do not import crop)


#%% computing productivity levels
agrica['season'] = 1
agrica['m'] = agrica['org_fert'].fillna(0)+ agrica['chem_fert'].fillna(0)+ agrica['pesticides'].fillna(0)+ agrica['seed_cost'].fillna(0)
agrica['l'] = agrica['hhlabor'].fillna(0)+ agrica['hired_labor'].fillna(0)
agrica['A'] = agrica['area_planted']
agrica['y'] = agrica['total2_value_p_sell']

agrica['y_over_A'] = agrica['y']/agrica['A']

variables = [ 'm', 'l', 'A', 'y', 'y_over_A']
for var in variables:
    agrica['ln'+var] = np.log(agrica[var].dropna()).replace(-np.inf, np.nan)
    

lnA = np.log(agrica['A'].dropna())
lnA = lnA.replace(-np.inf, np.nan)
lnA = lnA.dropna()

lny = (np.log(agrica['y'].dropna()).replace(-np.inf, np.nan)).dropna()

lnm = (np.log(agrica['m'].dropna()).replace(-np.inf, np.nan)).dropna()
lny_over_A = (np.log(agrica['y_over_A'].dropna()).replace([-np.inf,np.inf], np.nan)).dropna()


#Plot hours distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(agrica['l'], label="L")
plt.title('Distribution of Labour in farm in Uganda 2013-2014')
plt.xlabel('Farm Labour')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Area plot distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnA, label="A")
plt.title('Distribution of Plots Area in Uganda 2013-2014')
plt.xlabel('Plot Area (in Acres)')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Area plot distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnm, label="m")
plt.title('Distribution of inputs in Uganda 2013-2014')
plt.xlabel('log of inputs')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny, label="y")
plt.title('Distribution of Production in Uganda 2013-2014')
plt.xlabel('Agricultural Production')
plt.ylabel("Density")
plt.legend()
plt.show()


#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny_over_A, label="y/A")
plt.title('Distribution of Production per Acre in Uganda 2013-2014')
plt.xlabel('Agricultural Production per Acre')
plt.ylabel("Density")
plt.legend()
plt.show()


#%% OLS regression



count_crops = pd.value_counts(agrica['cropID'])



#%% Whole Sample
data = agrica
ols31= sm.ols(formula=" lny ~ lnk +lnA +lnm +lnl", data=data).fit()
ols31.summary()

#Test cobb-douglas form:
ftest = ols31.f_test(" lnk +lnm +lnA +lnl = 1")
print(ftest)
# NO REJECT HYPOTHESIS COEFFICIENTS SUM UP TO 1

ols32= sm.ols(formula=" lny_over_A ~ lnk +lnm +lnl", data=data).fit()
ols32.summary()



###################################################################################################
###################################################################################################





#%% AGRICULTURAL SEASON 2:


# =============================================================================
# Crop choice and Seeds costs
# =============================================================================

ag4a = pd.read_stata('agsec4a.dta')
ag4a = ag4a[["HHID", 'a4aq2','a4aq5','a4aq6' , 'a4aq8',  "a4aq11"]]
ag4a.columns = ["HHID", 'pltid','cropID' , 'crop_code', 'area_planted', 'seed_cost']


# =============================================================================
# Fertilizers & labor inputs
# =============================================================================

ag3b = pd.read_stata('agsec3b.dta')
ag3b = ag3b[["HHID", 'a3bq3', "a3bq8", "a3bq19", 'a3bq31', 'a3bq38', 'a3bq39', 'a3bq42a', 'a3bq42b','a3bq42c']]
ag3b['hhlabor'] = ag3b["a3bq39"].fillna(0) 
ag3b['hired_labor'] = ag3b["a3bq42a"].fillna(0)+ag3b["a3bq42b"].fillna(0)+ag3b["a3bq42c"].fillna(0) #Sum over hours men, women and kids. We assume all of them equally productive.

ag3b = ag3b[["HHID", 'a3bq3', "a3bq8", "a3bq19", "a3bq31",'hhlabor', 'hired_labor']]
ag3b.columns = ["HHID", 'pltid','org_fert', 'chem_fert', 'pesticides', 'hhlabor', 'hired_labor']



# =============================================================================
# Crop choice and Seeds costs
# =============================================================================

ag4b = pd.read_stata('agsec4b.dta')
ag4b = ag4b[["HHID", 'a4bq2','a4bq6', 'a4bq8', 'a4bq11']]

ag4b.columns = ["HHID", 'pltid','cropID','area_planted', 'seed_cost']
#COST


# =============================================================================
# Output
# =============================================================================

ag5b = pd.read_stata('agsec5b.dta')
ag5b = ag5b[["HHID",'a5bq3',"a5bq4","a5bq6a","a5bq6c","a5bq6d","a5bq7a","a5bq7c","a5bq8","a5bq10","a5bq12","a5bq13","a5bq14a","a5bq14b","a5bq15","a5bq22"]]
ag5b.columns = ["HHID", 'pltid', "cropID", "total","unit", "tokg", "sell", "unit2", "value_sells", "trans_cost", "gift", "cons", "food_prod", "animal", "seeds", "stored"]



# Convert all quantitites to kilos:
#1.1 get median conversations (self-reported values)
conversion_kg = ag5b.groupby(by="unit")[["tokg"]].median()
conversion_kg.reset_index(inplace=True)
conversion_kg.loc[conversion_kg.unit==1, "tokg"] = 1
conversion_kg.columns = ["unit","kgconverter"]
ag5b = ag5b.merge(conversion_kg, on="unit", how="left")

# Convert to kg
ag5b[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]] = ag5b[["total", "sell", "gift", "cons", "food_prod", "animal", "seeds", "stored"]].multiply(ag5b["kgconverter"], axis="index")

#1.2 Check reported quantities
ag5b["total"] = ag5b["total"].fillna(0)
ag5b["total2"] =  ag5b.loc[:,["sell","gift","cons","food_prod","animal", "seeds", "stored"]].sum(axis=1)
ag5b["diff_totals"] = ag5b.total -ag5b.total2
count_equal = len(ag5b.loc[ag5b.total==ag5b.total2])
count_bigger = len(ag5b.loc[ag5b.total>ag5b.total2])
count_smaller = len(ag5b.loc[ag5b.total<ag5b.total2])

#Prices
ag5b["prices"] = ag5b.value_sells.div(ag5b.sell, axis=0) 
prices = ag5b.groupby(by=["cropID"])[["prices"]].median()

#Check price values in Dollars: they make sense!!!!
prices_usd = ag5b.groupby(by=["cropID"])[["prices"]].median()/dollars
prices.reset_index(inplace=True)
prices.columns=["cropID","p_sell"]



ag5b = ag5b.merge(prices, on="cropID", how="left")


quant = ["total","total2","sell","gift","cons","food_prod","animal","seeds","stored"]
priceslist = ["p_sell"] 
#to check production value for the 3 type of prices uncomment:
#priceslist = ["p_sell", "p_c", "p_c_gate"] 
values_ag5b = ag5b[["HHID", 'pltid', 'cropID', "trans_cost"]]
#Generate values for each quantities and each type of price. Now I only use for sellings prices since the consumption ones where to big.
for q in quant:
    for p in priceslist:
        values_ag5b[q+"_value_"+p] = ag5b[q]*ag5b[p]


# Summarize results for crop
ag5bcrop = values_ag5b.groupby(by=["HHID", 'cropID']).sum()
ag5bcrop= ag5bcrop.replace(0, np.nan)
ag5bcrop = ag5bcrop.dropna(subset=['total_value_p_sell'])
ag5bcrop = ag5bcrop.reset_index()
crop_count = pd.value_counts(ag5bcrop['cropID'])/ len(ag5bcrop)
crop_sum = ag5bcrop.groupby(by=['cropID']).mean()/dollars
#Diversification of crops per household
hh_count = pd.value_counts(ag5bcrop['HHID']).to_frame() 
hh_count = pd.value_counts(hh_count['HHID'])/len(hh_count)

ag5bcrop = ag5bcrop.reset_index()
sumag5bcrop = ag5bcrop.describe()/dollars


ag5b = values_ag5b.groupby(by=["HHID",  'pltid']).sum()
ag5b = ag5b.reset_index()
ag5b = ag5b.drop_duplicates(subset=['HHID','pltid'], keep=False)



#Only the plots with one crop




'''
ag11 = pd.read_stata('agsec11.dta')
ag11 = ag11[['HHID', 'AGroup_ID','a11q3']]
'''
#Non-informative. Only information about yes/no use of animals but not quantity or value.

# Merge datasets -------------------------------------------

agricb= pd.merge(ag3b, ag4b, on=['HHID','pltid'], how='outer')
agricb = pd.merge(agricb, ag5b, on=['HHID','pltid'], how='right')
agricb.set_index(['HHID','pltid'], inplace=True)
agricb = agricb.reset_index()
agricb = agricb.drop_duplicates(subset=['HHID','pltid'], keep=False)

del ag3b, ag4b, ag5b, ag5bcrop, conversion_kg, count_bigger, count_equal, count_smaller, crop_count, crop_sum, p, prices, prices_usd, priceslist, q, quant, values_ag5b

#crop in production and planting coincide so we can eliminate one of them (in importing 2agsec4 do not import crop)


#%% computing productivity levels
agricb['season'] = 1
#agricb['k'] = agricb['farm_capital']
agricb['m'] = agricb['org_fert'].fillna(0)+ agricb['chem_fert'].fillna(0)+ agricb['pesticides'].fillna(0)+ agricb['seed_cost'].fillna(0)
agricb['l'] = agricb['hhlabor'].fillna(0)+ agricb['hired_labor'].fillna(0)
agricb['A'] = agricb['area_planted']
agricb['y'] = agricb['total2_value_p_sell']

agricb['y_over_A'] = agricb['y']/agricb['A']

variables = [ 'm', 'l', 'A', 'y', 'y_over_A']
for var in variables:
    agricb['ln'+var] = np.log(agricb[var].dropna()).replace(-np.inf, np.nan)
    


lnA = np.log(agricb['A'].dropna())
lnA = lnA.replace(-np.inf, np.nan)
lnA = lnA.dropna()

lny = (np.log(agricb['y'].dropna()).replace(-np.inf, np.nan)).dropna()

lnm = (np.log(agricb['m'].dropna()).replace(-np.inf, np.nan)).dropna()
lny_over_A = (np.log(agricb['y_over_A'].dropna()).replace([-np.inf,np.inf], np.nan)).dropna()



#Plot hours distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(agricb['l'], label="L")
plt.title('Distribution of Labour in farm in Uganda 2013-2014')
plt.xlabel('Farm Labour')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Area plot distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnA, label="A")
plt.title('Distribution of Plots Area in Uganda 2013-2014')
plt.xlabel('Plot Area (in Acres)')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Area plot distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnm, label="m")
plt.title('Distribution of inputs in Uganda 2013-2014')
plt.xlabel('log of inputs')
plt.ylabel("Density")
plt.legend()
plt.show()

#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny, label="y")
plt.title('Distribution of Production in Uganda 2013-2014')
plt.xlabel('Agricultural Production')
plt.ylabel("Density")
plt.legend()
plt.show()


#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny_over_A, label="y/A")
plt.title('Distribution of Production per Acre in Uganda 2013-2014')
plt.xlabel('Agricultural Production per Acre')
plt.ylabel("Density")
plt.legend()
plt.show()





#%% Whole Sample
data = agricb
ols31= sm.ols(formula=" lny ~ +lnA +lnm +lnl", data=data).fit()
ols31.summary()

#Test cobb-douglas form:
ftest = ols31.f_test(" +lnm +lnA +lnl = 1")
print(ftest)
# REJECT HYPOTHESIS COEFFICIENTS SUM UP TO 1

###################################################################################################
###################################################################################################


#%% Both seasons together


data = agrica.append(agricb)


count_crops = pd.value_counts(data['cropID'])

data_cassava= data.loc[data['cropID']=='Cassava', :]
ols1= sm.ols(formula=" lny ~  lnA +lnm +lnl", data=data_cassava).fit()
ols1.summary()


data_swpotatoes= data.loc[data['cropID']=='SWEET POTATOES', :]
ols2= sm.ols(formula=" lny ~  lnA +lnm +lnl", data=data_swpotatoes).fit()
ols2.summary()


data_beans= data.loc[data['cropID']=='BEANS', :]
ols3= sm.ols(formula=" lny ~  lnA +lnm  +lnl", data=data_beans).fit()
ols3.summary()


data_maize= data.loc[data['cropID']=='MAIZE', :]
ols4= sm.ols(formula=" lny ~  lnA +lnm +lnl", data=data_maize).fit()
ols4.summary()

data_groundnuts= data.loc[data['cropID']=='GROUNDNUTS', :]
ols5= sm.ols(formula=" lny ~  lnA +lnm +lnl", data=data_groundnuts).fit()
ols5.summary()

data_bananafood= data.loc[data['cropID']=='BANANA FOOD', :]
ols6= sm.ols(formula=" lny ~  lnA +lnm +lnl", data=data_bananafood).fit()
ols6.summary()

data_sorghum= data.loc[data['cropID']=='SORGHUM', :]
ols7= sm.ols(formula=" lny ~ lnA +lnm +lnl", data=data_sorghum).fit()
ols7.summary()

results = summary_col([ols1, ols2, ols3, ols4, ols5, ols7],stars=True)
print(results)


#%% As in the model
data_cassava= data.loc[data['cropID']=='CASSAVA', :]
ols1= sm.ols(formula=" lny ~  +lnm", data=data_cassava).fit()
ols1.summary()


data_swpotatoes= data.loc[data['cropID']=='SWEET POTATOES', :]
ols2= sm.ols(formula=" lny ~  +lnm", data=data_swpotatoes).fit()
ols2.summary()


data_beans= data.loc[data['cropID']=='BEANS', :]
ols3= sm.ols(formula=" lny ~  +lnm", data=data_beans).fit()
ols3.summary()


data_maize= data.loc[data['cropID']=='MAIZE', :]
ols4= sm.ols(formula=" lny ~  +lnm", data=data_maize).fit()
ols4.summary()

data_groundnuts= data.loc[data['cropID']=='GROUNDNUTS', :]
ols5= sm.ols(formula=" lny ~  +lnm", data=data_groundnuts).fit()
ols5.summary()

data_bananafood= data.loc[data['cropID']=='BANANA FOOD', :]
ols6= sm.ols(formula=" lny ~  +lnm", data=data_bananafood).fit()
ols6.summary()

data_sorghum= data.loc[data['cropID']=='SORGHUM', :]
ols7= sm.ols(formula=" lny ~  +lnm", data=data_sorghum).fit()
ols7.summary()


results = summary_col([ols1, ols2, ols3, ols4, ols5, ols7],stars=True)
print(results)


results_short = summary_col([ols1, ols2, ols3, ols4, ols5, ols7],stars=True)
print(results_short)







#%% Whole Sample

ols31= sm.ols(formula=" lny ~ lnk +lnA +lnm +lnl", data=data).fit()
ols31.summary()

#Test cobb-douglas form:
ftest = ols31.f_test(" lnk +lnm +lnA +lnl = 1")
print(ftest)