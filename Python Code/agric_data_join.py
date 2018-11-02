# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:29:34 2018

@author: rodri
"""

### Apppend datasets.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col
os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/Research/python Albert')
from data_functions_albert import remove_outliers
os.chdir('C:/Users/rodri/OneDrive/Documentos/IDEA/phd tesi/data')
pd.options.display.float_format = '{:,.2f}'.format

dollars = 2586.89

#%% Import data
data13 = pd.read_csv('agric_data13.csv')
crop_codes = pd.read_csv('crop_codes.csv')
crop_codes.columns = ['cropID','crop_name']
data13 = pd.merge(data13, crop_codes, on='cropID', how='left' )
data13['crop_name'] = data13['crop_name'].str.upper()
data13['wave'] = '2013-14'

data11 = pd.read_csv('agric_data11.csv')
data11['crop_name'] = data11['cropID'].str.upper()
data11['wave'] = '2011-12'
data10 = pd.read_csv('agric_data10.csv')
data10.rename(columns={'cropID':'crop_name'}, inplace=True)
data10['wave'] = '2010-11'
del data13['cropID'], data11['cropID']

data = data13.append(data11)
data = data.append(data10)

#%% Plot distributions
lnk = np.log(data['k'].dropna())
lnk = lnk.replace(-np.inf, np.nan)
lnk = lnk.dropna()

lnA = np.log(data['A'].dropna())
lnA = lnA.replace(-np.inf, np.nan)
lnA = lnA.dropna()

lny = (np.log(data['y'].dropna()).replace(-np.inf, np.nan)).dropna()

lnm = (np.log(data['m'].dropna()).replace(-np.inf, np.nan)).dropna()
lny_over_A = (np.log(data['y_over_A'].dropna()).replace([-np.inf,np.inf], np.nan)).dropna()

#Plot Capital distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnk, label="K")
plt.title('Distribution of Farm Capital in Uganda 2013-2014')
plt.xlabel('Log of Farm Capital')
plt.ylabel("Density")
plt.legend()
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/K_distribution.png')

#Plot hours distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(data['l'].dropna(), label="L")
plt.title('Distribution of Labour in farm in Uganda 2013-2014')
plt.xlabel('Farm Labour')
plt.ylabel("Density")
plt.legend()
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/L_distribution.png')


#Plot Area plot distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnA, label="A")
plt.title('Distribution of Plots Area in Uganda 2013-2014')
plt.xlabel('Plot Area (in Acres)')
plt.ylabel("Density")
plt.legend()
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/A_distribution.png')

#Plot Inputs distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lnm, label="m")
plt.title('Distribution of inputs in Uganda 2013-2014')
plt.xlabel('log of inputs')
plt.ylabel("Density")
plt.legend()
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/M_distribution.png')


#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny, label="y")
plt.title('Distribution of Production in Uganda 2013-2014')
plt.xlabel('Agricultural Production')
plt.ylabel("Density")
plt.legend()
plt.show()

fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/output_distribution.png')


#Plot Agricultural production distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.distplot(lny_over_A, label="y/A")
plt.title('Distribution of Production per Acre in Uganda 2013-2014')
plt.xlabel('Agricultural Production per Acre')
plt.ylabel("Density")
plt.legend()
plt.show()
fig.savefig('C:/Users/rodri/OneDrive/Documentos/IDEA/Phd tesi/figures/yield_distribution.png')


#%% OLS: Does Production function follows a cobb-douglas form?
count_crops = pd.value_counts(data['crop_name']).to_frame()
count_crops = count_crops.reset_index()

list_crops = count_crops.iloc[0:18,0]
del list_crops[12], list_crops[15]

list_ols = []
list_ftest = []
list_n = []


for item in list_crops:
     print(item)
     ols= sm.ols(formula=" lny ~ lnk + lnA +lnm +lnl", data=data.loc[data['crop_name']==item, :]).fit()
     print(ols.summary())  
     ftest = ols.f_test(" lnk +lnm +lnA +lnl = 1")
     list_ols.append(ols)
     list_ftest.append(ftest)
     n = len(ols.fittedvalues)
     list_n.append(n)

results_1 = summary_col([list_ols[0], list_ols[1], list_ols[2], list_ols[3], list_ols[4], list_ols[5], list_ols[6], list_ols[7]],stars=True)
results_1 = summary_col([ list_ols[1], list_ols[2], list_ols[4], list_ols[5], list_ols[6], list_ols[7]],stars=True)

print(results_1)
print(results_1.as_latex())


results_2 = summary_col([list_ols[8], list_ols[9], list_ols[11],  list_ols[12], list_ols[13], list_ols[14]],stars=True)
print(results_2.as_latex())




ftests= pd.DataFrame(np.array([list_ftest[0].fvalue[0,0], list_ftest[0].pvalue, list_ftest[1].fvalue[0,0], list_ftest[1].pvalue, list_ftest[2].fvalue[0,0], list_ftest[2].pvalue, list_ftest[3].fvalue[0,0], list_ftest[3].pvalue, list_ftest[4].fvalue[0,0], list_ftest[4].pvalue, list_ftest[5].fvalue[0,0], list_ftest[5].pvalue, list_ftest[6].fvalue[0,0], list_ftest[6].pvalue, list_ftest[7].fvalue[0,0], list_ftest[7].pvalue]))
ftests2= pd.DataFrame(np.array([list_ftest[8].fvalue[0,0], list_ftest[8].pvalue, list_ftest[9].fvalue[0,0], list_ftest[9].pvalue, list_ftest[10].fvalue[0,0], list_ftest[10].pvalue, list_ftest[11].fvalue[0,0], list_ftest[11].pvalue, list_ftest[12].fvalue[0,0], list_ftest[12].pvalue, list_ftest[13].fvalue[0,0], list_ftest[13].pvalue, list_ftest[14].fvalue[0,0], list_ftest[14].pvalue, list_ftest[15].fvalue[0,0], list_ftest[15].pvalue]))

list_f1 = [list_ftest[1].fvalue[0,0], list_ftest[2].fvalue[0,0],  list_ftest[4].fvalue[0,0],  list_ftest[5].fvalue[0,0], list_ftest[6].fvalue[0,0],  list_ftest[7].fvalue[0,0]]
list_pvalue1 = [list_ftest[1].pvalue, list_ftest[2].pvalue, list_ftest[4].pvalue,  list_ftest[5].pvalue, list_ftest[6].pvalue, list_ftest[7].pvalue ]


ftests= pd.DataFrame(np.array([list_ftest[0].fvalue[0,0], list_ftest[0].pvalue, list_ftest[1].fvalue[0,0], list_ftest[1].pvalue, list_ftest[2].fvalue[0,0], list_ftest[2].pvalue, list_ftest[3].fvalue[0,0], list_ftest[3].pvalue, list_ftest[4].fvalue[0,0], list_ftest[4].pvalue, list_ftest[5].fvalue[0,0], list_ftest[5].pvalue, list_ftest[6].fvalue[0,0], list_ftest[6].pvalue, list_ftest[7].fvalue[0,0], list_ftest[7].pvalue]))
ftests2= pd.DataFrame(np.array([list_ftest[8].fvalue[0,0], list_ftest[8].pvalue, list_ftest[9].fvalue[0,0], list_ftest[9].pvalue, list_ftest[10].fvalue[0,0], list_ftest[10].pvalue, list_ftest[11].fvalue[0,0], list_ftest[11].pvalue, list_ftest[12].fvalue[0,0], list_ftest[12].pvalue, list_ftest[13].fvalue[0,0], list_ftest[13].pvalue, list_ftest[14].fvalue[0,0], list_ftest[14].pvalue, list_ftest[15].fvalue[0,0], list_ftest[15].pvalue]))

list_f2 = [list_ftest[8].fvalue[0,0], list_ftest[9].fvalue[0,0],  list_ftest[11].fvalue[0,0],  list_ftest[12].fvalue[0,0], list_ftest[13].fvalue[0,0],  list_ftest[14].fvalue[0,0]]
list_pvalue2 = [list_ftest[8].pvalue, list_ftest[9].pvalue, list_ftest[11].pvalue,  list_ftest[12].pvalue, list_ftest[13].pvalue, list_ftest[14].pvalue ]

pd.options.display.float_format = '{:,.4f}'.format
ftests = pd.concat([ftests, ftests2], axis=1)


#%% As in the model


list_ols_short = []
list_ftest_short = []
list_n_short = []


for item in list_crops:
     print(item)
     ols= sm.ols(formula=" lny ~ lnm ", data=data.loc[data['crop_name']==item, :]).fit()
     print(ols.summary())  
     list_ols_short.append(ols)
     n = len(ols.fittedvalues)
     list_n_short.append(n)

results_1_short = summary_col([list_ols_short[0], list_ols_short[1], list_ols_short[2], list_ols_short[3], list_ols_short[4], list_ols_short[5], list_ols_short[6], list_ols_short[7]],stars=True)
print(results_1_short)
results_2_short = summary_col([list_ols_short[8], list_ols_short[9], list_ols_short[10], list_ols_short[11], list_ols_short[12], list_ols_short[13], list_ols_short[14], list_ols_short[15]],stars=True)
print(results_2_short)



pd.options.display.float_format = '{:,.4f}'.format
ftests = pd.concat([ftests, ftests2], axis=1)

#%% Risk analysis

list_crops = count_crops.iloc[0:18,0]
del list_crops[12], list_crops[15]

list_avg_prod = []
for item in list_crops:
    avg_prod = data.loc[data['crop_name']==item, :].groupby(by=['wave','season'])['y'].sum()
    list_avg_prod.append(avg_prod)

var_list=[]
for i in range(len(list_avg_prod)):
    var = np.std(list_avg_prod[i])/dollars
    var_list.append(var)


#%% Sample statistics per crop
data = data13.append(data11)
data = data.append(data10)

count_crops = pd.value_counts(data['crop_name']).to_frame()
count_crops = count_crops.reset_index()
list_crops = count_crops.iloc[0:18,0]
del list_crops[12], list_crops[15]

crop_summary = []

for item in list_crops:  
     data_crop=data.loc[data['crop_name']==item, ['y','y_over_A','A','k','m','l','chem_fert', 'org_fert', 'pesticides']]
     summary = data_crop.describe()
     crop_summary.append(summary)

sum_data = data[['y','y_over_A','A','k','m','l','chem_fert', 'org_fert', 'pesticides']].describe()
sum_cassava = crop_summary[0]
sum_swpotatoes = crop_summary[1]
sum_beans = crop_summary[2]
sum_bananafood = crop_summary[3]
sum_maize = crop_summary[4]
sum_groundnuts = crop_summary[5]
sum_sorghum = crop_summary[6]
sum_fingermillet = crop_summary[7]
sum_simsim = crop_summary[8]
sum_irishpotatoes = crop_summary[9]
sum_coffee = crop_summary[10]
sum_rice = crop_summary[11]
sum_sunflower = crop_summary[12]
sum_soyabean =crop_summary[13]
sum_fieldpeas = crop_summary[14]
sum_cotton = crop_summary[15]
