# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:18:39 2023

@author: 15485
"""

import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\15485\Desktop\UWaterloo_Academics\ECE657A\Assignments\Assignment1\abalone.csv', 
                 names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 
                      'Sucked_weight', 'Viscera_weight', 'Shell_weight',
                                          'Rings'], sep = ',')
df.isna().sum() #check if there is any missing data. for abalone dataset, there is no missing data
'''
Q. is there any missing data?
A. No, there is no missing data
'''
len(df)  #Determine the number of rows
df.shape #check the dimesnionality of the dataframe
df.info() #check the datatype of the column

#%%
print(df.head())
print(df.describe()) #get overviews of the values in the dataset

#Do some exploratory data analysis
sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True, rc=None)
pairplot = sns.pairplot(df, height=1.5, aspect=2, hue = 'Rings')

#To check if the datatset is balanced or not. Answer: The dataset is not balanced.
print(len(df['Rings'].unique()))
print(df['Rings'].value_counts(sort=True))   #use normalize = True to get the proprtion value

#%%
#Applying Z Normalization
df_sex = df['Sex']
df1 = df[df.columns.difference(['Sex'])]

def z_normalization(dataframe):
    df_zn_dataframe = dataframe.copy()
    for column in df_zn_dataframe:
        df_zn_dataframe[column] = (df_zn_dataframe[column] - df_zn_dataframe[column].mean()) / df_zn_dataframe[column].std()
    return df_zn_dataframe

df_z_normalized = z_normalization(df1)

print(df_z_normalized)
final_df = df_z_normalized.join(df_sex)
final_df_cols = final_df.columns.tolist()
final_df_cols = final_df_cols[-1: ] + final_df_cols[:-1]
print(final_df_cols)    
processed_df = final_df[final_df_cols]
print(processed_df)

#%%





