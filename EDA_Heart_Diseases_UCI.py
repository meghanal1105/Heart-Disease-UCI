# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 22:19:43 2021

@author: 91876
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.formula.api import ols
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import svm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


heart_disease_UCI= pd.read_csv("C:/Users/91876/Desktop/Kaggle Individual/3_Heart Disease UCI/heart.csv")
df= pd.DataFrame(heart_disease_UCI)
df.info()
df.isna().sum()
df.describe()
df.columns
df.shape


''' Target Variable '''
''' target '''
df.target.isna().sum()
df.target.value_counts().sort_index()
df.target.describe()
df.target.unique()
''' Categorical '''
# No null values

#Countplot
sns.countplot(x ='target', data = df)





'''########### 1- age  ################'''
df.age.isna().sum()
df.age.value_counts().sort_index()
df.age.describe()
df.age.unique()
''' Continuous '''
# No null values

#Histogram
sns.distplot(df.age, color = 'red')
plt.xlabel('age')
plt.title('Histogram of age')

#Boxplot
plt.boxplot(df['age'],1,'rs',1)
plt.xlabel('age')
plt.ylabel('counts')
plt.title('Boxplot of age')

# there are no outliers





'''############ 2- sex #################'''
df.sex.isna().sum()
df.sex.value_counts().sort_index()
df.sex.describe()
df.sex.unique()
''' Categorical '''
# No null values

#Countplot
sns.countplot(x ='sex', data = df)





'''############# 3- cp ###############'''
df.cp.isna().sum()
df.cp.value_counts().sort_index()
df.cp.describe()
df.cp.unique()
''' Categorical '''
# No null values

#Countplot
sns.countplot(x ='cp', data = df)

#Individual boxplot
sns.boxplot(x='cp', y='target', data = df)
tukey3= pairwise_tukeyhsd(df.target, df.cp, alpha=0.05)
print(tukey3)

#Merging
df.cp.value_counts().sort_index()
df['cp'].replace(2,1, inplace = True)
df['cp'].replace(3,1, inplace = True)
df.cp.value_counts().sort_index()

#Countplot after merging
sns.countplot(x ='cp', data = df)




df.info()
'''############### 4- trestbps ###############'''
df.trestbps.isna().sum()
df.trestbps.value_counts().sort_index()
df.trestbps.describe()
df.trestbps.unique()
''' Continuous '''
# No null values

#Histogram
sns.distplot(df.trestbps, color = 'red')
plt.xlabel('trestbps')
plt.title('Histogram of trestbps')

#Boxplot
plt.boxplot(df['trestbps'],1,'rs',1)
plt.xlabel('trestbps')
plt.ylabel('counts')
plt.title('Boxplot of trestbps')

# there are outliers

# Outliers Count
IQR4 = df['trestbps'].quantile(0.75) - df['trestbps'].quantile(0.25)
IQR4

UL4 = df['trestbps'].quantile(0.75) + (1.5*IQR4)
UL4

df.trestbps[(df.trestbps > UL4)].value_counts().sum()
# 9

df.trestbps = np.where(df.trestbps > UL4, UL4, df.trestbps)

df.trestbps[(df.trestbps > UL4)].value_counts().sum()
# 0

#Boxplot after treatment
plt.boxplot(df['trestbps'],1,'rs',1)
plt.xlabel('trestbps')
plt.ylabel('counts')
plt.title('Boxplot of trestbps')






'''############### 5- chol ################'''
df.chol.isna().sum()
df.chol.value_counts().sort_index()
df.chol.describe()
df.chol.unique()
''' Continuous '''
# No null values

#Histogram
sns.distplot(df.chol, color = 'red')
plt.xlabel('chol')
plt.title('Histogram of chol')

#Boxplot
plt.boxplot(df['chol'],1,'rs',1)
plt.xlabel('chol')
plt.ylabel('counts')
plt.title('Boxplot of chol')

# there are outliers

# Outliers Count
IQR5 = df['chol'].quantile(0.75) - df['chol'].quantile(0.25)
IQR5

UL5 = df['chol'].quantile(0.75) + (1.5*IQR5)
UL5

df.chol[(df.chol > UL5)].value_counts().sum()
# 5

df.chol = np.where(df.chol > UL5, UL5, df.chol)

df.chol[(df.chol > UL5)].value_counts().sum()
# 0

#Boxplot after treatment
plt.boxplot(df['chol'],1,'rs',1)
plt.xlabel('chol')
plt.ylabel('counts')
plt.title('Boxplot of chol')






'''############### 6- fbs ################'''
df.fbs.isna().sum()
df.fbs.value_counts().sort_index()
df.fbs.describe()
df.fbs.unique()
''' Categorical '''
# No null values

#Countplot
sns.countplot(x ='fbs', data = df)






'''############## 7- restecg ##################'''
df.restecg.isna().sum()
df.restecg.value_counts().sort_index()
df.restecg.describe()
df.restecg.unique()
''' Categorical '''
# No null values

#Countplot
sns.countplot(x ='restecg', data = df)

#Individual boxplot
sns.boxplot(x='restecg', y='target', data = df)
tukey7= pairwise_tukeyhsd(df.target, df.restecg, alpha=0.05)
print(tukey7)

#Merging
df.restecg.value_counts().sort_index()
df['restecg'].replace(2,1, inplace = True)
df.restecg.value_counts().sort_index()

#Countplot after merging
sns.countplot(x ='restecg', data = df)





'''############### 8- thalach ##################'''
df.thalach.isna().sum()
df.thalach.value_counts().sort_index()
df.thalach.describe()
df.thalach.unique()
''' Continuous '''
# No null values

#Histogram
sns.distplot(df.thalach, color = 'red')
plt.xlabel('thalach')
plt.title('Histogram of thalach')

#Boxplot
plt.boxplot(df['thalach'],1,'rs',1)
plt.xlabel('thalach')
plt.ylabel('counts')
plt.title('Boxplot of thalach')

# there are outliers

# Outliers Count
IQR8 = df['thalach'].quantile(0.75) - df['thalach'].quantile(0.25)
IQR8

LL8 = df['thalach'].quantile(0.25) - (1.5*IQR8)
LL8

df.thalach[(df.thalach < LL8)].value_counts().sum()
# 1

df.thalach = np.where(df.thalach < LL8, LL8, df.thalach)

df.thalach[(df.thalach < LL8)].value_counts().sum()
# 0

#Boxplot after treatment
plt.boxplot(df['thalach'],1,'rs',1)
plt.xlabel('thalach')
plt.ylabel('counts')
plt.title('Boxplot of thalach')





'''############### 9- exang ###################'''
df.exang.isna().sum()
df.exang.value_counts().sort_index()
df.exang.describe()
df.exang.unique()
''' Categorical '''
# No null values

#Countplot
sns.countplot(x ='exang', data = df)

# only 2 categories so no merging required





'''################ 10- oldpeak ##################'''
df.oldpeak.isna().sum()
df.oldpeak.value_counts().sort_index()
df.oldpeak.describe()
df.oldpeak.unique()
''' Continuous '''
# No null values

#Histogram
sns.distplot(df.oldpeak, color = 'red')
plt.xlabel('oldpeak')
plt.title('Histogram of oldpeak')

#Boxplot
plt.boxplot(df['oldpeak'],1,'rs',1)
plt.xlabel('oldpeak')
plt.ylabel('counts')
plt.title('Boxplot of oldpeak')

# there are outliers

# Outliers Count
IQR10 = df['oldpeak'].quantile(0.75) - df['oldpeak'].quantile(0.25)
IQR10

UL10 = df['oldpeak'].quantile(0.75) + (1.5*IQR10)
UL10

df.oldpeak[(df.oldpeak > UL10)].value_counts().sum()
# 5

df.oldpeak = np.where(df.oldpeak > UL10, UL10, df.oldpeak)

df.oldpeak[(df.oldpeak > UL10)].value_counts().sum()
# 0

#Boxplot after treatment
plt.boxplot(df['oldpeak'],1,'rs',1)
plt.xlabel('oldpeak')
plt.ylabel('counts')
plt.title('Boxplot of oldpeak')





'''############### 11- slope #################'''
df.slope.isna().sum()
df.slope.value_counts().sort_index()
df.slope.describe()
df.slope.unique()
''' Categorical '''
# No null values

#Countplot
sns.countplot(x ='slope', data = df)

#Individual boxplot
sns.boxplot(x='slope', y='target', data = df)
tukey11= pairwise_tukeyhsd(df.target, df.slope, alpha=0.05)
print(tukey11)

#Merging
df.slope.value_counts().sort_index()
df['slope'].replace(0,1, inplace = True)
df.slope.value_counts().sort_index()

#Countplot after merging
sns.countplot(x ='slope', data = df)





'''############### 12- ca #################'''
df.ca.isna().sum()
df.ca.value_counts().sort_index()
df.ca.describe()
df.ca.unique()
''' Categorical '''
# No null values

#Countplot
sns.countplot(x ='ca', data = df)

#Individual boxplot
sns.boxplot(x='ca', y='target', data = df)
tukey12= pairwise_tukeyhsd(df.target, df.ca, alpha=0.05)
print(tukey12)

#Merging
df.ca.value_counts().sort_index()
df['ca'].replace(4,0, inplace = True)
df['ca'].replace(2,1, inplace = True)
df['ca'].replace(3,1, inplace = True)
df.ca.value_counts().sort_index()

#Countplot after merging
sns.countplot(x ='ca', data = df)





'''############## 13- thal #####################'''
df.thal.isna().sum()
df.thal.value_counts().sort_index()
df.thal.describe()
df.thal.unique()
''' Categorical '''
# No null values

#Countplot
sns.countplot(x ='thal', data = df)

#Individual boxplot
sns.boxplot(x='thal', y='target', data = df)
tukey13= pairwise_tukeyhsd(df.target, df.thal, alpha=0.05)
print(tukey13)

#Merging
df.thal.value_counts().sort_index()
df['thal'].replace(2,0, inplace = True)
df['thal'].replace(3,1, inplace = True)
df.thal.value_counts().sort_index()

#Countplot after merging
sns.countplot(x ='thal', data = df)


''' EDA is done '''


df.to_csv("C:/Users/91876/Desktop/Kaggle Individual/3_Heart Disease UCI/Exported files/Heart_disease_UCI_EDA.csv")

