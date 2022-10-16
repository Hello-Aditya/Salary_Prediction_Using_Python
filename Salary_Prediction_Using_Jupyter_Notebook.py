#!/usr/bin/env python
# coding: utf-8

# problem statement - Given the dataset which captures gross salary of the employees and by training with such data we want to predict salary of the employees in the test data.

# Outlines
# 1. import libraries
# 2. import dataset
# 3. Data cleaning and data preparation
# 4. exploratory data analysis
# 5. feature engineering 
# 6. Train test data
# 7. model building 
# 8. model evaluation

# In[167]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split


# In[168]:


salary = pd.read_csv("D:\\Chandigarh University File\\dev town lecture\\files_dev\\train_salary.csv")
salary.head()


# 1. 18981 rows
# 2. total 7 columns
# 3. all7 are object

# In[169]:


df = salary.copy()


# In[170]:


df.columns


# In[171]:


# method_1 - df.columns =['Name', 'JobTitle','AgencyId','Agency','HireDate','AnnualSalary','GrossPay']
#method_2
df.columns =df.columns.str.strip()


# In[172]:


df.columns


# In[173]:


#method_1 - df['AnnualSalary'].str.replace('$','')
#method_2
#for removing $ sign from each row of 'AnnualSalary' column
df['AnnualSalary'] = df['AnnualSalary'].apply(lambda x: float(str(x)[1:]))


# In[174]:


df


# In[175]:


df.isnull().sum()


# In[176]:


df['HireDate'] =df['HireDate'].fillna(method='ffill')


# In[177]:


df.drop('GrossPay',axis=1,inplace=True)


# In[178]:


df.isnull().sum()


# ## which agency id has more hirings

# In[179]:


df.AgencyID.value_counts()


# In[180]:


df.Agency.value_counts()


# In[181]:


df.JobTitle.value_counts()


# In[182]:


sns.distplot(df.AnnualSalary)
plt.title('Annual Salary Distribution Plot',fontsize=15)


# In[183]:


df.AnnualSalary.plot.box()


# In[184]:


df.shape


# In[185]:


df


# In[186]:


len(df[df['AnnualSalary']>150000])


# In[187]:


df = df[df['AnnualSalary']<140000]


# In[188]:


df.AnnualSalary.plot.box()


# In[189]:


df['HireDay'] = df['HireDate'].apply(lambda x: int(str(x[3:5])))


# In[190]:


df


# In[191]:


df['Hiremonth'] = df['HireDate'].apply(lambda x : int(str(x[0:2])))


# In[192]:


df


# In[193]:


df['Hireyear'] = df['HireDate'].apply(lambda x: int(str(x[6:])))


# In[194]:


df


# In[195]:


## EDA 
## Top 10 jobs based on hirings
df.groupby(['JobTitle'])['Name'].count().sort_values(ascending=False).head(10).plot.bar()


# In[196]:


## Top 10 jobs that fetch highest salary
df.groupby(['JobTitle'])['AnnualSalary'].mean().sort_values(ascending=False).head(10).plot.bar()


# In[197]:


### top agencies with higher number of employees
df.groupby(['Agency'])['Name'].count().sort_values(ascending=False).head(10).plot.bar()
plt.figure(figsize=(20,20))


# In[198]:


## top agency ids with higher number of employees
df.groupby(['AgencyID'])['Name'].count().sort_values(ascending=False).head(10).plot.bar()


# In[199]:


## the jobs whose average pay is more than total average salary of entire data
mean_job = df.AnnualSalary.mean()

good_paying_jobs = df.groupby(['JobTitle'])['AnnualSalary'].mean().reset_index()


# In[200]:


mean_job


# In[201]:


good_paying_jobs


# In[202]:


good_paying_jobs[good_paying_jobs['AnnualSalary']> mean_job]['JobTitle'].count()


# In[203]:


len(df.JobTitle.unique())


# In[204]:


df.columns


# In[205]:


sns.pairplot(df)


# In[206]:


#plot a heatmap
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True)


# # Feature Engineering

# In[207]:


mean_job = df.groupby(['JobTitle'])['AnnualSalary'].mean()
df['JobTitle'] = df['JobTitle'].map(mean_job)


# In[208]:


df


# In[209]:


mean_agency =df.groupby(['Agency'])['AnnualSalary'].mean()
df['Agency'] = df['Agency'].map(mean_agency)


# In[210]:


mean_agencyId = df.groupby(['AgencyID'])['AnnualSalary'].mean()
df['AgencyID'] =df.AgencyID.map(mean_agencyId)


# In[211]:


df


# In[212]:


df = df.drop(['HireDate','Name'],axis=1)


# In[213]:


df


# In[214]:


###
train,test =train_test_split(df,train_size=0.7,random_state =42)


# In[215]:


train.shape


# In[216]:


test.shape


# In[217]:


y_train = train.pop('AnnualSalary')
x_train = train


# In[218]:


Y_test =test.pop('AnnualSalary')
x_test = test


# In[219]:


x_test


# In[220]:


x_train


# In[221]:


Y_test


# In[222]:


y_train


# In[223]:


## scale 
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_train[x_train.columns] = scaler.fit_transform(x_train[x_train.columns])


# In[224]:


x_train.describe()


# In[225]:


x_test[x_test.columns]= scaler.fit_transform(x_test[x_test.columns])


# In[226]:


lr = LinearRegression()
salary_reg =lr.fit(x_train, y_train)


# In[227]:


salary_reg.score(x_train, y_train)


# In[228]:


y_pred =salary_reg.predict(x_test)


# In[229]:


Y_test


# In[230]:


y_pred


# In[231]:


salary_reg.coef_


# In[232]:


salary_reg.intercept_


# In[233]:


model =str(salary_reg.intercept_)
for i in range(len(salary_reg.coef_)):
    model = model + '+' +(str(salary_reg.coef_[i])) + '*' +(str(x_train.columns[i]))


model


# In[234]:


from sklearn.metrics import r2_score
r2_score(Y_test,y_pred)

