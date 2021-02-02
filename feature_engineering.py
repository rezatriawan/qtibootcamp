#!/usr/bin/env python
# coding: utf-8

# # Take Home Test

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Import dataset
employee_data = pd.read_excel('dataset_employee.xlsx')
employee_data


# In[3]:


employee_data.shape


# In[4]:


#Inspecting table
employee_data.info()


# In[5]:


employee_data.describe()


# ## Data Cleansing

# In[6]:


#Inspecting missing values
employee_data.isnull().sum()


# In[7]:


# Distribution of rating_tahun_lalu column
employee_data['rating_tahun_lalu'].plot.hist(stacked = True)


# In[8]:


# Distribution of rating_tahun_lalu column
sns.distplot(employee_data['rating_tahun_lalu'])


# In[9]:


# Plotting categorical variable
for kolom in employee_data.select_dtypes('object'):
    sns.catplot(kolom, kind='count', orient='v', data=employee_data, aspect=2);


# In[10]:


employee_data['rating_tahun_lalu'].value_counts()


# In[11]:


# Filling NA values in 'rating_tahun_lalu' column with median
employee_data['rating_tahun_lalu'] = employee_data['rating_tahun_lalu'].fillna(employee_data['rating_tahun_lalu'].median(skipna=True))


# In[12]:


# Filling NA values in 'pendidikan' column with most frequent data
employee_data['pendidikan'] = employee_data['pendidikan'].fillna("Bachelor's")


# In[13]:


# Checking missing values info
employee_data.isnull().sum()


# In[14]:


employee_data[employee_data['dipromosikan']==1]['wilayah'].value_counts()


# ## Exploratory Data Analysis

# #### How is the distribution variable target? Imbalance or not?

# In[15]:


#Checking target column
employee_data['dipromosikan'].value_counts()


# In[16]:


#Plot the target column with a bar chart
employee_data.dipromosikan.value_counts().plot(kind="bar", color=["salmon", "lightblue"]);


# From the chart above we know that our dataset is imbalanced. It means we need to handle the imbalanced data to make a good machine learning model. 

# #### How is the distribution of age? Does age having much influence to get you promoted?

# In[17]:


# Create a distribution plot
employee_promotion = employee_data[employee_data['dipromosikan'] == 1]
employee_no_promotion = employee_data[employee_data['dipromosikan'] == 0]

fig, ax = plt.subplots( figsize=(12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

g1 = sns.distplot(employee_promotion['umur'], 
             color="r")
g1 = sns.distplot(employee_no_promotion['umur'],
             color='g')
g1.set_title('Age Distribution', fontsize=15)
g1.set_xlabel('Age')
g1.set_ylabel("Count")


plt.show()


# We can infer that age distribution for the employee is mostly around at the age of 30

# #### How is the correlation between promotion and KPI? Does having a good KPI means you will get promoted?

# In[18]:


#Create a plot
pd.crosstab(employee_data['dipromosikan'], employee_data['KPI_>80%']).plot(kind="bar", 
                                    figsize=(10,6), 
                                    color=["salmon", "lightblue"]);
#Add some attributes to it
plt.title('Promoted Employee Frequency')
plt.xlabel('0 = No Promotion, 1 = Get Promoted')
plt.ylabel('Total Employee')
plt.legend(["KPI_<80%", "KPI_>80%"])
plt.xticks(rotation=0);


# There are no guarantees if you get KPI more than 80% you will get promotion easier. There are another factor like training score or last year rating that could be deciding factor. 

# #### Does having a good last year rating give you more chance being promoted?

# In[19]:


employee_data['rating_tahun_lalu'].value_counts().plot(kind='bar')


# In[20]:


#Create a plot
pd.crosstab(employee_data.rating_tahun_lalu, employee_data.dipromosikan).plot(kind="bar", 
                                    figsize=(10,6), 
                                    color=["salmon", "lightblue"]);
#Add some attributes to it
plt.title('Promoted Employee Frequency')
plt.xlabel('Last Year Rating')
plt.ylabel('Total Employee')
plt.xticks(rotation=0);


# It looks like no guarantee you will get promotion if you have a good last year rating.

# ### How is the correlation between age, training score and promotion?

# In[21]:


# Create another figure
plt.figure(figsize=(10,6))

# Create plot for person who got promoted
plt.scatter(employee_data.umur[employee_data.dipromosikan==1], 
            employee_data.rata_rata_skor_training[employee_data.dipromosikan==1], 
            c="salmon")

# Create plot for person who not to get promoted
plt.scatter(employee_data.umur[employee_data.dipromosikan==0], 
            employee_data.rata_rata_skor_training[employee_data.dipromosikan==0], 
            c='chartreuse')

# Add some helpful info
plt.title("Employee promotion in Function of Age and Training Score")
plt.xlabel("Umur")
plt.legend(["Got Promoted", "No Promotion"])
plt.ylabel("Average Training Score");


# Based on scatter plot below, we can conclude employees who got promoted are mostly the one who got higher score on the training. We can infer form this plot that training score have big proportion to decide the employee got promoted or not.

# In[22]:


# Create another figure
plt.figure(figsize=(10,6))

# Create plot for person who get promoted 
plt.scatter(employee_data.rating_tahun_lalu[employee_data.dipromosikan==1], 
            employee_data.rata_rata_skor_training[employee_data.dipromosikan==1], 
            c="salmon") # axis always come as (x, y)

# Create plot for person who not to get promoted
plt.scatter(employee_data.rating_tahun_lalu[employee_data.dipromosikan==0], 
            employee_data.rata_rata_skor_training[employee_data.dipromosikan==0], 
            c='chartreuse')

# Add some helpful info
plt.title("Employee promotion in Function of Last Year Rating and Training Score")
plt.xlabel("Last Year Rating")
plt.legend(["Got Promoted", "No Promotion"])
plt.ylabel('Average Training Score');


# Last year rating doesn't have enough proportion to decide employee who get promoted. Instead, it looks like the training score again that become the deciding factor.

# In[23]:


# Create another figure
plt.figure(figsize=(10,6))

# Create plot for person who get promoted 
plt.scatter(employee_data.jumlah_training[employee_data.dipromosikan==1], 
            employee_data.rata_rata_skor_training[employee_data.dipromosikan==1], 
            c="salmon") # axis always come as (x, y)

# Create plot for person who not to get promoted
plt.scatter(employee_data.jumlah_training[employee_data.dipromosikan==0], 
            employee_data.rata_rata_skor_training[employee_data.dipromosikan==0], 
            c='chartreuse')

# Add some helpful info
plt.title("Employee promotion in Function of Last Year Rating and Training Score")
plt.xlabel("Total Training")
plt.legend(["Got Promoted", "No Promotion"])
plt.ylabel('Average Training Score');


# In[24]:


#Create a plot
pd.crosstab(employee_data.pendidikan, employee_data.dipromosikan).plot(kind="bar", 
                                    figsize=(10,6), 
                                    color=["salmon", "lightblue"]);
#Add some attributes to it
plt.title('Promoted Employee Frequency')
plt.xlabel('Education')
plt.ylabel('Total Employee')
plt.xticks(rotation=0);


# In[25]:


#Create a plot
pd.crosstab(employee_data.penghargaan, employee_data.dipromosikan).plot(kind="bar", 
                                    figsize=(10,6), 
                                    color=["salmon", "lightblue"]);
#Add some attributes to it
plt.title('Promoted Employee Frequency')
plt.xlabel('Award')
plt.ylabel('Total Employee')
plt.xticks(rotation=0);


# In[26]:


#Create a plot
pd.crosstab(employee_data.rekrutmen, employee_data.dipromosikan).plot(kind="bar", 
                                    figsize=(10,6), 
                                    color=["salmon", "lightblue"]);
#Add some attributes to it
plt.title('Promoted Employee Frequency')
plt.xlabel('Recruitment')
plt.ylabel('Total Employee')
plt.xticks(rotation=0);


# In[27]:


employee_data.info()


# ### Feature Engineering

# In[28]:


# Back up dataset
employee_data_2 = employee_data


# In[29]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
employee_data['departemen'] = lbl.fit_transform(employee_data['departemen'])
employee_data['wilayah'] = lbl.fit_transform(employee_data['wilayah'])
employee_data['pendidikan'] = lbl.fit_transform(employee_data['pendidikan'])
employee_data['jenis_kelamin'] = lbl.fit_transform(employee_data['jenis_kelamin'])
employee_data['rekrutmen'] = lbl.fit_transform(employee_data['rekrutmen'])


# In[31]:


# Drop id column
employee_data = employee_data.drop(columns=['id_karyawan'])


# In[32]:


# Melakukan feature scalling
from sklearn import preprocessing

# Get column names first
names = employee_data.columns

# Create the Scaler object
scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object
scaled_data = scaler.fit_transform(employee_data)
scaled_data = pd.DataFrame(scaled_data, columns=names)


# In[33]:


#Feature Correlation
corr_matrix = scaled_data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            linewidths=0.5, 
            fmt= ".2f", 
            cmap="YlGnBu");


# In[ ]:




