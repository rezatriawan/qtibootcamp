#!/usr/bin/env python
# coding: utf-8

# # Tugas Akhir

# In[60]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import pickle

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


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

# #### How about correlation between last year rating, average training score and promotion?

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

# #### Is the amount of total training could be the determining factor?

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


# #### Total of employee got promoted based on education level

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


# #### Total Promoted Employee Based on Awards

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


# #### Total Promoted Employee Based on Recruitment

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


# Feature scaling
employee_data['jumlah_training'] = scale(employee_data['jumlah_training'])
employee_data['umur'] = scale(employee_data['umur'])
employee_data['rating_tahun_lalu'] = scale(employee_data['rating_tahun_lalu'])
employee_data['masa_kerja'] = scale(employee_data['masa_kerja'])
employee_data['KPI_>80%'] = scale(employee_data['KPI_>80%'])
employee_data['penghargaan'] = scale(employee_data['penghargaan'])
employee_data['rata_rata_skor_training'] = scale(employee_data['rata_rata_skor_training'])


# In[31]:


# Encoding categorical variable
departemen = pd.get_dummies(employee_data['departemen'],drop_first=True)
wilayah = pd.get_dummies(employee_data['wilayah'],drop_first=True)
pendidikan = pd.get_dummies(employee_data['pendidikan'],drop_first=True)
jenis_kelamin = pd.get_dummies(employee_data['jenis_kelamin'],drop_first=True)
rekrutmen = pd.get_dummies(employee_data['rekrutmen'],drop_first=True)


# In[32]:


employee_data.drop(['departemen','wilayah','pendidikan','jenis_kelamin','rekrutmen'], axis=1, inplace=True)


# In[35]:


employee_data = pd.concat([employee_data,departemen,wilayah,pendidikan,jenis_kelamin,rekrutmen],axis=1)


# In[38]:


employee_data.corr()


# ## Machine Learning Model

# In[39]:


X = employee_data.drop('dipromosikan', axis=1)

#Target variable
y = employee_data.dipromosikan


# In[40]:


# Random seed for reproducibility
np.random.seed(42)

#Split into train & test set
X_train, X_val, y_train, y_val = train_test_split(X, 
                                                    y,
                                                    test_size = 0.2)


# ### K-fold method

# In[41]:


folding = StratifiedKFold(n_splits=10)

folding.get_n_splits(X, y)


# In[42]:


for training_index, test_index in folding.split(X, y):
    print("Index X: ", training_index, " dan Index y: ", test_index)


# ### Leave-One-Out

# In[43]:


leave_one = LeaveOneOut()

leave_one.get_n_splits(X)


# In[44]:


for training_index, test_index in leave_one.split(X):
    print("Fold ke")
    print("Ukuran X: ", training_index.shape, " dan ukuran y: ", test_index.shape)
    print("")


# ## Hyperparameter Tuning

# #### Initiation

# In[45]:


# menentukan jumlah estimator
n_estimators = [10, 20]
# Menentukan jumlah feature yang digunakan dalam tiap split
max_features = ['auto', 'sqrt']
# Menentukan tingkat kedalaman pohon keputusan
max_depth = [5, 6, 7, 8]
# Menentukan jumlah minimum sampel untuk melakukan split pada node
min_samples_split = [2, 5, 10]
# Menentukan jumlah sampel minimum untuk melakukan split pada leaf
min_samples_leaf = [1, 2, 4]
# Metode penentuan sampling
bootstrap = [True, False]

# inisiasi grid
inisiasi_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

inisiasi_grid


# In[46]:


# LogisticRegression hyperparameters
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# RandomForestClassifier hyperparameters
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           'max_features': ['auto', 'sqrt'],
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2),
           'bootstrap': [True, False]}

# DecisionTreeClassifier hyperparameters
dectree_grid = {'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [None, 3, 5, 10],
                'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
                'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
                }

# Different XGBoostClassifier hyperparameters
xgb_grid = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
            "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
            "min_child_weight" : [ 1, 3, 5, 7 ],
            "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
            "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }


# ### RandomSearchCV

# In[47]:


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()


# In[48]:


# RandomizedSearchCV
model_random_cv = RandomizedSearchCV(estimator=model_rf, 
                                     param_distributions=inisiasi_grid,
                                     n_iter=2,
                                     cv=folding,
                                    random_state=1000,
                                    n_jobs=3)

# Model fitting
model_random_cv.fit(X_train,y_train)


# In[49]:


# Checking best estimator
model_random_cv.best_estimator_


# In[50]:


pred_result = model_random_cv.predict(X_val)

pred_result


# In[51]:


from sklearn.metrics import accuracy_score
accuracy_score(y_val, pred_result)


# In[52]:


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=folding,
                                n_iter=10,
                                random_state = 1000,
                                n_jobs = 3,
                                verbose=True)

# Fit random hyperparameter search model
rs_log_reg.fit(X_train, y_train);


# In[53]:


print("Best Hyper Parameters:\n",rs_log_reg.best_params_)
# Prediction
prediction=rs_log_reg.predict(X_val)

# Evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,y_val))

# Evaluation(Confusion Metrics)
print("Confusion Metrics:\n",metrics.confusion_matrix(prediction,y_val))


# In[54]:


# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=folding,
                           n_iter=10,
                           random_state = 1000,
                           n_jobs = 4,
                           verbose=True)

# Fit random hyperparameter search model
rs_rf.fit(X_train, y_train);


# In[55]:


print("Best Hyper Parameters:\n",rs_rf.best_params_)
# Prediction
prediction=rs_rf.predict(X_val)

# Evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,y_val))

# Evaluation(Confusion Metrics)
print("Confusion Metrics:\n",metrics.confusion_matrix(prediction,y_val))


# In[56]:


# Setup random hyperparameter search for DecisionTreeClassifier
rs_dectree = RandomizedSearchCV(DecisionTreeClassifier(),
                           param_distributions=dectree_grid,
                           cv=folding,
                           n_iter=10,
                           random_state = 1000,
                           n_jobs = 4,
                           verbose=True)

# Fit random hyperparameter search model
rs_dectree.fit(X_train, y_train);


# In[57]:


print("Best Hyper Parameters:\n",rs_dectree.best_params_)
# Prediction
prediction=rs_dectree.predict(X_val)

# Evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,y_val))

# Evaluation(Confusion Metrics)
print("Confusion Metrics:\n",metrics.confusion_matrix(prediction,y_val))


# In[58]:


# Setup random hyperparameter search for KNNClassifier
rs_xgb = RandomizedSearchCV(XGBClassifier(),
                           param_distributions=xgb_grid,
                           cv=folding,
                           n_iter=10,
                           random_state = 1000,
                           n_jobs = 4,
                           verbose=True)

# Fit random hyperparameter search model
rs_xgb.fit(X_train, y_train);


# In[59]:


print("Best Hyper Parameters:\n",rs_xgb.best_params_)
# Prediction
prediction=rs_xgb.predict(X_val)

# Evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,y_val))

# Evaluation(Confusion Metrics)
print("Confusion Metrics:\n",metrics.confusion_matrix(prediction,y_val))


# In[61]:


# Saving the model
filename = 'finalized_model.sav'
pickle.dump(rs_xgb, open(filename, 'wb'))

