#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTING LIBARARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # GET THE DATA

# In[2]:


# IMPORT DATASET
from sklearn.datasets import load_boston
boston = load_boston()


# In[3]:


#  Detailed Information on dataset
print(boston.DESCR)


# In[4]:


# Column Name
boston.feature_names


# In[6]:


# Data to Pandas Data Frame
df = pd.DataFrame(boston.data, columns=boston.feature_names)


# In[7]:


df.head()


# In[8]:


# Add MEDV column to dataset
df['MEDV'] = boston.target


# In[9]:


# Dataset Info
df.info()


# In[10]:


#Dataset Info
df.describe()


# # Visualization

# In[11]:


# Creating a grid of Axes such that each numeric variable in data will by shared in the y-axis across a single row and in the x-axis across a single column.
sns.pairplot(df)


# In[13]:


# Creating a subplot of 2 columns and 7 rows
rows = 7
cols = 2


fig, ax = plt.subplots(nrows= rows, ncols= cols, figsize = (16,16))
col = df.columns
index = 0

# ploting distrubution plot of dataset columns
for i in range(rows):
    for j in range(cols):
        sns.distplot(df[col[index]], ax = ax[i][j])
        index = index + 1
plt.tight_layout()


# In[15]:


# Exploring Correlation between variables
fig, ax = plt.subplots(figsize = (16, 9))
sns.heatmap(df.corr(), annot = True, annot_kws={'size': 12})


# In[18]:


# Filter Correlation
def getCorrelatedFeature(corrdata, threshold):
    feature = []
    value = []
    for i, index in enumerate(corrdata.index):
        if abs(corrdata[index])> threshold:
            feature.append(index)
            value.append(corrdata[index])

    df = pd.DataFrame(data = value, index = feature, columns=['Corr Value'])
    return df


# In[19]:


# setting threshold and MEDV col as index to other columns
threshold = 0.4
corr_value = getCorrelatedFeature(df.corr()['MEDV'], threshold)


# In[21]:


# Output array
corr_value.index.values


# In[22]:


# Array to database
correlated_data = df[corr_value.index]
correlated_data.head()


# # ML Linear Regresion

# In[23]:


# Fitting X and y
X = correlated_data.drop(labels=['MEDV'], axis = 1)
y = correlated_data['MEDV']


# In[26]:


# importing ML modules, setting train and test dataset to 67% and 33% with random state 1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[27]:


# Importing Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[28]:


#Fiting data to model
lm.fit(X_train,y_train)


# In[29]:


#Setting on unseen answers to test model
predictions = lm.predict(X_test)


# In[31]:


# Plotting Predictions Scatter Plot
plt.scatter(y_test,predictions)


# In[32]:


# Plotting Prediction Distrubution
sns.distplot((y_test-predictions),bins=50)


# In[33]:


#y intercept of ML model
lm.intercept_


# In[34]:


# Coefficients of a linear regression function
lm.coef_


# In[35]:


# Defining Linear Regression Function
def lin_func(values, coefficients=lm.coef_, y_axis=lm.intercept_):
    return np.dot(values, coefficients) + y_axis


# In[39]:


# Function to test random data from dataset aganist prediction and measure difference
from random import randint
for i in range(5):
    index = randint(0,len(df)-1)
    sample = df.iloc[index][corr_value.index.values].drop('MEDV')
    print(
                'PREDICTION: ', round(lin_func(sample),2),
                ' // REAL: ',df.iloc[index]['MEDV'],
                ' // DIFFERENCE: ', round(round(lin_func(sample),2) - df.iloc[index]['MEDV'],2))


# In[ ]:




