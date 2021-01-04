#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Car price prediction
#


# In[80]:


## Loaad pre libraries
# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[90]:


## Load dataset
#
dataset_train =  pd.read_csv('train-data.csv')
dataset_train.head()

pin1 = len(dataset_train)
print("Training set last index : {}".format(pin1))

dataset_test =  pd.read_csv('test-data.csv')
dataset_train = dataset_train.append(dataset_test)
dataset_train.head()


# In[91]:


## Basic information of dataset
# 

# Total records
print("Total records : {}".format(len(dataset_train)))

# Total columns
print("Total columns : {}".format(len(dataset_train.columns)))

# Column names
print("Column names : {}".format(dataset_train.columns))


# In[92]:


## General information of dataset
#

dataset_train.info()


# In[93]:


## Statistical information 
#

dataset_train.describe()


# In[94]:


## Check missing values
#

dataset_train.isna().sum()


# In[95]:


## Handle mising values
#
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan,
                        strategy = 'most_frequent')
dataset_train[["Mileage", "Engine", "Power", "Seats"]] = imputer.fit_transform(dataset_train[["Mileage", "Engine", "Power", "Seats"]])

dataset_train.isna().sum()


# In[96]:


## Remove column
# New price column
dataset_train = dataset_train.drop(['New_Price'], axis = 1)

# Drop first column
dataset_train.drop(dataset_train.columns[[0]], 
                   axis = 1, 
                   inplace = True)

dataset_train.head()


# In[97]:


## Handle categorical column
# Name, Location, Fuel_Type, Transmission, Owner_Type, Mileage, Engine, Power
# Name
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

name_encoded_labels = list(dataset_train['Name'])
dataset_train.Name = encoder.fit_transform(dataset_train.Name)
name_encoded_labels_dict = dict(zip(list(dataset_train.Name), name_encoded_labels))

location_encoded_labels = list(dataset_train['Location'])
dataset_train.Location = encoder.fit_transform(dataset_train.Location)
location_encoded_labels_dict = dict(zip(list(dataset_train.Location), location_encoded_labels))

fueltype_encoded_labels = list(dataset_train['Fuel_Type'])
dataset_train.Fuel_Type = encoder.fit_transform(dataset_train.Fuel_Type)
fueltype_encoded_labels_dict = dict(zip(list(dataset_train.Fuel_Type), fueltype_encoded_labels))

transmissions_encoded_labels = list(dataset_train['Transmission'])
dataset_train.Transmission = encoder.fit_transform(dataset_train.Transmission)
transmissions_encoded_labels_dict = dict(zip(list(dataset_train.Transmission), transmissions_encoded_labels))

owner_encoded_labels = list(dataset_train['Owner_Type'])
dataset_train.Owner_Type = encoder.fit_transform(dataset_train.Owner_Type)
owner_encoded_labels_dict = dict(zip(list(dataset_train.Owner_Type), owner_encoded_labels))

mileage_encoded_labels = list(dataset_train['Mileage'])
dataset_train.Mileage = encoder.fit_transform(dataset_train.Mileage)
mileage_encoded_labels_dict = dict(zip(list(dataset_train.Mileage), mileage_encoded_labels))

engine_encoded_labels = list(dataset_train['Engine'])
dataset_train.Engine = encoder.fit_transform(dataset_train.Engine)
engine_encoded_labels_dict = dict(zip(list(dataset_train.Engine), engine_encoded_labels))

power_encoded_labels = list(dataset_train['Power'])
dataset_train.Power = encoder.fit_transform(dataset_train.Power)
power_encoded_labels_dict = dict(zip(list(dataset_train.Power), power_encoded_labels))

dataset_train.head()


# In[98]:


## Prepare testing dataset
#

dataset_test = dataset_train[6019:]
dataset_test.head()


# In[100]:


## Dependent and Independant variables
# 
features =['Name','Location','Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats']
target = 'Price'


# In[112]:


dataset_train.replace([np.inf, -np.inf], np.nan, inplace=True) 
dataset_train.dropna(inplace=True) 
  
dataset_test.replace([np.inf, -np.inf], np.nan, inplace=True) 
dataset_test.dropna(inplace=True) 


# In[113]:


## Split dataset
#
from sklearn.model_selection import train_test_split
Y = dataset_train['Price']
X = dataset_train.drop(columns=['Price'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=7)

print('X train - ', X_train.shape)
print('Y train -  ', Y_train.shape)
print('X test - ', X_test.shape)
print('Y test - ', Y_test.shape)


# In[ ]:





# In[109]:


X_train = X_train.reset_index()
Y_train = Y_train.reset_index()


# In[114]:


## Model 
# Random Forest
# Find best parameters for random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

estimator = RandomForestRegressor(random_state = 42,criterion='mse')
para_grids = {
            "n_estimators" : [10,50,100],
            "max_features" : ["auto", "log2", "sqrt"],
            'max_depth' : [4,5,6,7,8,9,15],
            "bootstrap"    : [True, False]
        }

Grid = GridSearchCV(estimator, para_grids,cv= 5)
Grid.fit(X_train, Y_train)
best_param = Grid.best_estimator_
print(best_param)


# In[116]:


# model
model = RandomForestRegressor(random_state = 42,criterion='mse',
                                bootstrap = False, 
                                max_depth = 15,
                                max_features = 'log2')

model.fit(X_train, Y_train)


# In[117]:


## Predictions
#
Y_pred = model.predict(X_test)


# In[119]:


## About accuracy
#

import sklearn.metrics as metrics

mae = metrics.mean_absolute_error(Y_test, Y_pred)
mse = metrics.mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse) # or mse**(0.5)  
r2 = metrics.r2_score(Y_test, Y_pred)

print("Results of sklearn.metrics:")
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)
 


# In[120]:


## Back to python
#
get_ipython().system('jupyter nbconvert --to script prise_pred.ipynb')


# In[ ]:





# In[ ]:




