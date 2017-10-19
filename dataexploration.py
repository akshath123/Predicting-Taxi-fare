import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import mode
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.neural_network import MLPRegressor  
from sklearn.utils import shuffle

import seaborn as sns 

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')

dataset = pd.read_csv('created_train.csv')
dataset_test = pd.read_csv('test_provide.csv')
dataset_test_1 = pd.read_csv('test_provide.csv')

# Dropping the columns which are non numerical data
dataset.drop(dataset[['pickup_datetime','dropoff_datetime', 'vendor_id', 'TID', 'store_and_fwd_flag', 'rate_code']],axis=1,inplace=True)
dataset_test.drop(dataset_test[['pickup_datetime','dropoff_datetime', 'vendor_id', 'TID', 'store_and_fwd_flag', 'rate_code']],axis=1,inplace=True)
dataset_test_1.drop(dataset_test_1[['pickup_datetime','dropoff_datetime', 'vendor_id', 'store_and_fwd_flag', 'rate_code']],axis=1,inplace=True)

# Checking for null values  
dataset.isnull().any()
dataset_test.isnull().any()

# Converting the categorical data to numerical data
dataset = pd.get_dummies(dataset,columns=['new_user','payment_type'])
dataset_test = pd.get_dummies(dataset_test,columns=['new_user','payment_type'])

#dataset = dataset.dropna(how = 'any', axis = 0)
#dataset_test = dataset_test.dropna(how = 'any', axis = 0)
#dataset_test_1 = dataset_test_1.dropna(how = 'any', axis = 0)

dataset = dataset.fillna(0)
dataset_test = dataset_test.fillna(0)
dataset_test_1 = dataset_test_1.fillna(0)

# Splitting the data 
X = dataset
X = X.drop('fare_amount', 1)
Y = dataset['fare_amount']

from sklearn.cross_validation import train_test_split
X, X_test_tdata, Y, y_test_tdata = train_test_split(X, Y, test_size = 0.33, random_state = 0)

X_test = dataset_test

# Filling all the missing values 
X['tip_amount'] = X['tip_amount'].fillna(0)
X_test['tip_amount'] = X_test['tip_amount'].fillna(0)
X_test_tdata['tip_amount'] = X_test_tdata['tip_amount'].fillna(0)
# Checking if any null value is there
X['tip_amount'].isnull().any()
X_test['tip_amount'].isnull().any()
X_test_tdata['tip_amount'].isnull().any()

# Finding the number of null values in the dataset
X['surcharge'] = X['surcharge'].fillna(0.5)
X_test['surcharge'] = X_test['surcharge'].fillna(0.5)
X_test_tdata['surcharge'] = X_test_tdata['surcharge'].fillna(0.5)
# Checking if any null value is there
X['surcharge'].isnull().any()
X_test['surcharge'].isnull().any()
X_test_tdata['surcharge'].isnull().any()

# Calculating the distane 
def haversine(lon1, lat1, lon2, lat2):
    # Convert coordinates to floats.
    lon1, lat1, lon2, lat2 = [float(lon1), float(lat1), float(lon2), float(lat2)]
    # Convert to radians from degrees.
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # Compute distance.
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    km = 6367 * c
    return km

#for lat1, long1, lat2, long2 in X['pickup_longitude'], X['pickup_latitude'], X['dropoff_latitude'], X['dropoff_longitude']:
    #print (lat1,long1)

X['distance'] = 0
X_test['distance'] = 0
X_test_tdata['distance'] = 0
dist_list = []
dist_list_test = []
dist_list_test_tdata = []
    
for index, row in X.iterrows():
    long1 = row["pickup_longitude"] 
    lat1 = row["pickup_latitude"]
    lat2 = row["dropoff_latitude"]
    long2 = row["dropoff_longitude"]
    dist_list.append(haversine(long1, lat1, long2, lat2))
    
dist_list = np.array(dist_list)

for index, row in X_test.iterrows():
    long1 = row["pickup_longitude"] 
    lat1 = row["pickup_latitude"]
    lat2 = row["dropoff_latitude"]
    long2 = row["dropoff_longitude"]
    dist_list_test.append(haversine(long1, lat1, long2, lat2))
dist_list_test = np.array(dist_list_test)

for index, row in X_test_tdata.iterrows():
    long1 = row["pickup_longitude"] 
    lat1 = row["pickup_latitude"]
    lat2 = row["dropoff_latitude"]
    long2 = row["dropoff_longitude"]
    dist_list_test_tdata.append(haversine(long1, lat1, long2, lat2))
    
dist_list_test_tdata = np.array(dist_list_test_tdata)

X_test
X["distance"] = dist_list
X_test["distance"] = dist_list_test
X_test_tdata["distance"] = dist_list_test_tdata

# Droping the pickup_longitude, pickup_latitude, dropoff_latitude and dropoff_longitude
X = X.drop('pickup_longitude', 1)
X = X.drop('pickup_latitude', 1)
X = X.drop('dropoff_latitude', 1)
X = X.drop('dropoff_longitude', 1)

X_test_tdata = X_test_tdata.drop('pickup_longitude', 1)
X_test_tdata = X_test_tdata.drop('pickup_latitude', 1)
X_test_tdata = X_test_tdata.drop('dropoff_latitude', 1)
X_test_tdata = X_test_tdata.drop('dropoff_longitude', 1)

X_test = X_test.drop('pickup_longitude', 1)
X_test = X_test.drop('pickup_latitude', 1)
X_test = X_test.drop('dropoff_latitude', 1)
X_test = X_test.drop('dropoff_longitude', 1)

regr = RandomForestRegressor(max_depth=10,random_state=0 )
regr.fit(X,Y)
Y_pred_test = regr.predict(X_test_tdata)
Y_pred = regr.predict(X_test)

Y1 = pd.DataFrame()
Y1["fare_amount"] = Y_pred
df = pd.DataFrame() 
df["TID"] = dataset_test_1["TID"]
df["fare_amount"] = Y1
df = pd.concat([dataset_test_1["TID"], Y1], axis = 1, join = "inner")

Y_preds = pd.DataFrame()
mlp_reg = MLPRegressor(hidden_layer_sizes = (13, 13), max_iter = 1000 )
regressor.fit(X, Y)
Y_pred_1 = regressor.predict(X_test)
Y_preds["fare_amount"] = Y_pred_1
df = pd.concat([dataset_test_1["TID"], Y_preds], axis = 1, join = "inner")

df.to_csv("output1.csv")
print(mean_squared_error(Y_pred, y_test_tdata))

