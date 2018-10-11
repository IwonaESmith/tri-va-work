# By Iwona Smith. Trivago task: Data Science challenge


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
import math
import seaborn as sns


# Importing the dataset for traing

df_action_train = pd.read_csv('cs_actions_train.csv', sep='\t')
df_booking_train = pd.read_csv('cs_bookings_train.csv', sep='\t')

df_booking_train.shape
df_action_train.shape


df_train = pd.concat([df_booking_train, df_action_train.loc[:,'action_id':'step']], axis=1, join='inner')


df_train.describe(include = "all")
df_train.isnull().sum()




#### exploring the data 


# heat map

# Code from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
sns.set(style="white")

# Compute the correlation matrix
corr = df_train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(15,8))
sns.countplot(x='referer_code',hue='traffic_type', data=df_train, palette="husl", ax=axis1)

sns.swarmplot(x='agent_id', y="step", data=df_train)

sns.stripplot(x= df_train["referer_code"],  y=df_train["reference"], hue = df_train['has_booking'])
sns.despine()

####### date features

df_train['ymd'] = pd.to_datetime(df_train['ymd'], format ="%Y%m%d")
df_train['day'] = df_train['ymd'].dt.day.astype(int)
df_train["weekday"] = df_train['ymd'].dt.weekday.astype(int)



##### user_id
user_counts = df_train['user_id'].value_counts().to_dict()
df_train['user_counts'] = df_train['user_id'].map(user_counts)
# returning_user
df_train['user_return']=df_train['user_counts'].apply(lambda x: 1 if x==1 else 0)
df_train['user_return'].value_counts()
df_train['user_counts_log'] = np.log(df_train['user_counts'])




#### BOOKING


### referer_code
referer_code_map =  {0:10, 1:11, 10:6, 11:7, 15:8, 17:1, 19:0, 21:4, 23:2, 24:5, 99:9}
df_train.referer_code = df_train.referer_code.map(referer_code_map)
### traffic 
traffic_map=  {1:1, 2:2, 3:3, 4:4, 6:5, 7:6, 10:7}
df_train.traffic_type = df_train.traffic_type.map(traffic_map)

### agent_id

frequ_agent =  df_train['agent_id'].value_counts('2').to_dict()
df_train['agent_1'] = df_train['agent_id'].map(frequ_agent)

#####

df_train['booking_var1'] = df_train['traffic_type'].multiply(df_train['referer_code'], axis="index")

df_train['booking_var2'] = df_train['agent_1'].multiply(df_train['referer_code'])







##### ACTIONS
##### action_id

frequ_action =  df_train['action_id'].value_counts().to_dict()
df_train['action1'] = df_train['action_id'].map(frequ_action)
df_train['action_log'] = np.log(df_train['action_id'])
df_train['action_1_log'] = np.log(df_train['action1'])



#### reference
counts = df_train['reference'].value_counts().head(20).to_dict()
print(counts)
ref_dict = {0: 20, 1: 19, 2: 18, 1110: 17, 4: 16, 38715: 15, 6: 14, 3: 13, 60: 12, 65536: 11, 
     10: 10, 212: 9, 262144: 8, 8: 7, 5: 6, 1048576: 5, 1210: 4, 40: 3, 38961: 2, 1312:1 }

df_train['reference_sig']= df_train['reference'].map(ref_dict).fillna(0)

df_train['ref_matrix'] =  df_train['reference_sig'].multiply( df_train['referer_code'])

#### action + reference
df_train['action_mix'] =  df_train['action_log'].multiply( df_train['reference_sig'])



### step

frequ_step =  df_train['step'].value_counts('2').to_dict()
df_train['step_num'] = df_train['step'].map(frequ_step)


bins = [0, 6, 13, 58,91 ,222,  np.inf]
group = [5,4,3, 2, 1, 0]
df_train['step_value'] = pd.cut(df_train.step, bins, labels = group)


df_train['multiple'] = df_train['step_num'].multiply( df_train['reference_sig'] )




columns = list(df_train.columns.values)
print(columns)


[(0.30080000000000001, 'step_num'),
 (0.27100000000000002, 'day'),
 (0.073899999999999993, 'agent_1'),
 (0.066600000000000006, 'multiple'),
 (0.044499999999999998, 'action_log'),
 (0.042900000000000001, 'action_1_log'),
 (0.042599999999999999, 'action1'),
 (0.036900000000000002, 'ref_matrix'),
 (0.035099999999999999, 'user_counts'),
 (0.0349, 'user_counts_log'),
 (0.032199999999999999, 'step_value'),
 (0.014500000000000001, 'reference_sig'),
 (0.0040000000000000001, 'user_return')]

                     'traffic_type'
                     'referer_code'
                     'user_return'
                     'weekday'
                     'user_return'
                     'user_counts_log'
                     'action_log'
                     
                     
                     
                     'step_num','agent_1', 'multiple', 'action_log', 'action_1_log'




df_train = df_train.drop(['ymd','user_id', 'session_id','is_app', 'action_id', 'reference', 'step', 'day',], axis=1)

df_train = df_train[['weekday', 'user_counts', 'user_return', 'user_counts_log', 'agent_1', 
 'booking_var1', 'booking_var2', 'action1', 'action_log',
 'action_1_log', 'reference_sig', 'ref_matrix', 'action_mix', 'step_num',
 'step_value', 'multiple', 'has_booking']]

['referer_code', 'agent_id', 'traffic_type', 
 
 'weekday', 'user_counts', 'user_return', 'user_counts_log', 'agent_1', 
 'booking_var1', 'booking_var2', 'action1', 'action_log',
 'action_1_log', 'reference_sig', 'ref_matrix', 'action_mix', 'step_num',
 'step_value', 'multiple']


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Fitting a model

X = df_train.loc[:, [ 'step_num','agent_1', 'multiple', 'action_log', 'action_1_log']]
y = df_train['has_booking']


# train set split to train and validate leaving the test set for prediction  
from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 0)






# Logistic Regression 
from sklearn.linear_model import LogisticRegression
logregClassifier = LogisticRegression(random_state = 0, solver = 'lbfgs')
logregClassifier.fit(X_train, y_train)
y_pred = logregClassifier.predict(X_val)
acc_logregClassifier = round(accuracy_score(y_val, y_pred)*100 , 2)
rmse_logistic = np.sqrt(metrics.mean_squared_error(y_val, y_pred))



# Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_val)
acc_gaussian =  round(accuracy_score(y_val, y_pred) * 100, 2)
rmse_gaussian = np.sqrt(metrics.mean_squared_error(y_val, y_pred))



# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_val)
random_forest.score(X_train, y_train)
acc_random_forest = round(accuracy_score(y_val, y_pred) * 100, 2)
rmse_random_forest = np.sqrt(metrics.mean_squared_error(y_val, y_pred))
names = X.columns.values
sorted(zip(map(lambda x:round(x,4),random_forest.feature_importances_),names ),reverse=True)








####

# SVM
from sklearn.svm import SVC
svcClassifier = SVC(kernel = 'rbf', random_state = 0)
svcClassifier.fit(X_train, y_train)
y_pred = svcClassifier.predict(X_val)
acc_svcClassifier = round(accuracy_score(y_val, y_pred)*100 , 2)
rmse_svc = sqrt(mean_squared_error(y_val, y_pred))


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
y_pred = gbk.predict(X_val)
acc_gbk = round(accuracy_score(y_val, y_pred) * 100, 2)
rmse_booster = np.sqrt(metrics.mean_squared_error(y_val, y_pred))
  

# XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_val)
acc_xgb = round(accuracy_score(y_val, y_pred) * 100, 2)
rmse_xgb = np.sqrt(metrics.mean_squared_error(y_val, y_pred))



# Out models I have tested the Xgboost produce the best accuracy, 
# but the performance metric was worst than Random Forest. 
#None of the models perform amazingly, but Random Forest had the best results (76.5% with RSME of 5.6)
#predicting the results
# This is the model I will use for predicitng results in test set. 



df_test = pd.read_csv('test_set.csv')

df_test = df_test.reindex_axis(['bookings','id', 'yyear', 'week_of_year', 'advertiser_id', 'hotel_id', 'clicks',  
  'city_id', 'stars',  'distance_to_city_centre',  'total_images', 'top_pos', 'impressions',], axis =1)

corr = df_test.corr()
# year 
df_test['yyear'] = df_test['yyear'].map({2017: 0 , 2018: 1})

# stars  for 0 replace with 3* as this type of hotel is mostly booked and star.mean() = 3.58
df_test['stars'] = df_test['stars'].map({0: 3, 1:1, 2:2, 3:3, 4:4, 5:5})

# number of top_pos v. impressions in relation to bookings made
df_test['top/impresion'] = (df_test['top_pos']/df_test['impressions'])
df_test = df_test.drop(['top_pos', 'impressions'], axis = 1)
df_test['top/impresion'] = df_test['top/impresion'].fillna(df_test['top/impresion'].mean())

# seting the test set
X_test = df_test.loc[:, "yyear":"top/impresion"]
# predicting values with random forest
y_predforest = random_forest.predict(X_test)

# saving results
result['y_pred'] = y_predforest
result = result.drop(["pred_bookings"], axis= 1)

result.to_csv('resultIWONASMITH.csv')





