
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Lasso, LassoCV

from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from copy import deepcopy

#sns.set()
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data_df = pd.read_csv('cleaned_data.csv')
print(data_df.shape)
data_df.head()


# In[3]:


data_df_clean = data_df.dropna()
data_df_clean.reset_index(drop=True, inplace=True)


# In[4]:


data_df_clean.info()


# In[5]:


data_df_clean.Country.unique()


# In[6]:


X = data_df_clean.drop(['Country', 'Year', '1100', 'Top_1', '6000'], axis=1)
X.head()


# In[7]:


X.tail()


# In[8]:


y = data_df_clean.Top_1


# In[9]:


y = y*100


# ### Modeling

# In[10]:


years = data_df_clean.Year
train_size = int(len(years) * .7)
test_size = len(years) - train_size
print(len(years), train_size, test_size)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



data_df_trimmed = pd.read_csv('trimmed_cleaned_data.csv')
print(data_df_trimmed.shape)

data_df_trimmed.dropna(subset=['Top_1'], inplace=True)


trim_clean_df = data_df_trimmed.dropna()
print(trim_clean_df.shape)
trim_clean_df.head()


# In[19]:


X_trim = trim_clean_df.drop(['Country', 'Year', '1100', 'Top_1', '6000'], axis=1)
X_trim.head()


X_train, X_test, y_train, y_test = train_test_split(X_trim, y, test_size=0.3, random_state=42)





# ### Try XGBoost



data_df_trimmed.dropna(thresh=5, inplace=True)

data_df_trimmed.reset_index(drop=True, inplace=True)


X = data_df_trimmed.drop(['Country', 'Year', '1100', 'Top_1', 
                               'GDP_growth', '6000'], axis=1)


y = data_df_trimmed.Top_1 * 100



#Split data into 3: 60% train, 20% validation, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=2019)



eval_set=[(X_train,y_train),(X_val,y_val)]

gbm = xgb.XGBRegressor( 
                       n_estimators=30000, #arbitrary large number
                       max_depth=3,
                       objective="reg:linear",
                       learning_rate=.1, 
                       subsample=1,
                       min_child_weight=1,
                       colsample_bytree=.8
                      )


fit_model = gbm.fit( 
                    X_train, y_train, 
                    eval_set=eval_set,
                    #eval_metric='rmse',
                    early_stopping_rounds=50,
                    verbose=True #gives output log as below
                   )


test_pred = gbm.predict(X_test)



print(mean_squared_error(y_test, test_pred))



print(r2_score(y_test, test_pred))


with open('gbm_initial.pkl', 'wb') as f:
    pickle.dump(gbm, f)

