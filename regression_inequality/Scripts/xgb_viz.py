
# coding: utf-8

# In[2]:


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
from xgboost import plot_importance

from copy import deepcopy

#sns.set()
#plt.style.use('fivethirtyeight')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


with open('gbm_initial.pkl', 'rb') as f:
    gbm = pickle.load(f)


# In[12]:


#xgb.plot_importance(gbm, max_num_features=6)
#xgb.plot_importance(gbm, importance_type='gain', max_num_features=6)


# In[6]:


gbm.feature_importances_


# In[13]:


data_df = pd.read_csv('trimmed_cleaned_data.csv')
data_df['Top_1'] = data_df['Top_1'] * 100
print(data_df.shape)
data_df.head()


# In[14]:


data_df.dropna(subset=['Top_1'], inplace=True)
data_df.drop(['GDP_growth'], axis=1, inplace=True)
data_df.dropna(thresh=4, inplace=True)
data_df.reset_index(drop=True, inplace=True)
print(data_df.shape)
data_df.head()


# In[15]:


usa_df = data_df[data_df.Country == 'United States']
print(usa_df.shape)
usa_df.head()


# In[17]:


X = usa_df.drop(['Country', 'Year', '1100', 'Top_1', '6000'], axis=1)
X.head()


# In[18]:


y = usa_df.Top_1
y.head()


# In[1]:


#preds = [0,1,2,3]


# In[ ]:


preds = gbm.predict(X)


# In[ ]:
print(preds)

with open('xgb_preds.pkl', 'wb') as f:
    pickle.dump(preds, f)

