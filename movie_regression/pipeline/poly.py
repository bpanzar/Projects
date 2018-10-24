
# coding: utf-8

# In[31]:


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV
import statsmodels.api as sm
import statsmodels.formula.api as smf 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# In[36]:


df = pd.read_csv('data/df_model.csv')


# In[37]:


print("The shape of the DataFrame is:", df.shape)
df.head()


# In[38]:


df.info()


# In[39]:


X = df.drop(['Title', 'Domestic_Gross', 'Worldwide_Gross', 
             'Release_Date'],axis=1)
y = df['Domestic_Gross']

X = pd.get_dummies(X)
print(X.shape)
X.head()


# ### Check Polynomial Degree

# In[ ]:


r2score = [[] for i in range(3)]

for i in range(1, 4):
    print("Degree:", i)
    kf = KFold(n_splits=5, shuffle=True, random_state = 10)
    k = 1
    for train_ind, test_ind in kf.split(X, y):
        print('k:', k)
        k += 1
        
        poly = PolynomialFeatures(degree=i, interaction_only=True)
        
        X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
        X_test, y_test = X.iloc[test_ind], y.iloc[test_ind] 

        #Feature transforms for train, val, and test so that we can run our poly model on each
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        lr_poly = LinearRegression()
        lr_poly.fit(X_train_poly, y_train)
        r2score[i-1].append(lr_poly.score(X_test_poly, y_test))


# In[ ]:


print(r2score)


# In[ ]:


for i in range(3):
    print('R^2 of degree %: %.3f +- %.3f' %(i+1, 
                            np.mean(r2score[i]),np.std(r2score[i])))
