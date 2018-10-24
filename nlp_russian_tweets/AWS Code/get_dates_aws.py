
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle

from datetime import datetime
# import matplotlib.pyplot as plt
# %matplotlib inline


# In[1]:


def load_csv(file):
    path = 'Data/russian-troll-tweets/'
    new_df = pd.read_csv(path + file)
    new_df = new_df[((new_df['account_category'] != 'NonEnglish') &
        (new_df['account_category'] != 'Commercial') &
        (new_df['account_category'] != 'Unknown'))]
    new_df['date'] = new_df['publish_date'].apply(pd.to_datetime)
    return new_df


# In[2]:


print('Loading csv 1...')
df = load_csv('IRAhandle_tweets_1.csv')
print('Loading csv 2...')
new_df = load_csv('IRAhandle_tweets_2.csv')
df = df.append(new_df)
print('Loading csv 3...')
new_df = load_csv('IRAhandle_tweets_3.csv')
df = df.append(new_df)
print('Loading csv 4...')
new_df = load_csv('IRAhandle_tweets_4.csv')
df = df.append(new_df)
print('Loading csv 5...')
new_df = load_csv('IRAhandle_tweets_5.csv')
df = df.append(new_df)
print('Loading csv 6...')
new_df = load_csv('IRAhandle_tweets_6.csv')
df = df.append(new_df)
print('Loading csv 7...')
new_df = load_csv('IRAhandle_tweets_7.csv')
df = df.append(new_df)
print('Loading csv 8...')
new_df = load_csv('IRAhandle_tweets_8.csv')
df = df.append(new_df)
print('Loading csv 9...')
new_df = load_csv('IRAhandle_tweets_9.csv')
df = df.append(new_df)


# In[6]:


print('Dropping columns')
df.drop(['external_author_id', 'publish_date', 'harvested_date', 'new_june_2018',
         'account_type', 'post_type'], axis=1, inplace=True)


# In[7]:


print('Creating hour, day, date columns')
df['hour'] = pd.DatetimeIndex(df.date).hour


# In[11]:


df['day'] = pd.DatetimeIndex(df.date).weekday


# In[13]:


df['date'] = pd.DatetimeIndex(df.date).date


# In[15]:


print('Calculating # tweets per hour')
hour_tweets = df.groupby('hour').content.count().reset_index()


# In[ ]:


mst_hours = pd.Series([3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,0,1,2])
est_hours = pd.Series([19,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
mst = pd.DataFrame(mst_hours)
mst.columns = ['hour']
mst['MST'] = hour_tweets.content
est = pd.DataFrame(est_hours)
est.columns = ['hour']
est['EST'] = hour_tweets.content


# In[ ]:


hour_tweets = pd.merge(mst, est, on='hour')


# In[16]:


print('Calculating # tweets per day')
day_tweets = df.groupby('day').content.count().reset_index()
day_tweets['day'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# In[17]:


print('Calculating # tweets per date')
date_tweets = df.groupby('date').content.count().reset_index()


# In[ ]:


print('Storing main DF')
df.to_csv('all_tweets.csv', index=False)
print('Storing sub DFs')
hour_tweets.to_csv('hour_tweets.csv', index=False)
day_tweets.to_csv('day_tweets.csv', index=False)
date_tweets.to_csv('date_tweets.csv', index=False)

print('Pickling DF')
with open('all_tweets_df.pkl', 'wb') as picklefile:
    pickle.dump(df, picklefile)

