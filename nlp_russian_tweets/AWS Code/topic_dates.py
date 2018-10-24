
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import pickle
import nltk

from datetime import datetime

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# In[39]:


def custom_tokenizer(text):
    full_punc = '’‘“”.–…�🇺🇸★➠' + string.punctuation
    # remove punctuation
    remove_punct = str.maketrans('', '', full_punc)
    text = text.translate(remove_punct)

    # remove digits and convert to lower case
    remove_digits = str.maketrans('', '', string.digits)
    text = text.lower().translate(remove_digits)

    # tokenize
    tokens = word_tokenize(text)

    # remove stop words
    punc = [str(i) for i in string.punctuation]
    cust_stop_words = (['rt', 'retweet', 'get', 'one', 'im', 'thing', 'get', 'dont', 'wow',
                       'lol', 'amp', 'n', 'didnt', 'people', 'like', 'want', 'know', 'go',
                        'think', 'need', 'right', 'good', 'would', 'going', 'never', 'see',
                        'time', 'call', 'said', 'got', 'us', 'p', 'look', 'mr'])
    stop_words = cust_stop_words + stopwords.words('english')
    tokens_stop = [y for y in tokens if y not in stop_words]

    # stem
#    stemmer = SnowballStemmer('english')
#    tokens_stem = [stemmer.stem(y) for y in tokens_stop] 

    return tokens_stop


# In[40]:


with open("rtrolls_df.pkl", 'rb') as picklefile:
    df_rtrolls = pickle.load(picklefile)    
    
import json
with open('topics2words.json', 'r') as fp:
    topic_dict = json.load(fp)


# In[41]:


df_rtrolls.head()


# In[42]:


#group by week
temp_df = df_rtrolls.groupby(["week", "topicnumber"]).count().reset_index()
# temp_df

topic_weeks_df = temp_df[['week', 'topicnumber', 'content']]
# topic_weeks_df


# In[43]:


temp_df = topic_weeks_df[((topic_weeks_df['topicnumber'] == 0) |
        (topic_weeks_df['topicnumber'] == 15) |
        (topic_weeks_df['topicnumber'] == 2) | 
        (topic_weeks_df['topicnumber'] == 4) |
        (topic_weeks_df['topicnumber'] == 19) |
        (topic_weeks_df['topicnumber'] == 11) |     
        (topic_weeks_df['topicnumber'] == 16) |               
        (topic_weeks_df['topicnumber'] == 7) |
        (topic_weeks_df['topicnumber'] == 5) |               
        (topic_weeks_df['topicnumber'] == 13))]


# In[44]:


data_fillna = temp_df.pivot_table('content', 'week', 'topicnumber').fillna(0).unstack().reset_index()


# In[45]:


data_fillna.head()


# In[46]:


#we lose the count label column in the previous steps, so we're just renaming it here, and reordering columns based on 
#how they are arranged in the viz csv
data_fillna.columns = ["topicnumber", "week", "content"]
data_fillna = data_fillna[["week", "topicnumber", "content"]]
data_fillna.head()


# In[47]:


data_fillna.sort_values('week', inplace=True)


# In[48]:


#backup file
data_fillna.to_csv("topicsbyweek.csv", index = False)

