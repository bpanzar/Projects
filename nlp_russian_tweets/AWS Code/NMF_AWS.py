
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import string
import pickle
import re


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# logging for gensim (set to INFO)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[2]:


with open("all_tweets_df.pkl", 'rb') as picklefile: 
    df = pickle.load(picklefile)


# In[3]:


print("Loaded DF with shape: ", df.shape)


# In[6]:


df_rtrolls = df[df['account_category'] == 'RightTroll']


# In[ ]:


df_rtrolls = df_rtrolls[df_rtrolls['content'].apply(len) > 40]


# In[ ]:


aus_handles = set(df_rtrolls[df_rtrolls['content'].str.contains('auspol')].author)

df_rtrolls = df_rtrolls[~df_rtrolls['author'].isin(aus_handles)]


# In[7]:


print("Right trolls df shape: ", df_rtrolls.shape)


# In[116]:


df_rtrolls.reset_index(drop=True, inplace=True)
df_rtrolls.head()


# In[18]:


# remove links
def remove_link(string):
    return re.sub(r'http[s]?\:\/\/[\S\s]\S+', '', string)


# In[24]:


df_rtrolls['content'] = df_rtrolls['content'].apply(remove_link)


# In[107]:


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
#     stemmer = SnowballStemmer('english')
#     tokens_stem = [stemmer.stem(y) for y in tokens_stop] 

    return tokens_stop


# In[108]:


tfidf = TfidfVectorizer(tokenizer=custom_tokenizer, min_df=5, max_df=0.85)
doc_vectors = tfidf.fit_transform(df_rtrolls.content)


# In[109]:


nmf = NMF(n_components=20, alpha=.1, l1_ratio=.5)
nmf_vecs = nmf.fit_transform(doc_vectors)


# In[110]:


feature_names = tfidf.get_feature_names()


# In[105]:


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# In[111]:


print_top_words(nmf, feature_names, 15)


# In[ ]:


topic_dict = {}
for topic_idx, topic in enumerate(nmf.components_):
    topic_dict[topic_idx] = ", ".join([feature_names[i]                                 for i in topic.argsort()[:-10 - 1:-1]])
    
print("Dictionary of topics to words:")
print(topic_dict)


# ### Storage

# In[ ]:


print("Storing data:")


# In[ ]:



with open('nmf.pkl', 'wb') as picklefile:
pickle.dump(nmf, picklefile)

with open('tfidf.pkl', 'wb') as picklefile:
pickle.dump(tfidf, picklefile)

import json
with open('topics2words.json', 'w') as fp:
json.dump(topic_dict, fp)


# In[ ]:


print("adding topics to DF")


# In[ ]:


import operator
topics = []
for item in nmf_vecs:
    max_index, max_value = max(enumerate(item), key=operator.itemgetter(1))
    topics.append(max_index) 
    
df_rtrolls["topicnumber"] = pd.Series(topics, index=df_rtrolls.index)


# In[ ]:


topics_likelihood = []
for item in nmf_vecs:
    max_index, max_value = max(enumerate(item), key=operator.itemgetter(1))
    topics_likelihood.append(max_value)
    
df_rtrolls["strengthoftopic"] = pd.Series(topics_likelihood, index=df_rtrolls.index)     


# In[ ]:


print(df_rtrolls.topicnumber.value_counts()) #let's make sure this is a good model...


# In[ ]:


df_rtrolls['week'] = (pd.DatetimeIndex(df_rtrolls.date).week + 
                    (pd.DatetimeIndex(df_rtrolls.date).year-2015)*52)


# In[ ]:


with open("rtrolls_df.pkl", 'wb') as picklefile:
    pickle.dump(df_rtrolls, picklefile) 

