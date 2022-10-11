#!/usr/bin/env python
# coding: utf-8

# # Crawling Twitter

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[128]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/webmining/webmining/')


# In[129]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')
get_ipython().system('pip install twint')
get_ipython().system('pip install aiohttp==3.7.0')


# In[130]:


import twint


# In[131]:


get_ipython().system('pip install nest_asyncio')
import nest_asyncio
nest_asyncio.apply() 


# In[132]:


get_ipython().system('pwd')


# In[133]:


c = twint.Config()
c.Search = '#percumalaporpolisi'
c.Pandas = True
c.Limit = 60
c.Store_csv = True
c.Custom["tweet"] = ["tweet"]
c.Output = "dataset.csv"
twint.run.Search(c)


# In[134]:


import pandas as pd


# In[135]:


import random
read_file = pd.read_csv ('dataset.csv')
label = ['positif','netral','negatif']
data = []
for i in range(read_file.size):
  rand = random.randint(0,2)
  data.append(label[rand])
read_file.insert(1,"label",data, True)
print(read_file)


# In[136]:


read_file.to_excel (r'dataset.xlsx', index = None, header=True)
data = pd.read_excel('dataset.xlsx')
data


# In[137]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# In[138]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# In[139]:


def remove_stopwords(text):
    with open('/content/drive/MyDrive/webmining/stopword.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# In[140]:


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    result = [stemmer.stem(word) for word in text]
    
    return result


# In[141]:


def preprocessing(text):
    #case folding
    text = text.lower()

    #remove non ASCII (emoticon, chinese word, .etc)
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")

    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')

    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())

    #replace weird characters
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('-', ' ')

    #tokenization and remove stopwords
    text = remove_stopwords(text)

    #remove punctuation    
    text = [''.join(c for c in s if c not in string.punctuation) for s in text]  

    #stemming
    text = stemming(text)

    #remove empty string
    text = list(filter(None, text))
    return text


# In[142]:


data['tweet'].apply(preprocessing).to_csv('preprocessing.csv')


# In[143]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_csv('/content/drive/MyDrive/webmining/webmining/twint/preprocessing.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])


# In[144]:


matrik_vsm=bag.toarray()
matrik_vsm.shape


# In[145]:


matrik_vsm[0]


# In[146]:


a=vectorizer.get_feature_names()


# In[147]:


dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# In[148]:


label = pd.read_excel('/content/drive/MyDrive/webmining/webmining/twint/dataset.xlsx')
dj = pd.concat([dataTF.reset_index(), label["label"]], axis=1)
dj


# In[149]:


dj['label'].unique()


# In[150]:


dj.info()


# In[151]:


get_ipython().system('pip install -U scikit-learn')


# In[152]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dj.drop(labels=['label'], axis=1),
    dj['label'],
    test_size=0.3,
    random_state=0)


# In[153]:


X_train


# In[154]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info


# In[155]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# In[156]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(50, 20))


# In[157]:


from sklearn.feature_selection import SelectKBest
sel_five_cols = SelectKBest(mutual_info_classif, k=100)
sel_five_cols.fit(X_train, y_train)
X_train.columns[sel_five_cols.get_support()]

