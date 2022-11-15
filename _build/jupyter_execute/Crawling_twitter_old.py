#!/usr/bin/env python
# coding: utf-8

# # Crawling Twitter Using Twint
# Twint adalah library python yang difungsikan untuk crawling data timeline di twitter dengan cara yang sangat mudah dan simpel.
# 
# Karena kodingan ini dibuat di colab oleh karena itu kita sambungkan terlebih dahulu ke google drive kita.

# Hubungkan colab ke google drive

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[6]:


get_ipython().run_line_magic('cd', 'drive/MyDrive/webmining/TugasWebmining/')


# instalasi twint. lebih jelasnya dapat dilihat di link yang akan di cloning

# In[7]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')
get_ipython().system('pip install twint')
get_ipython().system('pip install aiohttp==3.7.0')
get_ipython().system('pip install nest_asyncio')
import twint
import nest_asyncio
nest_asyncio.apply()


# cek, apakah kita sudah berada di direktori yang diinginkan

# In[8]:


get_ipython().system('pwd')


# Konfigurasi crawling dengan twint

# In[ ]:


# c = twint.Config()
# c.Search = '#percumalaporpolisi'
# c.Pandas = True
# c.Limit = 60
# c.Store_csv = True
# c.Custom["tweet"] = ["tweet"]
# c.Output = "dataset.csv"
# twint.run.Search(c)


# In[9]:


import pandas as pd


# In[10]:


import random
read_file = pd.read_csv ('dataset.csv')
# label = ['positif','netral','negatif']
# data = []
# for i in range(read_file.size):
#   rand = random.randint(0,2)
#   data.append(label[rand])
# read_file.insert(1,"label",data, True)
# pd.DataFrame(read_file).to_csv('dataset.csv')
print(read_file)


# In[11]:


read_file.to_excel (r'dataset.xlsx', index = None, header=True)
data = pd.read_excel('dataset.xlsx')
data


# In[12]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# In[38]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# In[39]:


def remove_stopwords(text):
    with open('/content/drive/MyDrive/webmining/stopword.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# In[40]:


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    result = [stemmer.stem(word) for word in text]
    
    return result


# In[41]:


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


# In[42]:


data['tweet'].apply(preprocessing).to_csv('preprocessing.csv')


# In[43]:


pd.read_csv('preprocessing.csv')


# In[44]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_csv('/content/drive/MyDrive/webmining/TugasWebmining/twint/preprocessing.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])


# In[45]:


matrik_vsm=bag.toarray()
matrik_vsm.shape


# In[46]:


matrik_vsm[0]


# In[47]:


a=vectorizer.get_feature_names_out()


# In[48]:


dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# In[49]:


label = pd.read_excel('/content/drive/MyDrive/webmining/TugasWebmining/twint/dataset.xlsx')
dj = pd.concat([dataTF.reset_index(), label["label"]], axis=1)
dj


# In[50]:


dj['label'].unique()


# In[51]:


dj.info()


# In[52]:


get_ipython().system('pip install -U scikit-learn')


# In[53]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dj.drop(labels=['label'], axis=1),
    dj['label'],
    test_size=0.3,
    random_state=0)


# In[54]:


y_train


# In[55]:


X_train


# In[56]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info


# In[57]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# In[58]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(50, 20))


# In[59]:


from sklearn.feature_selection import SelectKBest
sel_five_cols = SelectKBest(mutual_info_classif, k=100)
sel_five_cols.fit(X_train, y_train)
X_train.columns[sel_five_cols.get_support()]


# Prediksi Dengan Clustering

# In[60]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(dataTF)
prediksi = kmeans.predict(dataTF)
centroids = kmeans.cluster_centers_


# In[61]:


prediksi


# In[62]:


pd.DataFrame(prediksi,columns=['Cluster'])

