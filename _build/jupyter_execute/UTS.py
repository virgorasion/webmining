#!/usr/bin/env python
# coding: utf-8

# # Tugas UTS Web Mining
# Nama : M Nur Fauzan W || NIM  : 190411100064

# ## 1. Membuat Analisa Clustring dengan K-Means

# In[1]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')
get_ipython().system('pip install twint')
get_ipython().system('pip install aiohttp==3.7.0')
get_ipython().system('pip install nest_asyncio')


# In[2]:


get_ipython().run_line_magic('cd', '../..')
import twint
import pandas as pd
import re
import numpy as np
get_ipython().system('pip install nltk')
import nltk
nltk.download('punkt')
import string
get_ipython().system('pip install Sastrawi')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nest_asyncio
nest_asyncio.apply()

# c = twint.Config()
# c.Search = 'tragedi kanjuruhan'
# c.Pandas = True
# c.Limit = 60
# c.Store_csv = True
# c.Custom["tweet"] = ["tweet"]
# c.Output = "datatweet.csv"
# twint.run.Search(c)


# In[24]:


data = pd.read_csv('datatweet.csv')
data


# In[40]:


def remove_stopwords(text):
    with open('stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]      
    return text

def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    result = [stemmer.stem(word) for word in text]
    return result

def preprocessing(text):
    #case folding
    text = text.lower()

    #remove non ASCII (emoticon, chinese word, .etc)
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")

    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')

    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    text = ' '.join(re.sub("([0-9])","", text).split())

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


# In[41]:


data['tweet'].apply(preprocessing).to_csv('preprocessing_tweet.csv')


# In[59]:


pd.read_csv('preprocessing_tweet.csv')


# In[44]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_csv('preprocessing_tweet.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])

matrik_vsm=bag.toarray()
matrik_vsm.shape


# In[45]:


matrik_vsm[0]


# In[46]:


a=vectorizer.get_feature_names_out()


# In[47]:


dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# In[48]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(dataTF)
prediksi = kmeans.predict(dataTF)
centroids = kmeans.cluster_centers_


# In[49]:


prediksi


# In[73]:


cluster = pd.DataFrame(prediksi,columns=['Cluster'])
data = pd.read_csv('preprocessing_tweet.csv')
cluster


# ## 2. Membuat Ringkasan Dokumen Dari Berita Online Dengan Metode PageRank

# In[51]:


import requests
import pandas as pd
import pandas as pd
import re
import numpy as np
import nltk
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from bs4 import BeautifulSoup as bs
URL = "https://news.detik.com/berita/d-6334129/5-fakta-baru-tragedi-kanjuruhan-6-tersangka-hingga-soal-gas-air-mata?single=1"
resp = requests.get(URL)
soup = bs(resp.text, "lxml")
elements = soup.find("div", attrs={"class":"detail__body-text itp_bodycontent"}).find_all('p')
dokumen = []
split_kalimat = [] 
for i,paragraf in enumerate(elements):
    dokumen.append(paragraf.text)
    # print(kalimat)
    for j,kalimat in enumerate(dokumen[i].split(".")):
        # print(len(kalimat))
        if len(kalimat) == 0:
            continue
        else:
            split_kalimat.append(kalimat)
def remove_stopwords(text):
    with open('stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]              
    return text

def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    result = [stemmer.stem(word) for word in text]
    return result

def preprocessing(kalimat):
    import re
    res_kata = []
    for j,kata in enumerate(kalimat.split(" ")):
        kata = kata.lower()
        kata = kata.replace(",", "")
        kata = kata.replace("\r\n","")
        kata = kata.replace("\"","")
        kata = ''.join(re.sub("[0-9]","",kata))
        if len(kata) > 0:
            res_kata.append(kata)
    # print(text)
    return res_kata

df = pd.DataFrame(split_kalimat,columns=['kata'])
df['kata'].apply(preprocessing).to_csv("preprocessing_summary.csv")
data = pd.read_csv("preprocessing_summary.csv")
data


    


# In[52]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_csv('preprocessing_summary.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['kata'])


# In[53]:


matrik_vsm=bag.toarray()
matrik_vsm.shape


# In[54]:


matrik_vsm[0]


# In[55]:


a=vectorizer.get_feature_names_out()


# In[56]:


dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# In[57]:


import matplotlib.pyplot as plt
import networkx as nx
from nltk.probability import DictionaryConditionalProbDist
data = pd.read_csv("preprocessing_summary.csv")
G = nx.DiGraph()
G = nx.from_pandas_edgelist(data,edge_key='',edge_attr='kata',create_using=nx.Graph())
nx.draw(G)
plt.show()
# dataTF.

