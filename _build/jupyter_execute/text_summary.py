#!/usr/bin/env python
# coding: utf-8

# # TEXT SUMMARY

# In[1]:


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


    


# In[2]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_csv('preprocessing_summary.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['kata'])


# In[3]:


matrik_vsm=bag.toarray()
matrik_vsm.shape


# In[4]:


matrik_vsm[0]


# In[5]:


a=vectorizer.get_feature_names_out()


# In[6]:


dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# In[ ]:





# In[7]:


import matplotlib.pyplot as plt
import networkx as nx
from nltk.probability import DictionaryConditionalProbDist
data = pd.read_csv("preprocessing_summary.csv")
G = nx.DiGraph()
G = nx.from_pandas_edgelist(data,edge_key='',edge_attr='kata',create_using=nx.Graph())
nx.draw(G)
plt.show()
# dataTF.


# In[26]:


from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

tokens = nltk.FreqDist(dataTF)
tokens
tokens.plot(30,cumulative=False)
plt.show()


# In[27]:


#preprocessing
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet
from sklearn.decomposition import TruncatedSVD
#stop-words
stop_words=set(nltk.corpus.stopwords.words('indonesian'))
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)
vect=TfidfVectorizer(stop_words=stop_words,max_features=1000)
vect_text=vect.fit_transform(data['kata'])
lsa_top=lsa_model.fit_transform(vect_text)


# In[28]:


print(lsa_top)
# data_plot = pd.Series(lsa_top)
# data_plot.index = dataTF.columns
# data_plot.sort_values(ascending=False)
# data_plot.sort_values(ascending=False).plot.bar(figsize=(50, 20))


# In[29]:


label = pd.read_excel('/content/drive/MyDrive/webmining/TugasWebmining/twint/dataset.xlsx')
dj = pd.concat([dataTF.reset_index(), label["label"]], axis=1)
dj


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataTF,
    dataTF,
    test_size=0.3,
    random_state=0)


# In[ ]:


y_train


# In[ ]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info


# In[ ]:


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

