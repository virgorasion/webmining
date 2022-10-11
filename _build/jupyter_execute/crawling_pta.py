#!/usr/bin/env python
# coding: utf-8

# # CRAWLING WEB PTA TRUNOJOYO

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
URL = [
    "https://pta.trunojoyo.ac.id/c_search/byprod/10/1",
    "https://pta.trunojoyo.ac.id/c_search/byprod/10/2",
    "https://pta.trunojoyo.ac.id/c_search/byprod/10/3",
    "https://pta.trunojoyo.ac.id/c_search/byprod/10/4",
    "https://pta.trunojoyo.ac.id/c_search/byprod/10/5"]
arr_link_dokumen = []
for link in URL:
    resp = requests.get(link)
    soup = bs(resp.text, "lxml")
    elements = soup.find_all("a", attrs={"class":"gray button"})
    for link_dokumen in elements:
        # print(link_dokumen.get('href'))
        arr_link_dokumen.append(link_dokumen.get('href'))
# print(arr_link_dokumen)


# In[2]:


arr_title = []
arr_abstract = []
for link_dokumen in arr_link_dokumen:
    resp = requests.get(link_dokumen)
    soup = bs(resp.text, "lxml")
    abstract = soup.find("p", attrs={"align":"justify"})
    title = soup.find("a", attrs={"class","title"})
    arr_abstract.append(abstract.text.replace("\r\n",""))
    arr_title.append(title.text.replace("\r\n","").upper())
    # print(title.text.replace("\r\n","").upper())
# print(arr_abstract)
print(len(arr_abstract),len(arr_title))
data = {'Title':arr_title,'Abstract':arr_abstract}
df = pd.DataFrame(data).to_csv("crawling_pta.csv")


# In[3]:


show_csv = pd.read_csv("crawling_pta.csv")
show_csv


# In[4]:


URL_DOC = elements.get('href')
resp = requests.get(URL_DOC)
data = bs(resp.text,'lxml')
print(data.prettify())

