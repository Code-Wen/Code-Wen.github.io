---
title: "Matching Documents with TFIDF Weighted Document Vectors"
excerpt: "Combine the power of TFIDF and word vectors to relate documents."
collection: projects
---
## Recommending jobs based on resume 
### TFIDF, Document Vector, and TFIDF-Weighted Document Vector

First, install necessary packages.


```python
# install spacy and download english library
!pip install -U spacy
!python -m spacy download en_core_web_md
```

Read in data.


```python
!pip install docx2txt
```


```python
!pip install PyPDF2
```


```python
import glob
import docx2txt
import PyPDF2
# read in jobs
jobs = []
fnames = []
# docx documents
for fname in glob.iglob('data/jobs/*.docx', recursive=True):
    fnames.append(fname)
    jobs.append( docx2txt.process(fname) )
# DOCX documents
for fname in glob.iglob('data/jobs/*.DOCX', recursive=True):
    fnames.append(fname)
    jobs.append( docx2txt.process(fname) )
# pdf documents
for fname in glob.iglob('data/jobs/*.pdf', recursive=True):
    fnames.append(fname)
    with open(fname,'rb') as f:
        pdf_reader = PyPDF2.PdfFileReader(f)
        tmp = ''
        for i in range(pdf_reader.numPages):
            page = pdf_reader.getPage(i).extractText()
            tmp += page
    jobs.append(tmp)
```


```python
# read in resumes
resumes = []
# docx documents
for fname in glob.iglob('data/resumes/*.docx', recursive=True):
    fnames.append(fname)
    resumes.append( docx2txt.process(fname) )
# pdf documents
for fname in glob.iglob('data/resumes/*.pdf', recursive=True):
    fnames.append(fname)
    with open(fname,'rb') as f:
        pdf_reader = PyPDF2.PdfFileReader(f)
        tmp = ''
        for i in range(pdf_reader.numPages):
            page = pdf_reader.getPage(i).extractText()
            tmp += page
    resumes.append(tmp)
```


```python
# import and load library
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')
```


```python
# keep only lemmatization of words that are not space, stop words, punctuation or number
docs = []
# create a list for doc vector
docs_vec = []
# create a dictionary mapping the lemmatizations to word vectors
word_vec = dict()
for doc in jobs+resumes:
    tmp = nlp(doc.lower())
    docs_vec.append(tmp.vector)
    doc_i = ''
    for t in tmp:
        if t.is_alpha and t.has_vector and not (t.is_space or t.is_punct or t.is_stop or t.like_num): 
            doc_i += ' ' + t.lemma_
            if t not in word_vec:
                word_vec[t.lemma_] = t.vector
    docs.append(doc_i)
```


```python
# get word vectors in a numpy array
words_vec = np.vstack([v for k, v in word_vec.items()])
word_order = [k for k, v in word_vec.items() ]
```


```python
# get tfidf 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(docs)
tfs = tfidf.fit_transform(docs)
doc_tfidf = tfs.todense()
```


```python
# if different words included, check if important ones are missing
if words_vec.shape[0] != tfs.shape[1]:
    print(set(word_order).difference(set(tfidf.get_feature_names())))
```

    {'y', 'w', 'r', 'n', 'm', 'x', 'd', 't', 'e', 'l', 'o', 'c', 's', 'p', 'f'}



```python
# slice word_vec to have the same number of words as tfidf
ind = [word_order.index(i) for i in tfidf.get_feature_names()]
words_vec = words_vec[ind,]
```


```python
# calculate tfidf weighted word vec
doc_wt_vec = np.dot(doc_tfidf, words_vec)
```


```python
# Import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix
tfidf_cos_sim = cosine_similarity(tfs, tfs)
wt_vec_cos_sim = cosine_similarity(doc_wt_vec, doc_wt_vec)
doc_vec_cos_sim = cosine_similarity(docs_vec, docs_vec)
```


```python
import seaborn as sns
sns.heatmap(tfidf_cos_sim, cmap="YlGnBu", xticklabels='', yticklabels=fnames)
```



```python
sns.heatmap(wt_vec_cos_sim, cmap="YlGnBu", xticklabels=fnames, yticklabels=fnames)
```



```python
sns.heatmap(doc_vec_cos_sim, cmap="YlGnBu", xticklabels=fnames, yticklabels=fnames)
```


```python
def recommend_jobs(resume = 9):
    jobs_ind = np.array(['job' in i for i in fnames])
    jobs_sort = tfidf_cos_sim[resume,jobs_ind]
    # recommendation list
    rec_list = jobs_sort.argsort()[-1:-7:-1]
    # remove self
    rec_list = np.setdiff1d(rec_list,resume)
    rec_df = pd.DataFrame(dict(index = rec_list, 
                               filename = np.array(fnames)[rec_list]))
    return rec_df
```


```python
resume = 9
recommend_jobs(resume)
```




<div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>data/jobs/Asst Manager Trust Administration.docx</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>data/jobs/CDL - EVP Head of Asset Mgt.docx</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data/jobs/Citco - Hedge Fund Accountant JD.DOCX</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>data/jobs/Asst Manager Trust PDF.pdf</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>data/jobs/Corp Sec Senior Executive JD.pdf</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>data/jobs/Asst Finance Mgr - JD.pdf</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## Supervised Learning
Now, we can train a logistic regression model if we have match/nonmatch labels between resumes and jobs.

## Keras


```python
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
```

    Using TensorFlow backend.



```python
N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies
```
