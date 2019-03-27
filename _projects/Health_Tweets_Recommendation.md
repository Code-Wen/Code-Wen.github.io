---
title: "Health Tweets Recommendation"
excerpt: "Using NLP techniques, I build a recommendation system for health related tweets."
collection: projects
---

## 0. Introduction

Twitter has already become one of the most popular news and social networking online service. Almost all main-stream news agencies have their Twitter accounts to broadcast news in different topics. 

The number of tweets per hour is massive, thus it would be ideal to develop a **recommendation system** for Twitter users which suggests the most relevant tweets based on the current content. We try to use **Natural Language Processing (NLP)** techniques to achieve this goal here.

## 1. First glance of the data

Our data comes from the UCI Machine Learning Repository https://archive.ics.uci.edu/ml/index.php, a fantastic source of data sets for the machine learning community. 

The dataset at hand contains health news from more than 15 major health news agencies such as BBC, CNN, and NYT in 2015. The folder `Health-Tweets` contains 16 files in the `.txt` format, and each file stores the Health related tweets for one news agency. Each line contains the information of a single tweet `id|date and time|tweet content`. The separator is '|'.

As the first step,  we would like to take a look at the names of the files. Let's list all the `.txt` files and sort them alphabetically using `glob`.

 


```python
# Import library
import glob

# The books files are contained in this folder
folder = "Health-Tweets/"

# List all the .txt files and sort them alphabetically
files = glob.glob(folder+'*.txt')
# print files
print(files)
```

    ['Health-Tweets/cbchealth.txt', 'Health-Tweets/wsjhealth.txt', 'Health-Tweets/nprhealth.txt', 'Health-Tweets/goodhealth.txt', 'Health-Tweets/NBChealth.txt', 'Health-Tweets/cnnhealth.txt', 'Health-Tweets/foxnewshealth.txt', 'Health-Tweets/everydayhealth.txt', 'Health-Tweets/gdnhealthcare.txt', 'Health-Tweets/nytimeshealth.txt', 'Health-Tweets/bbchealth.txt', 'Health-Tweets/KaiserHealthNews.txt', 'Health-Tweets/usnewshealth.txt', 'Health-Tweets/latimeshealth.txt', 'Health-Tweets/reuters_health.txt', 'Health-Tweets/msnhealthnews.txt']


## 2. Importing the data into Python

Next, we need to load the content of these books into Python and do some basic pre-processing to facilitate the downstream analyses. We call such a collection of texts a corpus. Our recommendation system is unbiased towards different news agencies, hence we store all the tweets in a single list `txts`.






```python
# Import libraries
import re, os

# Initialize the object that will contain the texts and titles
txts = []
titles = []

for n in files:
    # Open each file
    f = open(n)
    txt = f.read()
    # Store the texts and titles of the books in two separate lists
    txts.extend(txt.splitlines())
    
# Print the number of all tweets
print(len(txts))
```

    63327


## 3. Tokenize the corpus

As a next step, we need to transform the corpus into a format that is easier to deal with for the downstream analyses. We will tokenize our corpus, i.e., transform each text into a list of the individual words (called tokens) it is made of. 

To this end, we first build a set of stop words which are non-informative called `stoplist`. Then, we remove the empty lines from the `txts` list and call the resulting list `txts_orig`, which stores all the original information. Next, we remove all the non-alpha-numeric characters except the in-line separator '|' and then change everything to lower case.  For simplicity, we will drop the idenfication number and the date/time, and focus on the contents of the tweets. After splitting the texts in each tweet, we remove the stop words and obtain our list of tokens called `texts`. 

To see what the elements in the list `texts` look like, we print out the first element.


```python
# Define a list of stop words
stoplist = set('http www com html for a of the and to in to be which some is at that we i who whom show via may my our might as well'.split())

# clean out tweets without two vertical lines
txts_orig = [i for i in txts if i.count('|')==2]

# Remove all non-alpha-numeric characters
txt = [ re.sub('[^|a-zA-Z0-9\s]', ' ', i) for i in txts_orig ]
    
# Convert the text to lower case 
txts_lower_case = [i.lower() for i in txt]
    
# Transform the text into tokens 
txts_split = [ i.split('|')[2].split() for i in txts_lower_case]

# Remove tokens which are part of the list of stop words
texts = [ [w for w in i if w not in stoplist] for i in txts_split]

# Print the first 20 tokens for the first file
print(texts[0])
```

    ['drugs', 'need', 'careful', 'monitoring', 'expiry', 'dates', 'pharmacists', 'say', 'cbc', 'ca', 'news', 'health', 'drugs', 'need', 'careful', 'monitoring', 'expiry', 'dates', 'pharmacists', 'say', '1', '3026749', 'cmp', 'rss']


## 4. Stemming of the tokenized corpus

One usually use different words to refer to a similar concept. This will dilute the weight given to this concept in the tweet and potentially bias the results of the analysis.

To solve this issue, it is a common practice to use a stemming process, which will group together the inflected forms of a word so they can be analysed as a single item: the stem. The package `nltk` provides an easy method of generating the stems. The stems are stored in the list `texts_stem`.


```python
# Load the Porter stemming function from the nltk package
from nltk.stem import PorterStemmer

# Create an instance of a PorterStemmer object
porter = PorterStemmer()

# For each token of each text, we generated its stem 
texts_stem = [[porter.stem(token) for token in text] for text in texts]

# Print the first stemmed tweet
print(texts_stem[0])
```

    [u'drug', 'need', u'care', u'monitor', u'expiri', u'date', u'pharmacist', 'say', 'cbc', 'ca', u'news', 'health', u'drug', 'need', u'care', u'monitor', u'expiri', u'date', u'pharmacist', 'say', '1', '3026749', 'cmp', u'rss']


## 5. Building a bag-of-word model

First, we need to create a universe of all words contained in our corpus of tweets, which we call a dictionary. Then, using the stemmed tokens and the dictionary, we will create bag-of-words models (BoW) for each of our tweets. The BoW models will represent our tweets as a list of all uniques tokens they contain associated with their respective number of occurrences.


```python
# Load the functions allowing to create and use dictionaries
from gensim import corpora

# Create a dictionary from the stemmed tokens
dictionary = corpora.Dictionary(texts_stem)

# Create a bag-of-words model for each tweet, using the previously generated dictionary
bows = [dictionary.doc2bow(i) for i in texts_stem]

# Print the first tweet's BoW model
print(bows[0])
```

    [(0, 1), (1, 1), (2, 1), (3, 2), (4, 1), (5, 1), (6, 2), (7, 2), (8, 2), (9, 1), (10, 2), (11, 2), (12, 1), (13, 2), (14, 1), (15, 2)]


## 6. The most common words of a given tweet

The results returned by the bag-of-words model is certainly easy to use for a computer but hard to interpret for a human. It is not straightforward to understand which stemmed tokens are present in a given tweet, and how many occurrences we can find.

In order to better understand how the model has been generated and visualize its content, we will transform it into a DataFrame, and sort the occurrences in descending order.


```python
# Import pandas to create and manipulate DataFrames
import pandas as pd

# Convert the BoW model for first tweet
df_bow_0 = pd.DataFrame(bows[0])

# Add the column names to the DataFrame
df_bow_0.columns = ['index','occurrences']

# Add a column containing the token corresponding to the dictionary index
df_bow_0['token'] = [dictionary[i] for i in df_bow_0['index']]

# Sort the DataFrame by descending number of occurrences and print 
df_bow_0.sort_values(by = 'occurrences', ascending = False, inplace=True)
print(df_bow_0)
```

        index  occurrences       token
    3       3            2        care
    6       6            2        date
    7       7            2        drug
    8       8            2      expiri
    10     10            2     monitor
    11     11            2        need
    13     13            2  pharmacist
    15     15            2         say
    0       0            1           1
    1       1            1     3026749
    2       2            1          ca
    4       4            1         cbc
    5       5            1         cmp
    9       9            1      health
    12     12            1        news
    14     14            1         rss


## 7. Build a tf-idf model

Some of the most recurring words are very common and unlikely to carry any information peculiar to the given tweet. We need to use an additional step in order to determine which tokens are the most specific to a book.

To do so, we will use a tf-idf model (term frequency–inverse document frequency). This model defines the importance of each word depending on how frequent it is in this text and how infrequent it is in all the other documents. As a result, a high tf-idf score for a word will indicate that this word is specific to this text. This can be done with the help of the package `gensim`.




```python
# Load the gensim functions that will allow us to generate tf-idf models
from gensim.models import TfidfModel

# Generate the tf-idf model
model = TfidfModel(corpus=bows)

# Print the model for the first tweet
print(model[bows[0]])
```

    [(0, 0.08581568812538018), (1, 0.30253680106149267), (2, 0.094606136667876), (3, 0.1928206151948339), (4, 0.09488248723818501), (5, 0.0960166646640679), (6, 0.3587333982617377), (7, 0.1826391735878661), (8, 0.5449305305305571), (9, 0.06169054042418596), (10, 0.3394512287510965), (11, 0.21504701074076696), (12, 0.09066898233627706), (13, 0.41043755517197905), (14, 0.09626385303997112), (15, 0.1628941252282041)]


## 8. The results of the tf-idf model
Once again, the format of those results is hard to interpret for a human. Therefore, we will transform it into a more readable version and display the words from the first tweet with their tf-idf scores.


```python
# Convert the tf-idf model for the first tweet into a DataFrame
df_tfidf = pd.DataFrame(model[bows[0]])

# Name the columns of the DataFrame id and score
df_tfidf.columns = ['id','score']

# Add the tokens corresponding to the numerical indices for better readability
df_tfidf['token'] = [dictionary[i] for i in df_tfidf['id']]

# Sort the DataFrame by descending tf-idf score and print the first 10 rows.
df_tfidf.sort_values(by = 'score', ascending=False, inplace=True)
print(df_tfidf)
```

        id     score       token
    8    8  0.544931      expiri
    13  13  0.410438  pharmacist
    6    6  0.358733        date
    10  10  0.339451     monitor
    1    1  0.302537     3026749
    11  11  0.215047        need
    3    3  0.192821        care
    7    7  0.182639        drug
    15  15  0.162894         say
    14  14  0.096264         rss
    5    5  0.096017         cmp
    4    4  0.094882         cbc
    2    2  0.094606          ca
    12  12  0.090669        news
    0    0  0.085816           1
    9    9  0.061691      health


## 9. Compute distance between texts

The results of the tf-idf algorithm now return stemmed tokens which are specific to each tweet.  Now that we have a model associating tokens to how specific they are to each tweet, we can measure how each tweet is related to the given tweet.

To this purpose, we will use a measure of similarity called **cosine similarity** and we will visualize the results as a distance matrix, i.e., a matrix showing all distances between each tweet and the given tweet.


```python
# Load the library allowing similarity computations
from gensim import similarities

# Compute the similarity matrix (pairwise distance between all texts)
corpus_tfidf = model[bows]
#sims = similarities.MatrixSimilarity(corpus=corpus_tfidf)
sims = similarities.Similarity('/Users/mshen/Desktop/',corpus_tfidf, num_features=len(dictionary))

```

## 10. Recommending the most related tweets

Now we are set! Once we have the similarity scores between the current tweet and each tweet in the dataset, we can sort the scores and get the top `n` tweets. Those should be the most related tweets and will likely be of interest to the reader! 


```python
import numpy as np

n = 5 # the number of top similar tweets to output
i = 10 # the index of the current tweet to look for similar ones

query_doc = model[bows[i]] # tf-idf for the current tweet
s = sims[query_doc] # similarity array
x = sorted(s, reverse=True) # sorted similarity array

thres = x[n] # the threshold similairty for top n similar tweets
ind = s >= thres # indices of the top similar tweet
filtered = np.array(txts_orig)[ind]

# print the current tweet
print ('The current tweet: \n' + filtered[0])

# print the top n similar tweets
print('\nThe top ' + str(n) + ' similar tweets:')
for i in filtered[1:]:
    print(i)
```

    The current tweet: 
    585906983091376128|Wed Apr 08 20:48:05 +0000 2015|Check expiry dates, Health Canada advises after Alesse 21 birth control pill recall http://www.cbc.ca/news/health/alesse-21-birth-control-pills-recalled-in-western-canada-1.3025078?cmp=rss
    
    The top 5 similar tweets:
    585579956371066880|Tue Apr 07 23:08:35 +0000 2015|Expired Alesse birth control exposes 'deficiency' http://www.cbc.ca/news/health/expired-alesse-birth-control-exposes-deficiency-1.3023939?cmp=rss
    585416006467624960|Tue Apr 07 12:17:07 +0000 2015|Shoppers Drug Mart mistakenly sells expired birth control pills in Western Canada http://www.cbc.ca/news/canada/edmonton/shoppers-drug-mart-warns-of-expired-alesse-birth-control-pills-in-western-canada-1.3022846?cmp=rss
    521665128786178048|Mon Oct 13 14:13:53 +0000 2014|Birth control pill threatens fish populations http://www.cbc.ca/news/technology/birth-control-pill-threatens-fish-populations-1.2796897?cmp=rss
    377881233252155393|Wed Sep 11 19:48:01 +0000 2013|Birth control pill recalls show 'weak link in chain' http://bit.ly/1b6s4ch
    164768791820124160|Wed Feb 01 17:55:15 +0000 2012|Pfizer Recalls 1 Million Packets of Birth Control Pills:  http://on-msn.com/wSMB0u


    /Users/mshen/.virtualenvs/test/lib/python2.7/site-packages/gensim/similarities/docsim.py:528: FutureWarning: arrays to stack must be passed as a "sequence" type such as list or tuple. Support for non-sequence iterables such as generators is deprecated as of NumPy 1.16 and will raise an error in the future.
      result = numpy.hstack(shard_results)


## 11. Discussions:

- Because there are so few words in each tweet, the results may be very unstable.

- We did not take the date/time of the tweets into consideration, for simplicity purposes. However one can easily extract the date/time information and build a recommendation system which only include the recent tweets.

- We can also analyse the emphasis of each news agency on Health and build a recommendation system for the Twitter users on news agencies. 


