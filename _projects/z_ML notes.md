---
title: "Machine Learning with Python"
excerpt: "Some code snippets for common machine learning practices."
collection: projects
---

## machine learning process  
1. build a data set  

2. explorative analysis (histograms/density plot for distributions, scatter plot (numeric-numeric), bar plot (categorical-numeric or categorical-categorical-proportions), box plot (numeric-categorical) for relationships

3. preprocess for duplicates, outliers, missing value, standardize, 

4. choose a model and a metric

5. train-test split, 20% hold out dataset, 5 fold cross-validation

6. train, parameter tuning

7. evaluation using test set

8. prediction, interpretation, and result packaging

## Tips:
remember to set random_state, to 42, preferably.

```
# check NA

dat.isnull().sum()

# correlation matrix

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)}) # figure size
dat_p = dat.copy()
for i in dat_p.select_dtypes( include = 'object').columns.values:
	dat_p[ i ] = dat_p[ i ].factorize()[1] 
sns.pairplot( dat_p, vars = [], hue = 'outcome_category', size = 3)
# Plot colored by continent for years 2000-2007
sns.pairplot(df[df['year'] >= 2000], 
             vars = ['life_exp', 'log_pop', 'log_gdp_per_cap'], 
             hue = 'continent', diag_kind = 'kde', 
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 4);
# Title 
plt.suptitle('Pair Plot of Socioeconomic Data for 2000-2007', 
             size = 28);
```

## ways to visualize text
1. plot top words

2. plot length by department

### plot top words
```
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['Review Text'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df2.groupby('ReviewText').sum()['count'].sort_values(ascending=False)

# show top words that distinguish resume from background
corpus = st.CorpusFromPandas(df, category_col='Department Name', text_col='Review Text', nlp=nlp).build()
print(list(corpus.get_scaled_f_scores_vs_background().index[:10]))

# show top words associated with a category
term_freq_df = corpus.get_term_freq_df()
term_freq_df['Tops Score'] = corpus.get_scaled_f_scores('Tops')
pprint(list(term_freq_df.sort_values(by='Tops Score', ascending=False).index[:10]))
```

## train test split
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)
```

## column transformer pipe:
```
## Standardize features by removing the mean and scaling to unit variance
X = data.loc[:, 'X0':'X41']
num_cols = X.select_dtypes(exclude = 'object').columns.values
cat_cols = X.select_dtypes(include = 'object').columns.values

## Numeric:
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
num_pipe = Pipeline([ ('impute', SimpleImputer(strategy='median')), 
 ('std',StandardScaler()) ])
 
## Categorical:
cat_pipe = Pipeline([  ('impute', SimpleImputer(strategy='constant',
                   fill_value='MISSING')),
('ohe', OneHotEncoder(sparse=False,
                    handle_unknown='ignore')) ])
ct = ColumnTransformer( [ ('num', num_pipe, num_cols), 
('cat', cat_pipe, cat_cols) ])
```

## nlp

```
drop na and empty:
blanks = []  # start with an empty list

for i,lb,rv in df.itertuples():  # iterate over the DataFrame
    if type(rv)==str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list
df.drop(blanks, inplace=True)
df.dropna(inplace=True)

```

## tfidf vectorize
```
from sklearn.feature_extraction.text import TfidfVectorizer
add this to the pipe, then use multinomial naive bayes: ('tfidf', TfidfVectorizer(sublinear_tf = True, stop_words = 'english', ngram_range=(1, 2)))
from sklearn.naive_bayes import MultinomialNB
('mnb', MultinomialNB() )
```

## multiclass
```
add this, or just use random forest: ('clf', OneVsRestClassifier(LinearSVC()))
```

## multilabel
```
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

X_train = np.array(["new york is a hell of a town",
                    "new york was originally dutch"])
y_train_text = [["new york"],["new york"]]

X_test = np.array(['nice day in nyc',
                   'hello welcome to new york. enjoy it here and london too'])
target_names = ['New York', 'London']

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train_text)

classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)

for item, labels in zip(X_test, all_labels):
    print('{0} => {1}'.format(item, ', '.join(labels)))
```

## machine pipe:
```
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers =[('cat', cat_pipe, cat_cols),
                    ('num', num_pipe, num_cols)] )
pl = ct.named_transformers_['cat']
ohe = pl.named_steps['ohe']
ohe.get_feature_names()

ml_pipe = Pipeline([('transform', ct), ('ridge', Ridge())])
from sklearn.model_selection import GridSearchCV
param_grid = {
    'transform__num__si__strategy': ['mean', 'median'],  ## notice the two underscores __
    'ridge__alpha': [.001, 0.1, 1.0, 5, 10, 50, 100, 1000],
    }
gs = GridSearchCV(ml_pipe, param_grid, cv=kf, scoring = 'neg_mean_absolute_error')
gs.fit(train, y)
gs.best_params_
{'ridge__alpha': 10, 'transform__num__si__strategy': 'median'}
gs.best_score_

```

## Elastic Net

```
from sklearn.linear_model import ElasticNet
ml_pipe = Pipeline([('transform', ct), ('enet', ElasticNet())])
params = {"enet__max_iter": [1, 5, 10],
                      "enet__alpha": [0.0001, 1, 10, 100],
                      "enet__l1_ratio": np.arange(0.0, 1.0, 0.5)}
gs = GridSearchCV( ml_pipe, param_grid = params, scoring='r2', cv=5)
gs.fit(X_train, X_test)
Logistic reg:
from sklearn.linear_model import LogisticRegression
ml_pipe = Pipeline( [('transform', ct), ('logit', LogisticRegression())])
params = { 'penalty = ['l1', 'l2'],
'C' = np.logspace(0, 4, 10)}
GridSearchCV( ml_pipe, param_grid = params, scoring = 'roc_auc', cv = 5)
```

## Random Forest:
max_features: the maximum number of features Random Forest is allowed to try in individual tree
n_estimators: the number of trees you want to build
min_sample_split: The minimum number of samples required to split an internal node
min_samples_leaf : The minimum number of samples required to be at a leaf node. 

## Regressor:
```
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestClassifier(n_estimators=100, criterion = 'mae')

param_grid = {"max_depth": [3, 10],
              "max_features": [1, 3, 10],
              "min_sample_leaf": [25, 50] }

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
Classifier:
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, class_weight = 'balanced', random_state = 42)

param_grid = {"max_depth": [3, 10],
              "max_features": [1, 3, 10],
              "min_sample_leaf": [25, 50] }

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, scoring = 'roc_auc', cv=5)
Test:
y_pred = fit.predict(X_test)

from sklearn.metrics import r2_score
r2_score( y_test, y_pred)

clf = gs.best_estimator_named_steps['eln']
clf.coef_

from sklearn import metrics
metrics.classification_report(y_test, y_pred)
```

## draw cv results plot or draw final pred report
```
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Product.values, yticklabels=category_id_df.Product.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

## importance:
```
rf = fit.best_estimator_names_steps['rf']
rf.feature_importance_
```

## stem and tfidf
```
import nltk
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

path = '/opt/datacourse/data/parts'
token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        shakes = open(file_path, 'r')
        text = shakes.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        token_dict[file] = no_punctuation
        
#this can take some time
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())

feature_names = tfidf.get_feature_names()
for col in response.nonzero()[1]:
    print feature_names[col], ' - ', response[0, col]
```

## gensim word2vec
```
gensim word2vec
nces = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)
```
get tfidf weighted word vector
```
import spacy
nlp  = spacy.load('en_core_web_md')

# keep only words that are not space, stop words, punctuation or number
def keep_token(t):
    return (t.is_alpha and 
            not (t.is_space or t.is_punct or 
                 t.is_stop or t.like_num))
# keep lemma
def lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if keep_token(t)]

docs = [lemmatize_doc(nlp(doc)) for doc in news_train.data]

# get docs dictionary
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full

docs_dict = Dictionary(docs)
docs_dict.filter_extremes(no_above=0.05)
docs_dict.compactify()

# get tfidf, a matrix with n rows (docs) and m columns (TF-IDF terms)
import numpy as np

docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
docs_tfidf  = model_tfidf[docs_corpus]
docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])

# get word vectors
tfidf_emb_vecs = np.vstack([nlp(docs_dict[i]).vector for i in range(len(docs_dict))])

# get tfidf weighted sum of word vectors for each document
docs_emb = np.dot(docs_vecs, tfidf_emb_vecs) 

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

get_recommendations('The Dark Knight Rises')

# plot with TSNE
from sklearn.decomposition import PCA
docs_pca = PCA(n_components=8).fit_transform(docs_emb)
# and then use t-sne to project the vectors to 2D.

from sklearn import manifold

tsne = manifold.TSNE()
viz = tsne.fit_transform(docs_pca)
Plotting with matplotlib.

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.margins(0.05) 

zero_indices = np.where(news_train.target == 0)[0]
one_indices = np.where(news_train.target == 1)[0]

ax.plot(viz[zero_indices,0], viz[zero_indices,1], marker='o', linestyle='', 
        ms=8, alpha=0.3, label=news_train.target_names[0])
ax.plot(viz[one_indices,0], viz[one_indices,1], marker='o', linestyle='', 
        ms=8, alpha=0.3, label=news_train.target_names[1])
ax.legend()

plt.show()
```


