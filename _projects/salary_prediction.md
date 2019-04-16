---
title: "Predicting Salaries for Job Postings"
excerpt: "Using regularized linear regression and random forest to predict salaries for job postings."
collection: projects
---




_Chenxu Wen, chenxu.wen.math@gmail.com_



# Table of contents
1. [Introduction and Import](#introduction)
    1. [Data Import](#import)
    2. [Sanity Checks](#sanity)

2. [Descriptive Analysis](#descriptive)
   
3. [Predictive Model](#predictive)
    1. [Linear Regression](#linear)
    2. [Random Forest Model](#rf)
    3. [Output Export](#output)
    
4. [Conclusions and Suggestions](#conclusion)
    
## Introduction and Import <a name="introduction"></a>

In this report, I performed descriptive and predictive analysis on a data set about job postings. The goal is to understand the factors affecting job salaries and to predict future salaries.

### Data Import <a name="import"></a>

The job posting datasets are imported into Python. The first five rows of data are shown.


```python
# import libraries
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd
```


```python
# import in the training data set
train = pd.read_csv("train_features_2013-03-07.csv", low_memory=False)
# only drop rows that are all NA:
train = train.dropna(how='all')

# read the target variable - salary 
target = pd.read_csv("train_salaries_2013-03-07.csv")

# read the test data
test = pd.read_csv("test_features_2013-03-07.csv")
```


```python
# take a look at the first 5 rows of train data
train.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>jobId</th>
      <th>companyId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JOB1362684407687</td>
      <td>COMP37</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>MATH</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JOB1362684407688</td>
      <td>COMP19</td>
      <td>CEO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>WEB</td>
      <td>3</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JOB1362684407689</td>
      <td>COMP52</td>
      <td>VICE_PRESIDENT</td>
      <td>DOCTORAL</td>
      <td>PHYSICS</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JOB1362684407690</td>
      <td>COMP38</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>CHEMISTRY</td>
      <td>AUTO</td>
      <td>8</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOB1362684407691</td>
      <td>COMP7</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>PHYSICS</td>
      <td>FINANCE</td>
      <td>8</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
# take a look at the first 5 rows of test data
test.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>jobId</th>
      <th>companyId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JOB1362685407687</td>
      <td>COMP33</td>
      <td>MANAGER</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>HEALTH</td>
      <td>22</td>
      <td>73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JOB1362685407688</td>
      <td>COMP13</td>
      <td>JUNIOR</td>
      <td>NONE</td>
      <td>NONE</td>
      <td>AUTO</td>
      <td>20</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JOB1362685407689</td>
      <td>COMP10</td>
      <td>CTO</td>
      <td>MASTERS</td>
      <td>BIOLOGY</td>
      <td>HEALTH</td>
      <td>17</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JOB1362685407690</td>
      <td>COMP21</td>
      <td>MANAGER</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>OIL</td>
      <td>14</td>
      <td>96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOB1362685407691</td>
      <td>COMP36</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>BIOLOGY</td>
      <td>OIL</td>
      <td>10</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>




```python
# take a look at the first 5 rows of target data
target.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>jobId</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JOB1362684407687</td>
      <td>130</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JOB1362684407688</td>
      <td>101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JOB1362684407689</td>
      <td>137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JOB1362684407690</td>
      <td>142</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOB1362684407691</td>
      <td>163</td>
    </tr>
  </tbody>
</table>
</div>



The top 5 rows in the train, test, and target data are shown above. From the table, we can see that there are 8 columns in the data, including job ID, company ID, job type, degree required, major required, industry, years of experience required, and distance from metropolis. The outcome value is saved in the target variable.

### Sanity Checks <a name="sanity"></a>

Once the data is imported, let's perform a few sanity checks on the data, including:

1. missingness - are there missing values in the data?
2. duplicates - are there duplicates in the data?
3. observation and feature order - do observations and feature appear in the same order?
4. outliers - are there outliers in both the predictors and the outcome?
5. categorical levels - do train and test data have the same factor levels?

#### Missing Data Check


```python
# check the number of entries and data type for each variable
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1000000 entries, 0 to 999999
    Data columns (total 8 columns):
    jobId                  1000000 non-null object
    companyId              1000000 non-null object
    jobType                1000000 non-null object
    degree                 1000000 non-null object
    major                  1000000 non-null object
    industry               1000000 non-null object
    yearsExperience        1000000 non-null int64
    milesFromMetropolis    1000000 non-null int64
    dtypes: int64(2), object(6)
    memory usage: 68.7+ MB
    


```python
# check the number of entries and data type for each variable
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 8 columns):
    jobId                  1000000 non-null object
    companyId              1000000 non-null object
    jobType                1000000 non-null object
    degree                 1000000 non-null object
    major                  1000000 non-null object
    industry               1000000 non-null object
    yearsExperience        1000000 non-null int64
    milesFromMetropolis    1000000 non-null int64
    dtypes: int64(2), object(6)
    memory usage: 61.0+ MB
    


```python
# check the number of entries and data type for each variable
target.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 2 columns):
    jobId     1000000 non-null object
    salary    1000000 non-null int64
    dtypes: int64(1), object(1)
    memory usage: 15.3+ MB
    

Information on each variable in the train and test data is shown above. From the information, we can see that this is a relatively large data set with 1 million observations, with no missing data.

#### Duplicates Check  


```python
## check duplicates in train data
print('Are all observations in train unique?')
print(len(train.jobId.unique()) == train.shape[0]) 
```

    Are all observations in train unique?
    True
    


```python
## check duplicates in test data
print('Are all observations in test unique?')
print(len(test.jobId.unique()) == test.shape[0]) 
```

    Are all observations in test unique?
    True
    


```python
# now check if there's any overlap between train and test in terms of job postings
print('The number of job IDs in the test dataset that did appear in the train dataset:')
print(len ( set(test['jobId']).intersection( train['jobId'] ) ) )

print('The number of job IDs in the train dataset that did appear in the test dataset:')
print(len ( set(train['jobId']).intersection( test['jobId'] ) ) )

```

    The number of job IDs in the test dataset that did appear in the train dataset:
    0
    The number of job IDs in the train dataset that did appear in the test dataset:
    0
    

It's good news that there's no overlap between train and test, because otherwise there'd be leakage in the model training.

#### Observation and Feature Order Check
Here let's check if the observations appear in the same order in the train data and target data. If not, we'll need to sort the target to match the train dataset.


```python
# let's check if the jobID appear in the same order in train and in target
# if not, we'll need to sort the target to match the train dataset
sum(train.jobId == target.jobId) == train.shape[0]
```




    True



Now, let's check if the features appear in the same order in the train and test data. If not, we'll need to sort the test data to match the train data.


```python
sum(train.columns == test.columns) == train.shape[1]
```




    True



#### Outliers Check
Now we know that there's no overlap between train and test observations, and that the features appear in the same order in train and test, we can safely concatenate the two DataFrames together for further analysis. Otherwise, we need to use the merge the two datasets.


```python
# for the purpose of less typing, let's concat the train and test data
data = pd.concat([train, test], keys=['train','test'])
# get a quick description of numeric values in the data
data.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.000000e+06</td>
      <td>2.000000e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.199724e+01</td>
      <td>4.952784e+01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.212785e+00</td>
      <td>2.888372e+01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.000000e+00</td>
      <td>2.500000e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.200000e+01</td>
      <td>5.000000e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.800000e+01</td>
      <td>7.500000e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.400000e+01</td>
      <td>9.900000e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
# histogram of years of experience
sns.distplot(data.yearsExperience, kde= False, bins=25)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x160243890>




![png](/images/salary_prediction/output_24_1.png)



```python
# histogram of miles from metropolis
sns.distplot(data.milesFromMetropolis, kde = False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1600df690>




![png](/images/salary_prediction/output_25_1.png)


The data on **years of experience** and **mile from metropolis** seem reasonable. Both the mean and median years of experience required are about 12 years, with a min of 0 and a max of 24. Both the mean and the median number of miles from metropolis are about 50 miles, with a min of 0 and a max of 99. The values are almost evenly distributed across the range for both variables. Thus there seem to be no apparent outliers.


```python
target.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>116.061818</td>
    </tr>
    <tr>
      <th>std</th>
      <td>38.717936</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>88.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>114.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>141.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>301.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# count the number of 0s
print('The number of 0s in salary: '+ str( sum(target.salary == 0)))
print('The minimum non-zero salary: ' + str(min(target.salary[target.salary != 0])))
```

    The number of 0s in salary: 5
    The minimum non-zero salary: 17
    

We can see that the target, salary, is relatively normally distributed, a bit positively skewed. Except for 5 zero salary, the rest are well above 0. These 5 may be outliers. Let's remove them for now. 


```python
# remove zero-salary observations
zero_index = target[target.salary == 0].index
train.drop(zero_index, inplace=True)
target.drop(zero_index, inplace=True)

# update data
data = pd.concat([train, test], keys=['train','test'])
# describe the new data
target.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>999995.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>116.062398</td>
    </tr>
    <tr>
      <th>std</th>
      <td>38.717163</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>88.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>114.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>141.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>301.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.select_dtypes(include=['object']).describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>jobId</th>
      <th>companyId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1999995</td>
      <td>1999995</td>
      <td>1999995</td>
      <td>1999995</td>
      <td>1999995</td>
      <td>1999995</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>1999995</td>
      <td>63</td>
      <td>8</td>
      <td>5</td>
      <td>9</td>
      <td>7</td>
    </tr>
    <tr>
      <th>top</th>
      <td>JOB1362685306512</td>
      <td>COMP39</td>
      <td>SENIOR</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>WEB</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>32197</td>
      <td>251088</td>
      <td>475230</td>
      <td>1066421</td>
      <td>286217</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot the distribution of the target
sns.distplot(target.salary)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x162acec10>




![png](/images/salary_prediction/output_32_1.png)


The description of categorical values shows that we are working with a fairly clean dataset. Of all the jobs posts included, there are only 63 companies, 8 job types, 9 majors, and 7 industries involoved. The jobs are fairly evenly distributed across companies, industry, degree, and job type. The only exception is major, with a lot of jobs not specifying a required major. The outcome variable is relatively normally distributed, slightly positively skewed. Since the skewness is only slight, we'll leave the outcome variable as it.

#### Categorical Levels Check
Here let's check if the categorical variables share the same levels in train and test data. If there are category levels in train but missing in test or vice versa, then we'd need to add a dummy level to test (or train) data to ensure consistent train and test data for the model.


```python
# get categorical variable names
cat_vars = train.select_dtypes(include=['object']).columns
for i in cat_vars:
    print('The variable ' + i + ' has the same levels in the train and the test dataset:')
    # check if the intersection of levels is the same set as in train data
    print(len ( set(train[i]).intersection( test[i] ) ) == len( set(train[i])) )
```

    The variable jobId has the same levels in the train and the test dataset:
    False
    The variable companyId has the same levels in the train and the test dataset:
    True
    The variable jobType has the same levels in the train and the test dataset:
    True
    The variable degree has the same levels in the train and the test dataset:
    True
    The variable major has the same levels in the train and the test dataset:
    True
    The variable industry has the same levels in the train and the test dataset:
    True
    

The output above showed that the levels of all categorical variables have consistent levels in the train and the test data, except jobId (which should be the case, in fact we expected no overlap between the two). This is good since now we don't need to worry about adding extra dummy levels for either train or test data.

#### Summary
In summary, we observed the following from the sanity checks:
1. missing data - there are no missing values in the data.
2. duplicates - there are no duplicates in the data.
3. observation and feature order - the observations and features appear in the same order in the train, test, and target data.
4. outliers - there are no apparent outliers in the data, 5 observations with a salary of 0 were removed. We'll also look for outliders during the exploratory analysis.
5. categorical levels - the train and test data have the same factor levels.

### Descriptive Analysis <a name="descriptive"></a>

In this section, let's analyze the correlation among the predictors and also between the predictors and the outcome. One nice way to present the correlations is to use a correlation matrix, such as the one that's provided by the *pairplot* function in *seaborn*. The challenge is that this function does not support categorical variables, but we have quite a few categorical predictors. Let's map the categorical levels to integers by converting them to factors.


```python
# first, for visualization purpose, create numeric representation for categorical variables 
train_cp = train.copy()
for i in train_cp.select_dtypes(include=['object']).columns.values:
    train_cp[i] = pd.factorize(train_cp[i])[0]
# add in the outcome var - salary 
train_cp['salary'] = target.salary
# remove job Id
train_cp.drop(['jobId'],axis=1, inplace=True)
```


```python
# second, Create a pair grid instance
# since the plot with all the train data took very long to generate
# let's plot on a random subset of the data
grid = sns.PairGrid(data= train_cp.sample(frac= 0.1) )

# Map the plots to the locations
grid = grid.map_offdiag(sns.regplot, marker = '+', x_jitter=.1, y_jitter=0.1)
grid = grid.map_diag(sns.distplot, kde = False)
```


![png](/images/salary_prediction/output_40_0.png)


From the plots above, we can see that the data is fairly evenly distributed across the category levels for both categorical and numeric features. The only exception is the *major* feature, where the *NONE* level is very frequent. 

As to the relationship between the outcome and the features, we have the following observations:

1. In general, salary is higher for more experience, and for positions closer to metropolis. But there is a wide range of salary for each experience level and distance level. Also there are some notable outliers way above the average for each level.
2. There are differences in salary across degrees, majors, industries, and job types, but not across companies. Similar to the relationship with numeric features, there is a wide range of salary for each level of the categorical features, maybe except for the JANITOR job type. Also there are some notable outliers way above the average for almost every level.

In the interest of space, I will not show individual bar plots, but we can learn a few of observations that are quite intuitive:
1. There is no apparent association between degree and salary, except that no-specific major means less money.
2. Job type matters. The C-suite position makes way more than the rest job types.
3. Finance, oil, and web are some of the highest-paid industries, while service and education are paid the least.

## Predictive Model <a name="predictive"></a>

In this section, let's predict salary using penalized linear models as a starting point. 

We have 5 categorical features, and a goal to not only build a model, but also to estimate the feature contributions. With scikit-learn, we'd need to one-hot encode these features, which means that the contribution information for each categorical feature would be dispersed into individual variables for each level. To circumvent this issue, let's use the h2o package, which encodes categorical features as factors and retains the level contributions to the single categorical feature.


```python
import h2o
h2o.init()
```

    Checking whether there is an H2O instance running at http://localhost:54321. connected.
    


<div style="overflow:auto"><table style="width:50%"><tr><td>H2O cluster uptime:</td>
<td>1 day 2 hours 16 mins</td></tr>
<tr><td>H2O cluster timezone:</td>
<td>America/Chicago</td></tr>
<tr><td>H2O data parsing timezone:</td>
<td>UTC</td></tr>
<tr><td>H2O cluster version:</td>
<td>3.22.1.6</td></tr>
<tr><td>H2O cluster version age:</td>
<td>22 days </td></tr>
<tr><td>H2O cluster name:</td>
<td>H2O_from_python_mshen_n00a2o</td></tr>
<tr><td>H2O cluster total nodes:</td>
<td>1</td></tr>
<tr><td>H2O cluster free memory:</td>
<td>2.624 Gb</td></tr>
<tr><td>H2O cluster total cores:</td>
<td>8</td></tr>
<tr><td>H2O cluster allowed cores:</td>
<td>8</td></tr>
<tr><td>H2O cluster status:</td>
<td>locked, healthy</td></tr>
<tr><td>H2O connection url:</td>
<td>http://localhost:54321</td></tr>
<tr><td>H2O connection proxy:</td>
<td>None</td></tr>
<tr><td>H2O internal security:</td>
<td>False</td></tr>
<tr><td>H2O API Extensions:</td>
<td>Amazon S3, XGBoost, Algos, AutoML, Core V3, Core V4</td></tr>
<tr><td>Python version:</td>
<td>2.7.14 final</td></tr></table></div>



```python
# convert pandas DataFrame to H2OFrame 
dat = train.merge(target, on = 'jobId')
df = h2o.H2OFrame(dat)
# split the data into train and test
splits = df.split_frame(ratios=[0.8], seed=42)
dtrain = splits[0]
dtest = splits[1]
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%
    


```python
y = 'salary'
x = list(df.columns)
x.remove(y)  #remove the response
x.remove('jobId') 
# The companyId feature doesn't seem to relate with salary, which comfirms our intuition. 
# Also in production, there's always the possibility of predicting on unseen companies before.
# Therefore, let's remove the companyId feature for now
x.remove('companyId') 
```

### Linear Model <a name="linear"></a>


```python
## grid search for best params with cross validation
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
# grid search criteria, in the interest of time, let's set an upper limit to the runtime and max models
gs_criteria = {'strategy': 'RandomDiscrete', 
              'max_models': 100,
              'max_runtime_secs': 6000, 
              'stopping_metric': 'mae',
              'stopping_tolerance': 3,
              'seed': 1234,
              'stopping_rounds': 15}

# linear model parameter space
lm_params = {'alpha': [0.1, 0.5, 0.9], 
             'lambda': [1e-4, 1e-2, 1e0, 1e2, 1e4]}

lm_gs = H2OGridSearch(model = H2OGeneralizedLinearEstimator(nfolds = 5), 
                      hyper_params = lm_params,
                     search_criteria = gs_criteria)
lm_gs.train(x=x,y=y, training_frame=dtrain)
# display model IDs, hyperparameters, and MSE
# lm_gs.show()
```


```python
# Get the grid results, sorted by validation mae
lm_gs_perf = lm_gs.get_grid(sort_by='mae', decreasing=True)
lm_gs_perf

# extract the top lm model, chosen by validation mae
lm_best = lm_gs.models[0]

# Now let's evaluate the model performance on a test set
# so we get an honest estimate of top model performance
lm_best_perf = lm_best.model_performance(dtest)

print('The best test MAE is %.2f' % lm_best_perf.mae())
print('The best test RMSE is %.2f' % lm_best_perf.rmse())
print('The best test R2 is %.2f' % lm_best_perf.r2())
```

    The best test MAE is 15.85
    The best test RMSE is 19.62
    The best test R2 is 0.75
    


```python
# get coef
coef = lm_best.coef()
# plot feature importance
coef_df = pd.DataFrame({'feature':coef.keys(), 'coefficient':coef.values()})

coef_df.reindex(coef_df.coefficient.abs().sort_values(ascending = False).index)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficient</th>
      <th>feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>115.651554</td>
      <td>Intercept</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-34.583555</td>
      <td>jobType.JANITOR</td>
    </tr>
    <tr>
      <th>8</th>
      <td>27.669385</td>
      <td>jobType.CEO</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-22.043900</td>
      <td>jobType.JUNIOR</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17.929292</td>
      <td>jobType.CFO</td>
    </tr>
    <tr>
      <th>28</th>
      <td>17.925786</td>
      <td>jobType.CTO</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-16.191104</td>
      <td>industry.EDUCATION</td>
    </tr>
    <tr>
      <th>24</th>
      <td>15.031780</td>
      <td>industry.OIL</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14.869542</td>
      <td>industry.FINANCE</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-12.049542</td>
      <td>jobType.SENIOR</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-11.159391</td>
      <td>industry.SERVICE</td>
    </tr>
    <tr>
      <th>25</th>
      <td>10.121587</td>
      <td>degree.DOCTORAL</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-9.430685</td>
      <td>degree.NONE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.327428</td>
      <td>major.ENGINEERING</td>
    </tr>
    <tr>
      <th>30</th>
      <td>7.794749</td>
      <td>jobType.VICE_PRESIDENT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-7.222763</td>
      <td>major.NONE</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-6.227290</td>
      <td>industry.AUTO</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-5.954474</td>
      <td>major.LITERATURE</td>
    </tr>
    <tr>
      <th>29</th>
      <td>5.882962</td>
      <td>industry.WEB</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-5.747840</td>
      <td>degree.HIGH_SCHOOL</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.334937</td>
      <td>major.BUSINESS</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5.005238</td>
      <td>degree.MASTERS</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.846280</td>
      <td>major.MATH</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-2.329465</td>
      <td>major.BIOLOGY</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-2.145721</td>
      <td>jobType.MANAGER</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2.009992</td>
      <td>yearsExperience</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.619933</td>
      <td>major.COMPSCI</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.268621</td>
      <td>major.CHEMISTRY</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-0.399243</td>
      <td>milesFromMetropolis</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.000000</td>
      <td>major.PHYSICS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>degree.BACHELORS</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>industry.HEALTH</td>
    </tr>
  </tbody>
</table>
</div>



### Random Forest Model <a name="rf"></a>


```python
# now let's use the random forest algorithm to see if we can improve the performance
# import 
from h2o.estimators.random_forest import H2ORandomForestEstimator
# rf parameters to be tuned
rf_params = {"ntrees" : [50,100], 
              "max_depth" : [5,10], 
              "sample_rate" : [0.3, 0.7], 
              "mtries":[-1, 3, 7],
              "min_rows" : [3,5] }
# parameters for rf that don't need to be tuned
rf_p = {"seed": 42 , 
         "score_tree_interval": 3,
         "nfolds": 5}
# set up the grid
rf_gs = H2OGridSearch(model = H2ORandomForestEstimator(**rf_p), 
                       hyper_params = rf_params,
                       search_criteria = gs_criteria)
# train the grid
rf_gs.train(x=x,y=y, training_frame=df)

# display model IDs, hyperparameters, and MSE
# rf_gs.show()
```


```python
# Get the grid results, sorted by validation mae
rf_gs_perf = rf_gs.get_grid(sort_by='mae', decreasing=True)
rf_gs_perf

# extract the top lm model, chosen by validation mae
rf_best = rf_gs.models[0]

# Now let's evaluate the model performance on a test set
# so we get an honest estimate of top model performance
rf_best_perf = rf_best.model_performance(dtest)

print('The best test MAE is %.2f' % rf_best_perf.mae())
print('The best test RMSE is %.2f' % rf_best_perf.rmse())
print('The best test R2 is %.2f' % rf_best_perf.r2())
```

    The best test MAE is 15.69
    The best test RMSE is 19.44
    The best test R2 is 0.75
    


```python
# plot feature importance
rf_best.varimp_plot()
```


![png](/images/salary_prediction/output_53_0.png)


### Output Export <a name="output"></a>


```python
# write output
# convert test data to h2o frame
test_h2o = h2o.H2OFrame(test)
# determine the best model based on mae and use it for prediction
if rf_best_perf.mae() < lm_best_perf.mae():
    pred = rf_best.predict(test_h2o)
else:
    pred = lm_best.predict(test_h2o)
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%
    drf prediction progress: |████████████████████████████████████████████████| 100%
    


```python
# examine the basic stats of the salary predictions
pred.describe()
```

    Rows:1000000
    Cols:1
    
    
    


<table>
<thead>
<tr><th>       </th><th>predict      </th></tr>
</thead>
<tbody>
<tr><td>type   </td><td>real         </td></tr>
<tr><td>mins   </td><td>29.2754525948</td></tr>
<tr><td>mean   </td><td>116.050909815</td></tr>
<tr><td>maxs   </td><td>216.278178101</td></tr>
<tr><td>sigma  </td><td>32.3859060281</td></tr>
<tr><td>zeros  </td><td>0            </td></tr>
<tr><td>missing</td><td>0            </td></tr>
<tr><td>0      </td><td>106.993746109</td></tr>
<tr><td>1      </td><td>100.914067307</td></tr>
<tr><td>2      </td><td>170.002967072</td></tr>
<tr><td>3      </td><td>108.953849945</td></tr>
<tr><td>4      </td><td>111.065993118</td></tr>
<tr><td>5      </td><td>149.859150085</td></tr>
<tr><td>6      </td><td>96.3085207367</td></tr>
<tr><td>7      </td><td>120.059133606</td></tr>
<tr><td>8      </td><td>104.54789505 </td></tr>
<tr><td>9      </td><td>102.726793747</td></tr>
</tbody>
</table>



```python
# concat jobId and salary prediction
out = test_h2o['jobId'].concat(pred)
# change names
out.set_names(list(target.columns))
# write the test output to a csv file
h2o.export_file(out, path = "test_salaries.csv")    
```

    Export File progress: |███████████████████████████████████████████████████| 100%
    

## Conclusions and Suggestions<a name="conclusion"></a>

1. In data exploration, we can observe that this dataset is quite clean and evenly distributed across levels, with only 5 0-salary observations.
2. To predict the salary outcome, I built both a regularized linear model and a random forest model using the h2o package. The random forest model performed slightly better but took much longer to tune.
3. Future directions - prediction explanations: If the rationale behind individual predictions is of interest, we can use the LIME method. LIME stands for Local Interpretable Model-Agnostic Explanations. Essentially, we attempt to understand why a model would make a certain prediction by building a locally-weighted linear model to approximate local predictions. Another viable approach is the SHAP values, which uses game theory to explain the contributions of each features to both individual and the overall predictions.
