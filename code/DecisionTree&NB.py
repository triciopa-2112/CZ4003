
# coding: utf-8

# In[2]:

__author__ = 'Tram Anh'

import sys
import pandas as pd
import sklearn as sk
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Binarizer
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[3]:

# Import csv data
raw_data = pd.read_csv('OnlineNewsPopularity_wLabels_deleteNoise.csv').iloc[:, 1:]      # read in csv, omit the first column of url
raw_data = raw_data.iloc[:, :-1] 
news_data = raw_data.iloc[:, :-1]      # Take up to the second last column
news_labels = raw_data.iloc[:, -1]      # Take shares column for labels

# Binarize
print '\nBinary Threshold:'
binary_threshold = np.median(raw_data[' shares'])
news_data = news_data.drop(' n_non_stop_words', 1)
print binary_threshold
binarizer = Binarizer(threshold=binary_threshold)
y_binary = binarizer.transform(news_labels).transpose().ravel() 


# In[ ]:

# Discretize


# In[25]:

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
print 'Decision Tree Classifier Accuracy Rate'
tree_score = cross_val_score(tree, news_data, y_binary, cv=10)
np.mean(tree_score)


# In[24]:

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
print 'Naive Bayes Classifier Accuracy Rate'
cv_score = cross_val_score(gnb, news_data, y_binary, cv=10)
np.mean(cv_score)

