
# coding: utf-8

# In[ ]:

__author__ = 'Lasse'

import sys
import pandas as pd
import statsmodels.api as sm 
import sklearn as sk
from sklearn import linear_model
from sklearn.metrics import mean_Csbsolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Binarizer
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[ ]:

# Import csv data
raw_data = pd.read_csv('OnlineNewsPopularity_wLabels_deleteNoise.csv').iloc[:, 1:]     # read in csv, omit the first column of url
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

# PREPROCESS SVM DATA
from sklearn import preprocessing

#preprocess X
scalerX = preprocessing.StandardScaler().fit(news_data)
X = scalerX.transform(news_data) 


# In[ ]:

from sklearn import svm

y=y_binary

# Run four different types of SVM for a given value of C
def cv_mean_std_SVM(C, cv):
    lin = svm.SVC(C=C, kernel='linear', max_iter=10000)
    poly_2 = svm.SVC(C=C, kernel='poly', degree=2, max_iter=10000)
    poly_3 = svm.SVC(C=C, kernel='poly', degree=3, max_iter=10000)
    rbf = svm.SVC(C=C, kernel='rbf', max_iter=10000)
    cv_lin = cross_val_score(lin, X, y_binary, cv=cv)#, dual=False)
    print 'finished computing linear cv'
    cv_poly_2 = cross_val_score(poly_2, X, y_binary, cv=cv)#, dual=False)
    print 'finished computing polydeg2 cv'
    cv_poly_3 = cross_val_score(poly_3, X, y_binary, cv=cv)#, dual=False)
    print 'finished computing polydeg3 cv'
    cv_rbf = cross_val_score(rbf, X, y_binary, cv=cv)#, dual=False)
    print 'finished computing rbf cv'
    return np.mean(cv_lin), np.std(cv_lin), np.mean(cv_poly_2), np.std(cv_poly_2), np.mean(cv_poly_3), np.std(cv_poly_3), np.mean(cv_rbf), np.std(cv_rbf)


# In[19]:

# Create an array of different values of C to try.
n_Cs = 15
n = n_Cs
Cs = np.empty(n)
cv = np.empty(n)

for i in range (0, n_Cs):
        Cs[i] = np.exp(i)/1000
ln_Cs = np.log(Cs) # this is what we plot on the axis to keep it readible
print 'Cs array:'
print Cs


# In[27]:

# Cycle through all different values of 
def cv_mean_std_arrays(X,y, Cs, n_Cs, cv=2):
    cv_lin_means, cv_lin_stds, cv_pol2_means, cv_pol2_stds, cv_pol3_means, cv_pol3_stds, cv_means_rbf, cv_stds_rbf = np.empty(n_Cs), np.empty(n_Cs), np.empty(n_Cs), np.empty(n_Cs), np.empty(n_Cs), np.empty(n_Cs), np.empty(n_Cs), np.empty(n_Cs)
    for i in range  (0, n_Cs):
        cv_lin_means[i], cv_lin_stds[i], cv_pol2_means[i], cv_pol2_stds[i], cv_pol3_means[i], cv_pol3_stds[i], cv_means_rbf, cv_stds_rbf = cv_mean_std_SVM(Cs[i], cv)
    return cv_lin_means, cv_lin_stds, cv_pol2_means, cv_pol2_stds, cv_pol3_means, cv_pol3_stds, cv_means_rbf, cv_stds_rbf


# In[28]:

# Save cv's to disc.
cv_lin_means, cv_lin_stds, cv_pol2_means, cv_pol2_stds, cv_pol3_means, cv_pol3_stds, cv_means_rbf, cv_stds_rbf = cv_mean_std_arrays(X,y, Cs, n_Cs, cv=2) #run algo on alphas

print cv_lin_means, cv_lin_stds, cv_pol2_means, cv_pol2_stds, cv_pol3_means, cv_pol3_stds


# Plot obtained output
import matplotlib
import matplotlib.pyplot as plt

plt.plot(Cs, cv_lin_means)
plt.plot(Cs, cv_pol2_means)
plt.plot(Cs, cv_pol3_means)
plt.plot(Cs, cv_means_rbf)
plt.legend(['Linear Kernel', 'Pol. deg. 2', 'Pol. deg. 3', 'RBF Kernel'], loc='upper right')
plt.ylabel('Classification Score')
plt.show()


# In[ ]:



