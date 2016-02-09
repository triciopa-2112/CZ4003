
# coding: utf-8

# In[25]:

get_ipython().magic(u'pylab inline')
import pandas as pd
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Binarizer


# In[33]:

# Import csv data
raw_data = pd.read_csv('OnlineNewsPopularity_wLabels_deleteNoise.csv').iloc[:, 1:]      # read in csv, omit the first column of url
raw_data = raw_data.iloc[:, :-1] 
news_data = raw_data.iloc[:, :-1]      # Take up to the second last column
news_labels = raw_data.iloc[:, -1]      # Take shares column for labels
# print raw_data[' shares'].describe()
news_data = news_data.drop(' n_non_stop_words', 1)
print '\nBinary Threshold:'
binary_threshold = np.median(raw_data[' shares'])
print binary_threshold
binarizer = Binarizer(threshold=binary_threshold)
y_binary = binarizer.transform(news_labels).transpose().ravel() 


# In[35]:

# run knn for a particular value of k, cv=cross validations
def knn_cv_mean_and_std(X, y, k, cv=20, max_iter=1000): 
    print 'computing cv for k=%d' %k
    knn = KNeighborsClassifier(n_neighbors=k) # arbitrary k
    cv = cross_val_score(knn, X, y, cv=cv)
    print 'successfully computed cv for k=%d' %k
    return np.mean(cv), np.std(cv)
knn_cv_mean_and_std(news_data, y_binary, k=30)


# In[30]:

# run knn for a array of values of k, cv=cross validations
def cv_mean_std_array(X, y, alphas, ks, n_a, n_k, cv=20):
    n = n_alphas*n_ks
    cv_mean = np.empty(n)
    cv_std = np.empty(n)
    regressors = pd.DataFrame()

    binarizer = Binarizer(threshold=1400)
    y_binary = binarizer.transform(y).transpose().ravel() 

    itt_counter = 0
    print 'size n_a: %d n_k: %d' %(n_a, n_k)
    for i in range (0, n_a):
    	print 'reg. column : %d' %(i*n_k)
    	temp_string = 'alpha=%f' %alphas[i*n_k]
    	print temp_string
    	print regressors.shape
    	df_temp = pd.DataFrame()
        print 'computing for alpha = %f' %(alphas[n_ks*i])
        X_lasso, df_temp[temp_string] = df_Lasso(X, y, alphas[i*n_k])
        regressors = pd.concat([regressors,df_temp], ignore_index=True, axis=1)
        for j in range(0, n_k):
            print 'i:%d, j:%d' %(i, j)
            print 'computing for alpha = %f and k = %f' %(alphas[n_ks*i+j], ks[n_ks*i+j])
            print 'X_lasso shape:' 
            print X_lasso.shape
            cv_mean[n_ks*i+j], cv_std[n_ks*i+j] = knn_cv_mean_and_std(X_lasso, y_binary, alphas[n_ks*i+j], ks[n_ks*i+j], cv=cv)
            itt_counter = itt_counter + 1
            print 'completed %dth iteration of knn cv mean:%f std:%f, at pos:%d' % (itt_counter, cv_mean[n_ks*i+j], cv_std[n_ks*i+j], n_ks*i+j)
    return cv_mean, cv_std, regressors



