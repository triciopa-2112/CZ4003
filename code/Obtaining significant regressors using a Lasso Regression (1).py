
# coding: utf-8

# Lasso is variation of OLS which aims to minimize the following expression:
# 
#                         (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
#                         
# Note how the first term is the OLS term, the second being the L_1 norm of the estimated regression coefficients.
# The L_1 norm (absolute value) forms a hypercube which punishes the model for having too many non-zero coefficients.
# 
# The higher the value of alpha, the larger the punishment and the more regression coefficient estimates will be forced to 0.
# 
# The value of alpha is user set. Usually, the optimal alpha value is determined using either some information criteria (AIC / BIC) or some kind of error (classification score, R^2, MAE).
# 
# In this case, we are not so much concerned with an 'optimal' value for alpha, we just want to obtain optimal leanness for our subsequent technique. Hence, we are interested in the classification error of the technique used AFTER applying Lasso and obtaining the right regression matrix. Below is an example in which I use Lasso to obtain a smaller regression matrix, and then k-NN to classify.

# In[1]:

get_ipython().magic(u'pylab inline')
import pandas as pd
import statsmodels.api as sm 
import sklearn as sk
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np


# In[3]:

# Import csv data
raw_data = pd.read_csv('OnlineNewsPopularity.csv').iloc[:, 1:]      # read in csv, omit the first column of url

news_data = raw_data.iloc[:, :-1]      # Take up to the second last column
news_labels = raw_data.iloc[:, -1]      # Take shares column for labels
# print raw_data[' shares'].describe()
print '\nBinary Threshold:'
binary_threshold = np.median(raw_data[' shares']) # Set the classification threshold to median shares
print binary_threshold 


# Below, I create a bunch of nonlinear regressors out of the linear regressors. This is necessary for this kind of procedure. The decisions of which regressor variations to create is done based on a combination of intuition, logical reasoning and plain trial and error.
# 
# The end result is a few polynomial regressors (based on which ones were found to be significant in regular OLS), a few interactions between dummies (not necessarily significant) and interactions between continuous (significant) regressors and dummies.
# 
# If we create some regressors here which are highly insignificant and create OLS problems (violating model assumptions), it is not a problem as Lasso will zero the coefficients before you can say OLS.

# In[41]:

# Create variations for regressors for which these polynomial variations are significant in OLS.
news_data[' timedelta_2'] = news_data[' timedelta']**2
news_data[' timedelta_3'] = news_data[' timedelta']**3
news_data[' ln_timedelta'] = np.log(news_data[' timedelta'])

news_data[' n_tokens_title_2'] = news_data[' n_tokens_title']**2
news_data[' n_tokens_content_2'] = news_data[' n_tokens_content']**2
# news_data[' ln_n_tokens_content'] = np.log(news_data[' n_tokens_content']) # has some 0 values resulting in -inf!

# Generate an interaction variable for every specific newstopic: data_channel_is_XXX * is_weekend
# Note that there is no other category, but this is not required as this is the default case (intercept)
for column in news_data.columns:
    if column.startswith(' data_channel_is_'): 
        interaction_name = ' is_weekend' + ' *' + column
        news_data[interaction_name] = (news_data[column] * news_data[' is_weekend'])

# weekend interaction between other factors
arr = [' n_tokens_title', ' n_tokens_content', ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos', ' num_keywords']
for att in arr:
     interaction_name = att + ' * is_weekend'
     news_data[interaction_name] = (news_data[att] * news_data[' is_weekend'])

# Code to create interactions between data_channel and the array of other attributes
for column in news_data.columns:
    if column.startswith(' data_channel_is_'):  
        for att in arr:
            interaction_name = att + ' *' + column
            news_data[interaction_name] = (news_data[att] * news_data[column])
news_data.shape


# Now that we have the Lasso estimates, we can multiply the original data with the Lasso estimates vector (piecewise). We then remove all the zero columns to obtain a lean data frame with only Lasso significant regressors.
# 
# Note I: Here, the value you choose for alpha determines the leanness of the matrix. Higher means leaner.
#            
# Note II: The fit is done on the whole set of data. One might argue this should be done on the training set.

# In[103]:

#obtain a dataframe with only the Lasso regressors
X = news_data
y = news_labels
lm = linear_model.Lasso(alpha=100000, fit_intercept=True) #initialize Lasso model; how do we choose alpha?
lm.fit(X, y)
lasso_est = lm.coef_

X = (news_data * lasso_est.transpose()) # multiply element wise with lasso estimate
df_Lasso = X[X.columns[(X != 0).any()]] # remove columns where all elements are zero
print df_Lasso.shape # number of columns should significantly shrink depending on choice of alpha
df_Lasso.columns.values.tolist()


# In[104]:

#obtain a split
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(df_Lasso, news_labels)

#binarize
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=binary_threshold)
binary_labels = binarizer.transform(news_labels).transpose().ravel()     # .ravel() is to fix "Too many array indices error"
print binary_labels.shape


# In[107]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

knn = KNeighborsClassifier(n_neighbors=1) # arbitrary k
cv = cross_val_score(knn, df_Lasso, binary_labels, cv=10)
print "Cross Validation Scores"
print cv
print 'Mean Cross Validation Score'
print np.mean(cv)



