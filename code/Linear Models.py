
# coding: utf-8

# In[12]:

get_ipython().magic(u'pylab inline')
import pandas as pd
import statsmodels.api as sm
import sklearn as sk
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np


# In[13]:

# Import csv data
raw_data = pd.read_csv('OnlineNewsPopularity_wLabels_deleteNoise.csv').iloc[:, 1:]      # read in csv, omit the first column of url
print np.median(raw_data[' shares'])
raw_data = raw_data.iloc[:, :-1] # Remove the binarized column of shares
news_data = raw_data.iloc[:, :-1]      # Take up to the second last column
news_labels = raw_data.iloc[:, -1]      # Take shares column for labels

print '\nBinary Threshold:'
binary_threshold = np.median(raw_data[' shares']) # the median amount of shares is the boundary of our independent variable


# In[15]:

#  Remove n non stopped words, 
news_data = news_data.drop(' n_non_stop_words', 1)

# Generate nonlinear variations of regressors
news_data[' timedelta_2'] = news_data[' timedelta']**2
news_data[' timedelta_3'] = news_data[' timedelta']**3

news_data[' n_tokens_title_2'] = news_data[' n_tokens_title']**2
news_data[' n_tokens_content_2'] = news_data[' n_tokens_content']**2

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


# In[16]:

# Test Lasso on a certain set to obtain a measure for optimal alpha
from sklearn.cross_validation import train_test_split

# Lasso regression
def lasso(alpha):
    lm = linear_model.Lasso(alpha=alpha, fit_intercept=True, max_iter=1000) #initialize Lasso model; how do we choose alpha?
    news_data_train, news_data_test, news_labels_train, news_labels_test = train_test_split(news_data, news_labels)
    lm.fit(news_data_train, news_labels_train)
    R2 = lm.score(news_data_test, news_labels_test)
#     print 'MAE'
    MAE = mean_absolute_error(news_labels_test, lm.predict(news_data_test))
#     print 'MSE'
    MSE = mean_squared_error(news_labels_test, lm.predict(news_data_test))

    from sklearn.preprocessing import Binarizer
    binarizer = Binarizer(threshold=binary_threshold)  # Threshold at 1400 because median of shares is 1400
    binary_labels = binarizer.transform(news_labels_test)
    binary_prediction = binarizer.transform(lm.predict(news_data_test))

#     print 'Classification score'
    cs = 1 - np.mean(np.absolute((binary_labels - binary_prediction)))
    return R2, MAE, MSE, cs


# In[17]:

# Ridge Regression
def ridge(alpha):
    lm = linear_model.Ridge(alpha=alpha, fit_intercept=True) 
    news_data_train, news_data_test, news_labels_train, news_labels_test = train_test_split(news_data, news_labels)
    lm.fit(news_data_train, news_labels_train)

    R2 = lm.score(news_data_test, news_labels_test)
#     print 'MAE'
    MAE = mean_absolute_error(news_labels_test, lm.predict(news_data_test))
#     print 'MSE'
    MSE = mean_squared_error(news_labels_test, lm.predict(news_data_test))

    from sklearn.preprocessing import Binarizer
    binarizer = Binarizer(threshold=binary_threshold)  # Threshold at 1400 because median of shares is 1400
    binary_labels = binarizer.transform(news_labels_test)
    binary_prediction = binarizer.transform(lm.predict(news_data_test))

#     print 'Classification score'
    cs = 1 - np.mean(np.absolute((binary_labels - binary_prediction)))
    return R2, MAE, MSE, cs


# In[18]:

# REGULAR REGRESSION MODEL
def ols(alpha):
    X = news_data
    y = news_labels
    lm = linear_model.Lasso(alpha=alpha, fit_intercept=True) #initialize Lasso model; how do we choose alpha?
    lm.fit(X, y)

    b = lm.coef_
    b[np.nonzero(b)] = 1
    X = (X * b.transpose()) # multiply element wise with lasso estimate
    X = X[X.columns[(X != 0).any()]]
    news_data_train, news_data_test, news_labels_train, news_labels_test = train_test_split(X, y)
    lm.fit(news_data_train, news_labels_train)

    R2 = lm.score(news_data_test, news_labels_test)
#     print 'MAE'
    MAE = mean_absolute_error(news_labels_test, lm.predict(news_data_test))
#     print 'MSE'
    MSE = mean_squared_error(news_labels_test, lm.predict(news_data_test))

    from sklearn.preprocessing import Binarizer
    binarizer = Binarizer(threshold=binary_threshold)  # Threshold at 1400 because median of shares is 1400
    binary_labels = binarizer.transform(news_labels_test)
    binary_prediction = binarizer.transform(lm.predict(news_data_test))

#     print 'Classification score'
    cs = 1 - np.mean(np.absolute((binary_labels - binary_prediction)))
    return R2, MAE, MSE, cs
ols(1)


# In[19]:
# Run all three models for different values of alpha
def lm_scores(alphas):
    df_ols = pd.DataFrame(data=np.empty((alphas.size, 4)), columns=['R2', 'MAE', 'MSE', 'cs'])
    df_ridge = pd.DataFrame(data=np.empty((alphas.size, 4)), columns=['R2', 'MAE', 'MSE', 'cs'])
    df_lasso = pd.DataFrame(data=np.empty((alphas.size, 4)), columns=['R2', 'MAE', 'MSE', 'cs'])
    for i in range(0, alphas.size):
        df_ols.iloc[i] = ols(alphas[i])
        df_ridge.iloc[i] = ridge(alphas[i])
        df_lasso.iloc[i] = lasso(alphas[i])
        print 'iteration %d succesful' %i
    return df_ols, df_ridge, df_lasso

# Generate alphas
n_alphas = 25
alphas = np.empty(n_alphas)
for i in range (0, n_alphas):
        alphas[i] = np.exp(i)/1000
print alphas

# Execute the algorithm
df_ols, df_ridge, df_lasso = lm_scores(alphas)


# In[22]:
#  save the data to folder
df_ols.to_csv(path_or_buf='ols.csv')
df_ridge.to_csv(path_or_buf='ridge.csv')
df_lasso.to_csv(path_or_buf='lasso.csv')


# In[23]:
# Plot the obtained data
get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
print df_ols.shape
n_reg = np.array([58,57,56,55,55,54,52,48,46,44,36,27,24,20,18,15,14,12,12,11,10,8,6,6,5])
print n_reg.size

plt.plot(alphas, df_ols['cs'])
plt.plot(alphas, df_ridge['cs'])
plt.plot(alphas, df_lasso['cs'])
plt.legend(['OLS', 'Ridge', 'Lasso'], loc='upper right')
plt.xlabel('value of alpha')
plt.ylabel('classification score')
plt.show()

plt.plot(alphas, df_ols['MSE'])
plt.plot(alphas, df_ridge['MSE'])
plt.plot(alphas, df_lasso['MSE'])
plt.legend(['OLS', 'Ridge', 'Lasso'], loc='upper right')
plt.xlabel('value of alpha')
plt.ylabel('Mean Square Error')
plt.show()

plt.plot(alphas, df_ols['cs'])
plt.plot(alphas, df_ridge['cs'])
plt.plot(alphas, df_lasso['cs'])
plt.legend(['OLS', 'Ridge', 'Lasso'], loc='upper right')
plt.xlabel('value of alpha')
plt.ylabel('classification score')
plt.show()


# In[28]:

n_regs = np.array([58,57,56,55,55,54,52,48,46,44,36,27,24,20,18,15,14,12,12,11,10,8,6,6,5])
plt.plot(np.log(alphas), n_regs)
plt.xlabel('ln(alpha)')
plt.ylabel('Number of Regressors')
plt.show


# In[100]:

#obtain a dataframe with only the Lasso regressors
# X = news_data
# y = news_labels
# lm = linear_model.Lasso(alpha=10000, fit_intercept=True) #initialize Lasso model; how do we choose alpha?
# lm.fit(X, y)
# b = lm.coef_
# b[np.nonzero(b)] = 1
# X = (X * b.transpose()) # multiply element wise with lasso estimate
# df_Lasso = X[X.columns[(X != 0).any()]]
# df_Lasso.shape

# # here, look at OLS using Lasso obtained model
# Now that the right data matrix is made, it time to run regressions!
X = df_Lasso
y = news_labels

# Add a constant and fit the OLS model
X = sm.add_constant(X)
est = sm.OLS(y, X).fit() 
est.summary()
# In[23]:

# # create test and train and compare regression model
lm = linear_model.LinearRegression()
news_data_train, news_data_test, news_labels_train, news_labels_test = train_test_split(news_data, news_labels)
lm.fit(news_data_train, news_labels_train)

from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=binary_threshold)  # Threshold at 1400 because median of shares is 1400
binary_labels = binarizer.transform(news_labels_test)
binary_prediction = binarizer.transform(lm.predict(news_data_test))

print 'classification score'
print 1 - np.mean(np.absolute((binary_labels - binary_prediction)))


# In[24]:

#Testing for heteroskedasticity
X = news_data
y = news_labels

# Add a constant and fit the OLS model
lm = linear_model.LinearRegression()
lm.fit(X, y)
resid = y - lm.predict(X)
# print resid.describe()
resid_2 = (resid**2)
# print resid_2.describe()

#Test for Heteroskedasticity
X = sm.add_constant(X)
est = sm.OLS(resid_2, X).fit() 
est.summary()


# In[ ]:



