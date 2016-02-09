I. Group Details
Group ID: Gasanova
Group member names: 
	1. Elvira Gasanova
	2. Nguyen Ngoc Tram Anh
	3. Marcio Porto
	4. Liu Shaobo
	5. Lasse Vuursteen
	6. Patricio Beltran

II. Instructions to reproduce results
** For Python code **
For classifier implemented in Python, there is a separate and stand alone 
.py file which can run independently.

For libraries, the following libraries should be installed:
SciKit Learn - http://scikit-learn.org/
Numpy - http://www.numpy.org/
Matplotlib - http://matplotlib.org/
Statsmodels - http://statsmodels.sourceforge.net/
Pandas - http://pandas.pydata.org/
Optionally if inline output is desired: iPython - http://ipython.org/

** For Weka **
The following describes how to reproduce the results in Weka:
1. Download the data from UCI website
2. Using Excel, remove column n_non_stop_words
3. Using Excel, remove all rows where n_tokens_content=0
4. Using Excel, remove rows where n_unique_tokens>1
5. Using Excel, binarize shares column where label=1 if shares>1400 and label=0 otherwise
6. Import the resulted csv in Weka
7. Remove shares, url, timedelta columns
8. Nominal the label column using filter
9. If run Decision Tree or Naive Bayes, discretize all attributes (except label) with bins 30 (for DT), and 40(for NB)
10. Run the desired classifier in 'Classify' tab

