Final Project for Data Mining and Analytics course.
Using several Machine LEarning apporaches, we determine the elements of and article responsible for its mediatic success.
Nanyang Technological University, Singapore, 2015


Producing and sharing online content has become easier than ever thanks to the countless social platforms out there. But not all content shared becomes equally popular. Research shows that a very small number of items receive the most attention, while most content receives only a few views. So what are the elements that make an item more popular than others? That is the answer we set out to find in this project. Guided by previous research, our team analyzed a dataset obtained from UCI Machine Learning Repository, which consists of articles gathered over a two year period from mashable.com, in order to find elements that can lead to content popularity. We found that the factors with the greatest positive impact are subjectivity, data channel (topic) and whether an article is released on the weekend. From different classification methods used, Decision Trees and Naïve Bayes seem to be the best performing models. These results could potentially help content pro- ducers, such as online news agencies, reach larger audiences by producing content that is more likely to be read and shared by social media users.
-----------------------------------------

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

