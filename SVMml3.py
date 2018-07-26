#Breast cancer repository#

import numpy as np
import sklearn
import preprocessing
from sklearn import cross_validation,svm
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('breastcancerdata.txt')

df.replace('?', -99999, inplace=True)
#if there is a missing data with ? then we replace it with -99999.
#This is way of dealing with missing data by treating it as large outliers.
#Python considers -99999 as an outlier and proceeds instead of dumping missing data which is usually large in many cases.

df.drop(['id'], 1, inplace=True) #dropping the first column 'id' which is useless.
#in SVM, even if we don't drop outliers, it still does relatively better than knn in case of accuracy.

x = np.array(df.drop(['class'],1))# x is the features
y = np.array(df['class']) # y is the labels

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = svm.SVC()#svm calls support vector machine. instead of knn USING svm.

clf.fit(x_train, y_train) 

accuracy = clf.score(x_test,y_test)#testing

print(accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1])#only one random sample
example_measures = example_measures.reshape(1,-1)#(2,-1) in case of two samples/arrays
#alternatively we can use example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)

print(prediction)

