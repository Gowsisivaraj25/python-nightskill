# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:56:37 2022

@author: Student
"""

from sklearn import datasets
import pandas as pd


#loading the iris plants datasets(classification)
iris = pd.read_csv('C:/Users/Student/Downloads/nightskill/Iris.csv')
print(iris.head())
X,y = datasets.load_iris( return_X_y = True)

#spliting arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30)
print(iris.SepalWidthCm )

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#Creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)

#training the model on the training dataset
#fit function is used to train he model 
clf.fit(X_train, y_train)
