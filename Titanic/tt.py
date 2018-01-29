# coding:utf-8
'''
Created on 2018/1/18.

@author: chk01
'''
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

# Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import pandas as pd

# from pandas.tools.plotting import scatter_matrix
data_raw = pd.read_csv('data/train.csv')

# a dataset should be broken into 3 splits: train, test, and (final) validation
# the test file provided is the validation file for competition submission
# we will split the train set into train and test data in future sections
data_val = pd.read_csv('data/test.csv')

# to play with our data we'll create a copy
# remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
data1 = data_raw.copy(deep=True)
# however passing by reference is convenient, because we can clean both datasets at once
data_cleaner = [data1, data_val]

# preview data
# data_raw.info()
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
# data_raw.head() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html
# data_raw.tail() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html
# print(data_raw.sample(10))  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html
# print('Train columns with null values:\n', data1.isnull().sum())
# print("-" * 10)
#
# print('Test/Validation columns with null values:\n', data_val.isnull().sum())
# print("-" * 10)

tt = data_raw.describe(include='all')

###COMPLETING: complete or delete missing values in train and test/validation dataset
for dataset in data_cleaner:
    # complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

    # complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

    # complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

# delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId', 'Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace=True)

# print(data1.isnull().sum())
# print("-" * 10)
# print(data_val.isnull().sum())

###CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_cleaner:
    # Discrete variables
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1  # initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0  # now update to no/0 if family size is greater than 1

    # quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    # print(dataset['Name'].str.split(", ", expand=True)[1])
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    # Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    # Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    # Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

# cleanup rare title names
# print(data1['Title'].value_counts())
stat_min = 10  # while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (
    data1['Title'].value_counts() < stat_min)  # this will create a true false series with title name as index
print(title_names)
# apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
# print(data1['Title'].value_counts())
# print("-" * 10)

# preview data again
# data1.info()
# data_val.info()
# data1.sample(10)
