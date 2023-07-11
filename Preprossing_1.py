""""
These are the libraries we are going to use for the prepocessing
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


"""
We take an instance of the data we are going to use
X is the matrix of features, it consist of the columns of the dataset that contains relevant info for helping 
the IA to predict 
Y is the dependent variable vector, that is what we want to predict
"""

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print("Dataset values: ")
print("\nFeatures matrix")
print(X)
print("\nDependent variable vector")
print(y)


print(dataset.isnull().sum())

"""
This is a use case for the simpleimputer class that we use for replacing any NaN value in the feature matrix
using the method fit, it calculates the value depending the method we chose, in this case 'mean', 
and transform applies this changes in the feature's matrix
"""
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("\nNaN Values corrected")
print(X)


"""
These classes are used for encoding the categorical columns like the country one, transforming it using a
method called one hot enconding that consist in transforming the values into binary vectors, this is important 
for 2 reasons, the first one is for aliminating the implicit hierarchy that can be misintepretated, and the second one
is that machine learning models learn easier with numerical variables

we use np.array for the fact that future classes for modeling training are going to expect it
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

"""
Label works for transforming categorical data that doesn't have order relation
"""
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

"""
We are going to split the dataset in training and test sets, this process is important to do it before feature scaling
for avoiding the data contamination, what means that the test and training sets would be related if we do this before 
splitting
"""

x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

"""
This type of normalization is used for making the machine larning model a way to easy to interpretate the data avoiding
misinterpretated dominance or recessiveness.
in x_test we use the same scale than x_train but without taking them in account, because the test is suppoused to be new
data, no related to the training one
"""

sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:])
print(x_train)
print(x_test)