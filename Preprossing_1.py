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



"""
This is a case use for the simpleimputer class that we use for replacing any NaN value in the feature matrix
using the method fit it calculates the value depending the method we chose, in this case 'mean', 
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
for 2 reasons, the first one is that eliminate the implicit hierarchy that can be misintepretated, and the second one
is that machine learning models learn easier with numerical variables

we use np.array for the fact that future classes for modeling training are going to expect it
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

"""

"""
le = LabelEncoder()
y = le.fit_transform(y)
print(y)




