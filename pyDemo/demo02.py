# -*- coding: utf-8 -*-
"""
@Time : 2020/10/14
@Author : Jashofan
@File : demo02
@Description : 
"""
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

print("Keys of iris_dataset:\n", iris_dataset.keys())
print(iris_dataset['DESCR'][:193] + "\n...")

print("Target names:", iris_dataset['target_names'])
print("Feature names:\n", iris_dataset['feature_names'])

print("Type of data:", type(iris_dataset['data']))
print("Shape of data:", iris_dataset['data'].shape)

print("First five rows of data:\n", iris_dataset['data'][:5])
print("Type of target:", type(iris_dataset['target']))
print("Shape of target:", iris_dataset['target'].shape)
print("Target:\n", iris_dataset['target'])

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60,
						   alpha=.8, cmap=mglearn.cm3)
plt.show()
