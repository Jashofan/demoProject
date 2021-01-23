# -*- coding: utf-8 -*-
"""
@Time : 2020/9/22
@Author : Jashofan
@File : demo00
@Description : this is a demo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from IPython.display import display

# python示例代码

# Numpy
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))

# scipy
# 创建一个二维NumPy数组，对角线为1，其余都为0
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

# 将NumPy数组转换为CSR格式的SciPy稀疏矩阵
# 只保存非零元素
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n{}".format(eye_coo))

# matplotlib
# 在-10到10之间生成一个数列，共1000个数
x = np.linspace(-10, 10, 100)
# 用正弦函数创建第二个数组
y = np.sin(x)
# plot函数绘制一个数组关于另一个数组的折线图
plt.plot(x, y, marker="x")
plt.show()

# pandas
# 创建关于人的简单数据集
data = {'Name': ["John", "Anna", "Peter", "Linda"], 'Location': ["New York", "Paris", "Berlin", "London"],
		'Age': [24, 13, 53, 33]}
data_pandas = pd.DataFrame(data)
display(data_pandas)

# Select all rows that have an age column greater than 30
display(data_pandas[data_pandas.Age > 30])