# -*- coding: utf-8 -*-
"""
@Time : 2020/11/10
@Author : Jashofan
@File : demo04
@Description : 线性回归模型训练
"""

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
import ipykernel
import d2lzh_pytorch as d2l

sys.path.append("..")

# 人工生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
# randn返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量形状由参数决定
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
# [:,0]表示二维数组中第二维下标为0的元素
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 加上一个服从均值为0、标准差为0.01的正态分布的噪声项，size为形状
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

# features的每一行是一个长度为2的向量，而labels的每一行是一个长度为1的向量（标量）
print(features[0], labels[0])

# def use_svg_display():
# 	# 用矢量图显示
# 	display.set_matplotlib_formats('svg')
#
#
# def set_figsize(figsize=(3.5, 2.5)):
# 	use_svg_display()
# 	# 设置图的尺寸
# 	plt.rcParams['figure.figsize'] = figsize


# # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# import sys
# sys.path.append("..")
# from d2lzh_pytorch import *

d2l.set_figsize()
# 房龄与售价的散点图，数字一代表散点图中点的大小
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.show()

# # 本函数已保存在d2lzh包中方便以后使用
# #每次返回batch_size（批量大小）个随机样本的特征和标签。
# def data_iter(batch_size, features, labels):
# 	num_examples = len(features)
# 	indices = list(range(num_examples))
# 	random.shuffle(indices)  # 样本的读取顺序是随机的
# 	for i in range(0, num_examples, batch_size):
# 		j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
# 		yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10

# 读取第一个小批量数据样本并打印。
for X, y in d2l.data_iter(batch_size, features, labels):
	print(X, y)
	break

# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# 之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此我们要让它们的requires_grad=True。
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
# 	return torch.mm(X, w) + b
# 用mm函数做矩阵乘法，第一个矩阵乘第二个
#
# 用平方损失来定义线性回归的损失函数。注意需要将y变成预测值y_hat一样的形状
# def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
# 	# 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
# 	return (y_hat - y.view(y_hat.size())) ** 2 / 2
#
# 小批量随机梯度下降算法。它通过不断迭代模型参数来优化损失函数。
# 这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。我们将它除以批量大小来得到平均值。
# def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
# 	for param in params:
# 		param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


lr = 0.03  # 学习率
num_epochs = 3  # 迭代周期个数
net = d2l.linreg
loss = d2l.squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
	# 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
	# 和y分别是小批量样本的特征和标签
	for X, y in d2l.data_iter(batch_size, features, labels):
		l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
		l.backward()  # 小批量的损失对模型参数求梯度
		d2l.sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

		# 不要忘了梯度清零
		w.grad.data.zero_()
		b.grad.data.zero_()
	train_l = loss(net(features, w, b), labels)
	print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)
