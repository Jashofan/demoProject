# -*- coding: utf-8 -*-
"""
@Time : 2020/11/25
@Author : Jashofan
@File : demo07
@Description : softmax回归
"""

import torch
import torchvision
import numpy as np
import sys
import d2lzh_pytorch as d2l

sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch

# 获取和读取数据
batch_size = 256  # 批量大小为256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs = 784  # 28*28
num_outputs = 10  # 输出维度为10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

# 设置模型参数梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 实现softmax运算
'''只对其中同一列（dim=0）或同一行（dim=1）的元素求和，并在结果中保留行和列这两个维度（keepdim=True）'''
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))  # 同一列求和
print(X.sum(dim=1, keepdim=True))  # 同一行 求和

'''softmax运算会先通过exp函数对每个元素做指数运算，再对exp矩阵同行元素求和，最后令矩阵每行各元素与该行元素之和相除。
	这样一来，最终得到的矩阵每行元素和为1且非负。softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率。'''


# 矩阵X的行数是样本数，列数是输出个数
def softmax(X):
	X_exp = X.exp()  # 对X做指数运算，保证数据为正数
	partition = X_exp.sum(dim=1, keepdim=True)  # 同行元素求和
	return X_exp / partition  # 这里应用了广播机制


# 对于随机输入，将每个元素变成了非负数，且每一行和为1
X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


# 定义模型
# 通过view函数将每张原始图像改成长度为num_inputs的向量。
def net(X):
	return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# 定义损失函数
# 变量y_hat是2个样本在3个类别的预测概率，变量y是这2个样本的标签类别。
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
print(y_hat.gather(1, y.view(-1, 1)))  # 不改变y_hat的值，需要输出y_hat.gather()的值
'''gather函数，前一个参数为dim，0为列，1为行，后一个参数为index,且需要与LongTensor同型'''

# 在代码中，标签类别的离散值是从0开始逐一递增的。

# 交叉熵损失函数
def cross_entropy(y_hat, y):
	return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 计算分类准确率
def accuracy(y_hat, y):
	return (y_hat.argmax(dim=1) == y).float().mean().item()
'''其中y_hat.argmax(dim=1)返回矩阵y_hat每行中最大元素的索引，且返回结果与变量y形状相同'''

print(accuracy(y_hat, y))


# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
# 评价模型net在数据集data_iter上的准确率。
def evaluate_accuracy(data_iter, net):
	acc_sum, n = 0.0, 0
	for X, y in data_iter:
		acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
		n += y.shape[0]
	return acc_sum / n


print(evaluate_accuracy(test_iter, net))

num_epochs, lr = 5, 0.1
# 迭代周期数和学习率

# 训练模型
# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
	for epoch in range(num_epochs):
		train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
		for X, y in train_iter:
			y_hat = net(X)
			l = loss(y_hat, y).sum()

			# 梯度清零
			if optimizer is not None:
				optimizer.zero_grad()
			elif params is not None and params[0].grad is not None:
				for param in params:
					param.grad.data.zero_()

			l.backward()
			if optimizer is None:
				d2l.sgd(params, lr, batch_size)
			else:
				optimizer.step()  # “softmax回归的简洁实现”一节将用到

			train_l_sum += l.item()
			train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
			n += y.shape[0]
		test_acc = evaluate_accuracy(test_iter, net)
		print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
			epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# 预测
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
