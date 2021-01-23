# -*- coding: utf-8 -*-
"""
@Time : 2020/11/11
@Author : Jashofan
@File : demo05
@Description :线性回归模型训练简洁实现
"""

import torch
import torch.utils.data as Data  # 提供了有关数据处理的工具
import torch.optim as optim  # 提供了很多常用的优化算法
import torch.nn as nn  # 定义了大量神经网络的层
import numpy as np
import random
import sys
import ipykernel
import d2lzh_pytorch as d2l

sys.path.append("..")

# 人工模拟训练集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# 随机读取一批数据集
for X, y in data_iter:
	print(X, y)
	break

'''nn的核心数据结构是Module，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示
	一个包含很多层的神经网络。在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络/层。
	一个nn.Module实例应该包含一些层以及返回输出的前向传播（forward）方法。
	下面是用nn.Module实现一个线性回归模型。'''
# # torch.nn仅支持输入一个batch的样本不支持单个样本输入，如果只有单个样本，可使用input.unsqueeze(0)来添加一维。
# class LinearNet(nn.Module):
# 	def __init__(self, n_feature):
# 		super(LinearNet, self).__init__()
# 		self.linear = nn.Linear(n_feature, 1)

# 	# forward 定义前向传播
# 	def forward(self, x):
# 		y = self.linear(x)
# 		return y

# net = LinearNet(num_inputs)
# print(net)  # 使用print可以打印出网络的结构

'''还可以用nn.Sequential来更加方便地搭建网络，Sequential是一个有序的容器，
	网络层将按照在传入Sequential的顺序依次被添加到计算图中。'''
# # 写法一
# net = nn.Sequential(nn.Linear(num_inputs, 1)# 此处还可以传入其他层
# )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

# # 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#           ('linear', nn.Linear(num_inputs, 1))
#           # ......
#         ]))

print(net)
print(net[0])

# 通过net.parameters()来查看模型所有的可学习参数，此函数将返回一个生成器。
for param in net.parameters():
	print(param)

'''PyTorch在init模块中提供了多种参数初始化方法。这里的init是initializer的缩写形式。
	我们通过init.normal_将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布。偏差会初始化为零。'''
nn.init.normal_(net[0].weight, mean=0, std=0.01)
nn.init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

# 使用nn模块提供的均方误差损失作为模型的损失函数。
loss = nn.MSELoss()

# 创建一个用于优化net所有参数的优化器实例，并指定学习率为0.03的小批量随机梯度下降（SGD）为优化算法。
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

# 可以为不同子网络设置不同的学习率，这在finetune时经常用到。
# optimizer =optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net[0].subnet1.parameters()}, # lr=0.03
#                 {'params': net[0].subnet2.parameters(), 'lr': 0.01}
#             ], lr=0.03)

'''调整学习率主要有两种做法。一种是修改optimizer.param_groups中对应的学习率，
	另一种是更简单也是较为推荐的做法——新建优化器，由于optimizer十分轻量级，构建开销很小，故而可以构建新的optimizer。
	但是后者对于使用动量的优化器（如Adam），会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况。'''
# 调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍

'''在使用Gluon训练模型时，通过调用optim实例的step函数来迭代模型参数。
	按照小批量随机梯度下降的定义，在step函数中指明批量大小，从而对批量中样本梯度求平均。'''
num_epochs = 3
for epoch in range(1, num_epochs + 1):
	for X, y in data_iter:
		output = net(X)
		l = loss(output, y.view(-1, 1))
		optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
		l.backward()
		optimizer.step()
	print('epoch %d, loss: %f' % (epoch, l.item()))

# 分别比较学到的模型参数和真实的模型参数。
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
