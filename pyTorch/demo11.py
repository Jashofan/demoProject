# -*- coding: utf-8 -*-
"""
@Time : 2020/12/2
@Author : Jashofan
@File : demo11
@Description :  多层感知机的简洁实现
"""

import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import d2lzh_pytorch as d2l

sys.path.append("..")

num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(d2l.FlattenLayer(), nn.Linear(num_inputs, num_hiddens), nn.ReLU(),
					nn.Linear(num_hiddens, num_outputs), )

for params in net.parameters():
	init.normal_(params, mean=0, std=0.01)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)