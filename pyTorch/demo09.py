# -*- coding: utf-8 -*-
"""
@Time : 2020/11/25
@Author : Jashofan
@File : demo09
@Description : 激活函数
"""

import torch
import numpy as np
import matplotlib.pylab as plt
import sys
import d2lzh_pytorch as d2l

sys.path.append("..")


def xyplot(x_vals, y_vals, name):
	d2l.set_figsize(figsize=(5, 2.5))
	d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
	d2l.plt.xlabel('x')
	d2l.plt.ylabel(name + '(x)')
	plt.show()


# Relu函数
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')

y.sum().backward()
xyplot(x, x.grad, 'grad of relu')

# sigmoid函数
y = x.sigmoid()
xyplot(x, y, 'sigmoid')

x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')

# tanh函数
y = x.tanh()
xyplot(x, y, 'tanh')

x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')
