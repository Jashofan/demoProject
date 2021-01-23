# -*- coding: utf-8 -*-
"""
@Time : 2020/11/10
@Author : Jashofan
@File : demo03
@Description : 两种向量加法对比
"""

import torch
from time import time

a = torch.ones(1000)
b = torch.ones(1000)

# 两个向量按元素逐一做标量加法
start = time()
c = torch.zeros(1000)
for i in range(1000):
	c[i] = a[i] + b[i]
print(time() - start)
# 将这两个向量直接做矢量加法
start = time()
d = a + b
print(time() - start)
'''我们应该尽可能采用矢量计算，以提升计算效率。'''

a = torch.ones(3)
b = 10
print(a + b)
