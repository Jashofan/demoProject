# -*- coding: utf-8 -*-
"""
@Time : 2020/11/6
@Author : Jashofan
@File : Demo00
@Description : create tensor and operate
"""

import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.float64)  # 返回的tensor默认具有相同的torch.dtype和torch.device
print(x)

x = torch.randn_like(x, dtype=torch.float)  # 指定新的数据类型
print(x)

# 加法形式一
y = torch.randn(5, 3)
print(x + y)
# 加法形式二
print(torch.add(x, y))
# 指定输出
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# 加法形式三 inplace
# adds x to y
y.add_(x)
print(y)

# 索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。
y = x[0, :]
y += 1
print(y)
print(x[0, :])  # 源tensor也被改了

# 用view()来改变Tensor的形状，view仅仅是改变了对这个张量的观察角度，内部数据并未改变
y = x.view(15)
z = x.view(-1, 3)  # -1所指的维度可以根据其他维度的值推出来,此处-1应为5
print(x.size(), y.size(), z.size())
print(y)

# 不共享data的方法，clone一个
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)

# item(), 它可以将一个标量Tensor转换成一个Python number
x = torch.randn(1)
print(x)
print(x.item())

