# -*- coding: utf-8 -*-
"""
@Time : 2020/11/6
@Author : Jashofan
@File : demo01
@Description : 广播机制
"""

import torch
import numpy as np

# 广播（broadcasting）机制：先适当复制元素使这两个Tensor形状相同后再按元素运算。
'''由于x和y分别是1行2列和3行1列的矩阵，如果要计算x + y，那么x中第一行的2个元素
	被广播（复制）到了第二行和第三行，而y中第一列的3个元素被广播（复制）到了第二
	列。如此，就可以对2个3行2列的矩阵按元素相加。'''
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

# y = x + y这样的运算是会新开内存的，然后将y指向新内存
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before)  # False
# 把x + y的结果通过[:]写进y对应的内存中
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before)  # True
print(id(y[:]))
print(id(y))

# 用运算符全名函数中的out参数或者自加运算符+=(也即add_())达到上述效果
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y)  # y += x, y.add_(x)
print(id(y) == id_before)  # True

print(y[:])  # 张量y的所有元素,和原来的y不指向同一个内存地址
print(y[:-1])  # 去掉最后一个元素
print(y[1:])  # 去掉第一个元素

# 使用numpy()将Tensor转换成NumPy数组
a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

# 使用from_numpy()将NumPy数组转换成Tensor
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)
'''这两个函数所产生的的Tensor和NumPy中的数组共享相同的内存（所以他们之间的转换很快），改变其中一个时另一个也会改变！！！'''

# 用方法to()可以将Tensor在CPU和GPU（需要硬件支持）之间相互移动
# 以下代码只有在PyTorch GPU版本上才会执行
if torch.cuda.is_available():
	device = torch.device("cuda")  # GPU
	y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
	x = x.to(device)  # 等价于 .to("cuda")
	z = x + y
	print(z)
	print(z.to("cpu", torch.double))  # to()还可以同时更改数据类型
