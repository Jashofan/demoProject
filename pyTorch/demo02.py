# -*- coding: utf-8 -*-
"""
@Time : 2020/11/6
@Author : Jashofan
@File : demo02
@Description :梯度
"""

import torch

'''requires_grad有两个值：True和False，True代表此变量处需要计算梯度，False代表不需要。默认是False'''
x = torch.ones(2, 2, requires_grad=True)  # 设为True即开始追踪(track)其所有操作，打印时会同步输出
print(x)
print(x.grad_fn)
'''每个Tensor都有一个.grad_fn属性，该属性即创建该Tensor的Function, 就是说该Tensor是不是通过某些运算得到的，
	若是，则grad_fn返回一个与这些运算相关的对象，否则是None。grad_fn的值可以得知该变量是否是一个计算结果，
	也就是说该变量是不是一个函数的输出值。'''

y = x + 2
print(y)
print(y.grad_fn)
'''注意x是直接创建的，所以它没有grad_fn, 而y是通过一个加法操作创建的，所以它有一个为<AddBackward>的grad_fn。
	像x这种直接创建的称为叶子节点，叶子节点对应的grad_fn是None'''

z = y * y * 3
out = z.mean()  # 求该张量的均值，标量
print(z, out)

a = torch.randn(2, 2)  # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad)  # False 此时a的grad_fn为none
print(a.grad_fn)
a.requires_grad_(True)  # 通过.requires_grad_()来用in-place的方式改变requires_grad属性
print(a.requires_grad)  # True
b = (a * a).sum()
print(b.grad_fn)

'''不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量。'''
'''注意在y.backward()时，如果y是标量，则不需要为backward()传入任何参数；否则，需要传入一个与y同形的Tensor'''
# out.backward()  # 等价于 out.backward(torch.tensor(1.))，指定out对x求导
out.backward(torch.tensor(1))
print(x.grad)  # out关于x的梯度 d(out)/dx

'''注意：grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，
	梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。'''
# 再来反向传播一次，注意grad是累加的，个人理解：链式法则
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()  # 将梯度清零
out3.backward()
print(x.grad)

x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)  # 即改变形状后的y
print(z)

v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)  # z对x求导，z 不是一个标量，所以在调用backward时需要传入一个和z同形的权重向量进行加权求和得到一个标量。
print(x.grad)

x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
print(y1)
print('---------------------------------------')
with torch.no_grad():  # 被此函数包裹的变量的有关的梯度是不会回传的，因此也不能调用backward()
	y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad)  # True
print(y2, y2.requires_grad)  # False
print(y3, y3.requires_grad)  # True

y3.backward()
print(x.grad)  # 故结果不是5，而是1

x = torch.ones(1, requires_grad=True)

# .data——获得该节点的值，即Tensor类型的值
print(x.data)  # 还是一个tensor
print(x.data.requires_grad)  # 但是已经是独立于计算图之外

y = 2 * x
x.data *= 100  # 只改变了值，不会记录在计算图，所以不会影响梯度传播

y.backward()
print(x)  # 更改data的值也会影响tensor的值
print(x.grad)
