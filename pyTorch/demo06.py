# -*- coding: utf-8 -*-
"""
@Time : 2020/11/11
@Author : Jashofan
@File : demo06
@Description : 图像分类数据集
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

'''transforms.ToTensor()将尺寸为 (H x W x C) 且数据位于[0, 255]的PIL图片或者数据类型为np.uint8的NumPy数组转换为尺寸
    为(C x H x W)且数据类型为torch.float32且位于[0.0, 1.0]的Tensor。'''
mnist_train = torchvision.datasets.FashionMNIST(root='E:\workspace\demoProject\pyTorch\Datasets', train=True, download=False,
												transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='E:\workspace\demoProject\pyTorch\Datasets', train=False, download=False,
											   transform=transforms.ToTensor())

'''训练集中和测试集中的每个类别的图像数分别为6,000和1,000。因为有10个类别，所以训练集和测试集的样本数分别为60,000和10,000。'''
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 通过下标来访问任意一个样本
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width

# 以下函数可以将数值标签转成相应的文本标签。
# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	return [text_labels[int(i)] for i in labels]


# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
	d2l.use_svg_display()
	# 这里的_表示我们忽略（不使用）的变量
	_, figs = plt.subplots(1, len(images), figsize=(12, 12))
	for f, img, lbl in zip(figs, images, labels):
		f.imshow(img.view((28, 28)).numpy())
		f.set_title(lbl)
		f.axes.get_xaxis().set_visible(False)
		f.axes.get_yaxis().set_visible(False)
	plt.show()


X, y = [], []
for i in range(10):
	X.append(mnist_train[i][0])
	y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

batch_size = 256
if sys.platform.startswith('win'):
	num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
	num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

start = time.time()
for X, y in train_iter:
	continue
print('%.2f sec' % (time.time() - start))
