# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:44:33 2021

@author: 成世杰
"""
#导入数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
import matplotlib.pyplot as plt

#训练集的张量
print('训练集张量形状',mnist.train.images.shape)
print('训练集标签形状',mnist.train.labels.shape)
print('验证集张量形状',mnist.validation.images.shape)
print('验证集标签形状',mnist.validation.labels.shape)
print('测试集张量形状',mnist.test.images.shape)
print('测试集标签形状',mnist.test.labels.shape)

#数字图片的显示
for i in range(9):
    im=mnist.train.images[i]
    im=im.reshape(-1,28)
    #-1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1
    plt.subplot(3,3,i+1)
    plt.imshow(im)
plt.show()



