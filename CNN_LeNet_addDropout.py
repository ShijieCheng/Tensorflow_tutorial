# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 17:04:28 2021

@author: 成世杰
"""

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#定义权重变量
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
#定义偏置函数
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape) 
    return tf.Variable(initial)
#定义卷积函数，返回一个步长为1的二维卷积层，等大填充
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
#定义池化函数，最大值池化，尺寸为2*2，步长2*2，等大填充
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#构建卷积神经网络模型，用于MNIST数据集手写数字分类
def deepnn(x):
    x_image=tf.reshape(x,[-1,28,28,1])
    
    #第一层卷积层，初始化卷积核参数，偏置，卷积层大小5*5，
    #一个通道有32个不同的卷积核
    W_conv1=weight_variable([5,5,1,32])
    b_conv1=bias_variable([32])
    h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    #第一个池化层，对数据进行2倍增幅下采样
    h_pool1=max_pool_2x2(h_conv1)
    
    #第二层卷积层，初始化卷积核参数，偏置，卷积层大小5*5，
    #32个通道有64个不同的卷积核
    W_conv2=weight_variable([5,5,32,64])
    b_conv2=bias_variable([64])
    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    #第二个池化层，对数据进行2倍增幅下采样
    h_pool2=max_pool_2x2(h_conv2)
    
    #全连接层1，产生权值参数、偏置变量，通过2次下采样操作，输入的28x28图像，
    #转换成7x7x64特征映射图，映射1024个特征点
    W_fc1=weight_variable([7*7*64,1024])
    b_fc1=bias_variable([1024])
    
    h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    
    #使用dropout层控制模型的复杂度，防止特征点相互干扰
    keep_prob=tf.placeholder(tf.float32)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
    
    #1024个特征点，映射到10个类，每类为一个数字
    W_fc2=weight_variable([1024,10])
    b_fc2=bias_variable([10])
    y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2
    
    return y_conv,keep_prob

#输入数据占位符
x=tf.placeholder(tf.float32,[None,784])
#数据标签占位符
y_=tf.placeholder(tf.float32,[None,10])

#使用深度神经网络
y_conv,keep_prob=deepnn(x)

#损失函数
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_,logits=y_conv))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#准确率
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #进行训练
    for i in range(1000):
        batch=mnist.train.next_batch(50)
        if i%100==0:
            train_accuracy=sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            print(i,train_accuracy)
        sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
        
    
    
    
    
    
    