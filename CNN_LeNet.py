# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 11:10:58 2021

@author: 成世杰
"""

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)


#输入数据占位符
x=tf.placeholder('float',[None,784])
#数据标签占位符
y_=tf.placeholder('float',[None,10])
#输入图片的数据形状转化为28*28矩阵
x_image=tf.reshape(x,[-1,28,28,1])

#第一层卷积层，初始化卷积核参数，偏置，卷积层大小5*5，
#一个通道有6个不同的卷积核，步长1*1,等大填充
filter1=tf.Variable(tf.truncated_normal([5,5,1,6]))
bias1=tf.Variable(tf.truncated_normal([6]))
conv1=tf.nn.conv2d(x_image,filter1,strides=[1,1,1,1],padding="SAME")
h_conv1=tf.nn.sigmoid(conv1+bias1)
#第一层池化层，最大值池化，尺寸为2*2，步长为2*2，等大填充
maxPool1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第二层卷积层，初始化卷积核参数，偏置，卷积层大小5*5，
#6个输入通道有16个不同的卷积核，步长1*1,等大填充
filter2=tf.Variable(tf.truncated_normal([5,5,6,16]))
bias2=tf.Variable(tf.truncated_normal([16]))
conv2=tf.nn.conv2d(maxPool1,filter2,strides=[1,1,1,1],padding="SAME")
h_conv2=tf.nn.sigmoid(conv2+bias2)
#第二层池化层，最大值池化，尺寸为2*2，步长为2*2，等大填充
maxPool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第三层卷积层，初始化卷积核参数，偏置，卷积层大小5*5，
#16个输入通道有120个不同的卷积核，步长1*1,等大填充
filter3=tf.Variable(tf.truncated_normal([5,5,16,120]))
bias3=tf.Variable(tf.truncated_normal([120]))
conv3=tf.nn.conv2d(maxPool2,filter3,strides=[1,1,1,1],padding="SAME")
h_conv3=tf.nn.sigmoid(conv3+bias3)


#全连接层，产生权值参数，偏置变量
W_fc1=tf.Variable(tf.truncated_normal([7*7*120,80]))
b_fc1=tf.Variable(tf.truncated_normal([80]))
#形状转化，将卷积的输出展开
h_pool2_flat=tf.reshape(h_conv3,[-1,7*7*120])
#神经网络计算，并添加sigmoid函数
h_fc1=tf.nn.sigmoid(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#输出层，使用softmax进行多分类
#产生权值参数，偏置变量
W_fc2=tf.Variable(tf.truncated_normal([80,10]))
b_fc2=tf.Variable(tf.truncated_normal([10]))
#神经网络计算
y_conv=tf.nn.softmax(tf.matmul(h_fc1,W_fc2)+b_fc2)


#损失函数
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

#准确率
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #进行训练
    for i in range(1000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        if i%100==0:
            train_accuracy=accuracy.eval(feed_dict={x:batch_xs,y_:batch_ys})
            print(i,train_accuracy)
        train_step.run(feed_dict={x:batch_xs,y_:batch_ys})
        
    #使用test测试数据集和训练好的模型
    xdat=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
    print('测试模型的结果',xdat)





























