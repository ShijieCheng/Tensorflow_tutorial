# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 19:46:59 2021

@author: 成世杰
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#图片的像素大小
n_input=784
#数字种类
n_output=10
#输入数据占位符
net_input=tf.placeholder(tf.float32,[None,n_input])
#数据标签占位符
y_true=tf.placeholder(tf.float32,[None,10])

W=tf.Variable(tf.zeros([n_input,n_output]))
b=tf.Variable(tf.zeros([n_output]))

#线性回归模型
net_output=tf.nn.softmax(tf.matmul(net_input,W)+b)
#交叉熵函数
cross_entropy=-tf.reduce_sum(y_true*tf.log(net_output))
optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#计算准确率
correct_prediction=tf.equal(tf.arg_max(net_output,1),
                            tf.arg_max(y_true,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

#开始训练，迭代次数n_epochs=10,每次训练批量batch_size=100
batch_size=100
n_epochs=100

for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples//batch_size):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={net_input:batch_xs,y_true:batch_ys})
        
    #计算每次迭代的相关参数
    xdat=sess.run(accuracy,feed_dict=
                  {net_input:mnist.validation.images,
                   y_true:mnist.validation.labels})
    print(epoch_i," ",xdat)
    
#使用test测试数据和训练好模型，计算相关参数
xdat=sess.run(accuracy,feed_dict=
                  {net_input:mnist.test.images,
                   y_true:mnist.test.labels})
print('测试模型训练结果',xdat)
sess.close()











