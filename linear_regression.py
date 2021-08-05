# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:34:29 2021

@author: 成世杰
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#阈值
threshold=1.0e-2
#normal distribution(#randn)x_data
x_data=np.random.randn(100).astype(np.float32)
y_data=x_data*2+1
#权值变量与偏置变量
weight=tf.Variable(1.0)
bias=tf.Variable(1.0)
#x,y的占位符
x_=tf.placeholder(tf.float32)
y_=tf.placeholder(tf.float32)

#线性模型的预测值
y_model=tf.add(tf.multiply(x_,weight),bias)
#损失函数，梯度下降算法训练
loss=tf.reduce_mean(tf.pow((y_model-y_),2))
train_op=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#执行图计算

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  
    while (sess.run(loss,feed_dict={x_:x_data,y_:y_data}) > threshold): 
        sess.run(train_op,feed_dict={x_:x_data,y_:y_data})
        print(sess.run(weight,feed_dict={x_:x_data,y_:y_data}),
                       bias.eval(sess))#两种表达方式
        
    plt.plot(x_data,y_data,'ro',label='Original data')
    plt.plot(x_data,sess.run(weight)*x_data+sess.run(bias),label='Fitted Line')
    plt.legend()
    plt.show()

            





























