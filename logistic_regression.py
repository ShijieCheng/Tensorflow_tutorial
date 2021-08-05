# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:10:12 2021

@author: 成世杰
"""

import tensorflow as tf
import numpy as np

#数据个数
number_point=100
#生成实验数据
vectors_set=[]
for i in range(number_point):
    x1=np.random.normal(0.0,1)
    y1=1 if x1*0.3+0.1+np.random.normal(0.0,0.03)>0 else 0
    vectors_set.append([x1,y1])
x_data=[v[0] for v in vectors_set]
y_data=[v[1] for v in vectors_set]

#生成权值变量和偏置变量
W=tf.Variable(tf.random_uniform([1],-1.0,1.0))#范围是(-1，0)
b=tf.Variable(tf.zeros([1]))

#生成预测值
y=tf.sigmoid(W*x_data+b)

one=tf.ones(y.get_shape(),dtype=tf.float32)
#交叉熵损失函数
loss=-tf.reduce_mean(y_data*tf.log(y)+(one-y_data)*tf.log(one-y))

train=tf.train.GradientDescentOptimizer(0.4).minimize(loss)

#图计算
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #threshold
    th=tf.ones_like(one,dtype=tf.float32)*0.5
    
    #评估
    #计算预测准确率，将预测值与标签值进行比较
    #equal（）判断向量是否相等，返回布尔值
    correct_prediction=tf.equal(tf.cast(y_data,tf.int32),
                                tf.cast(tf.greater(y,th),tf.int32))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    for i in range(200):
        sess.run(train)
        if i%20 == 0:
            print("Accuracy:",sess.run(accuracy))
            print("loss",sess.run(loss))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    