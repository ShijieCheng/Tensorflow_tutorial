# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:19:19 2021

@author: 成世杰
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:34:29 2021

@author: 成世杰
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#阈值
threshold=1.0e-2
#normal distribution(#randn)x1_data,x2_data
x1_data=np.random.randn(100).astype(np.float32)
x2_data=np.random.randn(100).astype(np.float32)

y_data=x1_data*2+x2_data*2+1
#权值变量与偏置变量
weight1=tf.Variable(1.0)
weight2=tf.Variable(1.0)
bias=tf.Variable(1.0)
#x,y的占位符
x1_=tf.placeholder(tf.float32)
x2_=tf.placeholder(tf.float32)
y_=tf.placeholder(tf.float32)

#线性模型的预测值
y_model=tf.add(tf.add(tf.multiply(x1_,weight1),tf.multiply(x2_,weight2)),bias)
#损失函数，梯度下降算法训练
loss=tf.reduce_mean(tf.pow((y_model-y_),2))
train_op=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#执行图计算

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  
    while (sess.run(loss,feed_dict={x1_:x1_data,x2_:x2_data,y_:y_data}) > threshold): 
        sess.run(train_op,feed_dict={x1_:x1_data,x2_:x2_data,y_:y_data})
        print(sess.run(weight1),weight2.eval(sess),bias.eval(sess))
        
    fig=plt.figure()
    ax=Axes3D(fig)
    X,Y=np.meshgrid(x1_data,x2_data)
    Z=sess.run(weight1)*X+sess.run(weight2)*Y+sess.run(bias)
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.cm.hot)
    
            





























