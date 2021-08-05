# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 21:36:46 2020

@author: 成世杰
"""

import tensorflow as tf
import numpy as np

x=np.random.rand(100).astype(np.float32)
y=x*0.1+0.3

Weights=tf.Variable(tf.random_uniform([1],-1,1))
biases=tf.Variable(tf.zeros([1]))

y1=Weights*x+biases
loss=tf.reduce_mean(tf.square(y-y1))
optimizer=tf.train.GradientDescentOptimizer(0.4)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()

sess.run(init)
for step in range(201):
    sess.run(train)
    if step % 3 ==0:
        print(step,sess.run(Weights),sess.run(biases),
              sess.run(loss))