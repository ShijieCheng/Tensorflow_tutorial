# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 11:29:15 2021

@author: 成世杰
"""
import tensorflow as tf

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.4],input2:[3.4]}))