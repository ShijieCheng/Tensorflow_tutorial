# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:44:33 2021

@author: 成世杰
"""

import tensorflow as tf

#创建三个节点，2个常量，一个operation
#a=tf.constant(3)
#b=tf.constant(4)
#c=tf.add(a,b)
#
##执行Session
##with tf.Session() as sess:
##    print(sess.run(c))
#sess=tf.Session()
#print(sess.run(c))
#sess.close()


#a=tf.placeholder(tf.float32)
#b=tf.placeholder(tf.float32)
#add_node=a-b
#
#with tf.Session() as sess:
#    print(sess.run(add_node,{a:1,b:2}))


#W=tf.Variable([2],dtype=tf.float32)
#b=tf.Variable([-1],dtype=tf.float32)
#x=tf.placeholder(tf.float32)
#linear_model=W*x+b
#
#init = tf.global_variables_initializer()
#with tf.Session() as sess:
#    sess.run(init)
#    print("linear_model:",linear_model)
#    print(sess.run(linear_model,{x:[1,2,3,4]}))



#b=tf.get_variable(name='var6',shape=[1,2])
#init = tf.global_variables_initializer()
#with tf.Session() as sess:
#    sess.run(init)
#    print(b.eval())


#var1=tf.get_variable(name="zero_var",shape=[1,2,3],dtype=tf.float32)
#var2=tf.get_variable(name='user_var',initializer=tf.constant([1,2,3],dtype=tf.float32))




















