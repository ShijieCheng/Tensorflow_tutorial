# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:54:54 2021

@author: 成世杰
"""

import tensorflow as tf

state=tf.Variable(9,name='counter')#初始值，名字
print(state.name)
one=tf.constant(16)

new_value=tf.add(state,one)
update=tf.assign(state,new_value)
init=tf.global_variables_initializer()

with tf.Session() as see:
    see.run(init)
    for _ in range(4):
        see.run(update)
        print(see.run(state))