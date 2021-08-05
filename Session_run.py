# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:28:26 2021

@author: 成世杰
"""

import tensorflow as tf

matrix1=tf.constant([[3,3]])

matrix2=tf.constant([[2],[2]])

product=tf.matmul(matrix1,matrix2)#矩阵乘法
#method1
#sess=tf.Session()
#result=sess.run(product)
#print(result)
#sess.close()

#method2
with tf.Session() as sess:
    result2=sess.run(product)
    print(result2)