# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:31:04 2021

@author: 成世杰
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

#数字种类，像素大小，隐藏层大小，批次，迭代次数
num_classes=10
input_size=784
hidden_units_size=30
batch_size=100
training_iterations=10000

X=tf.placeholder(tf.float32,shape=[None,input_size])
Y=tf.placeholder(tf.float32,shape=[None,num_classes])

#定义隐藏层权重与偏置
W1=tf.Variable(tf.random_normal([input_size,hidden_units_size],stddev=0.1))
B1=tf.Variable(tf.constant(0.1),[hidden_units_size])
#定义输出层权重与偏置
W2=tf.Variable(tf.random_normal([hidden_units_size,num_classes],stddev=0.1))
B2=tf.Variable(tf.constant(0.1),[num_classes])
#隐藏层输出计算
hidden_opt=tf.matmul(X,W1)+B1
hidden_opt=tf.nn.relu(hidden_opt)
#输出层计算
final_opt=tf.matmul(hidden_opt,W2)+B2
final_opt=tf.nn.relu(final_opt)

#损失函数
loss1=tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=final_opt)
loss=tf.reduce_mean(loss1)
#梯度计算并且反向传播
opt=tf.train.GradientDescentOptimizer(0.05).minimize(loss)
init=tf.global_variables_initializer()

#计算预测准确率，将预测值与标签值比较
#argmax（）返回张量最大值索引
#equal（）判断向量是否相等，返回布尔值
correct_prediction=tf.equal(tf.argmax(Y,1),tf.arg_max(final_opt,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
#执行图计算
sess=tf.Session()
sess.run(init)

for i in range(training_iterations):
    #MNIST批数据
    batch=mnist.train.next_batch(batch_size)
    #MNIST输入批数据
    batch_input=batch[0]
    #MNIST标签批数据
    batch_labels=batch[1]
    #训练数据
    training_loss=sess.run([opt,loss],feed_dict={X:batch_input,Y:batch_labels})
    
    if i%1000 == 0:
        train_accuracy=accuracy.eval(session=sess,
                                     feed_dict={X:batch_input,Y:batch_labels})
        print("step=%d,accuracy=%g"%(i,train_accuracy))
#测试集验证       
xdat=sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels})
print('测试结果',xdat)
sess.close()














