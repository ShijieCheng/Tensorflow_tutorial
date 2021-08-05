# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:59:10 2021

@author: 成世杰
"""

import numpy as np
import pandas as pd


#numpy库基本用法

#a=np.array([[1,2,3,4],[2,3,1,2],[3,5,6,1]])
#b=np.array([10,20,30,40])
#c=a/b
#d=a*b
#e=np.ones((3,2))
#f=np.random.random((4,3))
#print(f)
#print(a.sum(axis=0))
#print(a.sum(axis=1))

#a = np.arange(12).reshape(3,4)
#print(a[-2])
#print(a.max())
#print(np.split(a,3))
#print(np.vsplit(a,3))

#arr1=np.array([1,2,3,4])
#print(arr1)
#arr2=arr1+1
#print(arr2)
#print(arr1[-4:-2])
#
##生成1（10的0次方）到10（10的1次方）20个元素的等比数列
#print(np.logspace(0,1,20))



#pandas库基本用法

#s=pd.Series([1,2,3],index=['a','b','c'])
#print(s)
#data=pd.DataFrame([[1,2,3],[2,3,4]],columns=['a','b','c'])
#print(data)
#data1=pd.read_excel("D:\工作簿1.xlsx")
#print(data1.head(2))

#10个随机数
#print(np.random.random(10))
##生成服从均匀分布的随机数
#print(np.random.rand(3,4))
##生成服从正态分布的随机数
#print(np.random.randn(3,4))
##上下限范围的随机数
#print(np.random.randint(1,5,size=[2,3]))

#arr=np.arange(10)
#print(arr[0:-1:2])#每隔2取一个元素

#创建numpy矩阵
matr1=np.mat("1,2,3;4,5,6")
print(matr1)
matr2=np.matrix([[1,2,3],[4,5,6]])
print(matr2)

#bmat创建分块矩阵
arr1=np.eye(3)
arr2=3*arr1
arr3=np.bmat("arr1,arr2;arr2,arr1")
print(arr3)
print(arr1+arr2)

#矩阵相乘和对应元素相乘
a1=np.mat("1,2,3;4,5,6;7,6,5")
a2=np.mat("3,2,3;4,1,6;4,6,5")

print(a1*a2)
print(np.multiply(a1,a2))

a=np.array([[1,2,3],[2,3,1],[3,5,6]])
b=np.array([[1,2,3],[2,3,1],[3,5,6]])
print(a*b)
print(np.multiply(a,b))




















