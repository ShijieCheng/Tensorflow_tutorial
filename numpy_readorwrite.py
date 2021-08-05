# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:56:31 2021

@author: 成世杰
"""
import numpy as np
#arr=np.arange(100).reshape(10,10)
#np.save("numpy_savearr",arr)
#print(arr)

#多个数组存储
#a1=np.array([[1,2,3],[4,3,2]])
#a2=np.arange(0,2,0.2)
#np.savez('numpy_savetwoarr',a1,a2)
#
##读取
#load_data=np.load("numpy_savearr.npy")
#print(load_data)
#
#loaded_data=np.load('numpy_savetwoarr.npz')
#print(loaded_data['arr_1'])

#文件的存储与读取
#arr=np.arange(0,12.0,0.5,dtype='float').reshape(4,-1)
#np.savetxt('arrtext.txt',arr,fmt='%d',delimiter=',,')
#load_data=np.loadtxt('arrtext.txt',delimiter=',,')
#print(load_data)

#排序
#np.random.seed(42)
#arr=np.random.randint(1,10,size=10)
#print(arr)
#arr.sort()
#print(arr)

#arr=np.random.randint(1,10,size=(3,3))
#print(arr)
#arr.sort(axis=0)
#print(arr)

#arr=np.array([2,3,6,8,0,7])
#print(arr.argsort())
#
#a=np.array([3,2,6,4,5])
#b=np.array([50,20,70,50,40])
#c=np.array([300,200,700,900,500])
#d=np.lexsort((a,b,c))
#print(list(zip(a[d],b[d],c[d])))


#去重与重复
#names=np.array(['1','1','2','3'])
#print(np.unique(names))
#
#arr=np.arange(1,5).reshape(2,2)
#print(np.tile(arr,3))#对数组重复

#np.random.seed(42)
#arr=np.random.randint(0,10,size=(3,3))
#print(arr.repeat(2,axis=1))#对数组元素重复

#length=np.loadtxt('D:\Spyder-Code\Tensoeflow-code\Python数据分析与应用\第2章\任务程序\data\iris_sepal_length.csv',
#                  delimiter=',')
#print(length)
#length.sort()
#print('排序后',length)
##去重
#print(np.unique(length))
#np.savetxt('D:\Spyder-Code\Tensoeflow-code\Python数据分析与应用\第2章\任务程序\data\mmm.csv',
#           np.unique(length),fmt='%f',delimiter=',')

a=np.ones([6,6])

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if (i%2==1 and j%2==0) or (i%2==0 and j%2==1):
            print(i+1,j+1)
            a[i,j]=0
print(a)













