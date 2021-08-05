# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:23:43 2021

@author: 成世杰
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


#    x=np.linspace(0,10,100)
#    y=np.cos(2*x)
#    z=np.sin(x**2)

#    plt.figure(figsize=(8,4),dpi=100)
#    plt.plot(x,y,label='$cos(2*x)$',color='red',linewidth=2)
#    plt.plot(x,z,'b--',label='$sin(x^2)$')
#    plt.xlabel("Time(s)")
#    plt.ylabel("Volt")
#    plt.title("the plt example")
#    plt.ylim(-1.3,1.3)
#    plt.legend()
#    
#    plt.savefig("first.jpg",dpi=200)
#    plt.show()



#for idx,color in enumerate('bgrcmk'):
#    plt.subplot(2,3,0+idx+1,axisbg=color)
#plt.show()

#plt.figure(1)
#plt.figure(2)
#ax1=plt.subplot(211)
#ax2=plt.subplot(212)
#x=np.linspace(0,10,100)
#
#for i in range(5):
#    #选择图标一
#    plt.figure(1)
#    plt.plot(x,np.sin(i*x/6))
#    
#    plt.sca(ax1)
#    plt.plot(x,np.sin(i*x/2))
#    plt.sca(ax2)
#    plt.plot(x,np.cos(i*x/2))
#plt.show()


#fig=plt.figure()
#ax=fig.add_subplot(111,projection='3d')
#X=[1,3,2,4]
#Y=[3,4,2,6]
#Z=[0,2,1,9]
#ax.plot_trisurf(X,Y,Z)
#plt.show()

#fig=plt.figure()
#ax=Axes3D(fig)
#x=np.arange(-5,5,0.25)
#y=np.arange(-5,5,0.25)
#X,Y=np.meshgrid(x,y)
#R=np.sqrt(X**2+Y**2)
#Z=np.sin(R)
#ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
#ax.contour(X,Y,Z,zdim='z',offset=-2,cmap='rainbow')
#ax.set_zlim(-3,3)
#plt.show()


plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
data=np.load('D:\Spyder-Code\Tensoeflow-code\Python数据分析与应用\第3章\任务程序\data\国民经济核算季度数据.npz',
             allow_pickle=True)

#数据标签
name=data['columns']
#数据的存在位置
values=data['values']
print(values[-1,3:6])
#散点图

#plt.figure(figsize=(8,7))
##plt.scatter(values[:,0],values[:,2],marker='o')
#plt.plot(values[:,0],values[:,2],linestyle='--',marker='o')
#
#plt.xlabel('年份');plt.ylabel('生产总值（亿元）')
#
#plt.ylim((0,225000))
##换成自己想要的标签
#plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
#plt.title('2000-2017各季度国民生产总值')
#plt.show()

#plt.figure(figsize=(8,7))
#plt.scatter(values[:,0],values[:,3],marker='o',c='red')
#plt.scatter(values[:,0],values[:,4],marker='D',c='blue')
#plt.scatter(values[:,0],values[:,5],marker='v',c='yellow')
#
#plt.xlabel('年份');plt.ylabel('生产总值（亿元）')
#plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
#plt.title('2000-2017各季度各产业国民生产总值')
#plt.legend(['第一产业','第二产业','第三产业'])
#plt.show()

#折线图
#plt.figure(figsize=(8,7))
#plt.plot(values[:,0],values[:,3],'bs-',
#         values[:,0],values[:,4],'ro-',
#         values[:,0],values[:,5],'gH--')
#
#plt.xlabel('年份');plt.ylabel('生产总值（亿元）')
#
#plt.ylim((0,225000))
##换成自己想要的标签
#plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
#plt.title('2000-2017各季度国民生产总值')
#plt.legend(['第一产业','第二产业','第三产业'])
#plt.show()


#子图
#plt.figure(figsize=(8,7))
#
#plt.subplot(2,1,1)
#plt.scatter(values[:,0],values[:,3],marker='o',c='red')
#plt.scatter(values[:,0],values[:,4],marker='D',c='blue')
#plt.scatter(values[:,0],values[:,5],marker='v',c='yellow')
#plt.legend(['第一产业','第二产业','第三产业'])
#
#plt.subplot(212)
#plt.scatter(values[:,0],values[:,6],marker='o',c='red')
#plt.scatter(values[:,0],values[:,7],marker='D',c='blue')
#plt.scatter(values[:,0],values[:,8],marker='v',c='yellow')
#plt.scatter(values[:,0],values[:,9],marker='8',c='g')
#plt.scatter(values[:,0],values[:,10],marker='p',c='c')
#plt.scatter(values[:,0],values[:,11],marker='+',c='m')
#plt.scatter(values[:,0],values[:,12],marker='s',c='k')
#plt.scatter(values[:,0],values[:,13],marker='*',c='purple')
#plt.legend(['农业','工业','建筑','批发','交通','餐饮','金融','房地产'])
#
#plt.xlabel('年份');plt.ylabel('生产总值（亿元）')
#plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
#plt.title('2000-2017各季度各产业国民生产总值')
#plt.show()


#直方图
#label=['第一产业','第二产业','第三产业']
#plt.bar(range(3),values[-1,3:6],width=0.5)
#plt.xticks(range(3),label)
#plt.xlabel('产业');plt.ylabel('生产总值（亿元）')
#plt.title('2017各产业国民生产总值')
#plt.show()

#饼图
#plt.figure(figsize=(6,6))
#label=['第一产业','第二产业','第三产业']
#explode=[0.01,0.01,0.01]
#plt.pie(values[-1,3:6],explode=explode,labels=label,
#        autopct='%1.1f%%')
#plt.title('2017各产业国民生产总值')
#plt.show()


#箱线图
label=['第一产业','第二产业','第三产业']
gdp=(list(values[:,3]),list(values[:,4]),list(values[:,5]))
plt.boxplot(gdp,notch=True,labels=label,meanline=True)
plt.title('2000-2017各产业国民生产总值')
plt.show()
















