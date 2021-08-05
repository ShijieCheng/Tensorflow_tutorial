# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 22:58:14 2021

@author: 成世杰
"""

import random
import numpy as np
import matplotlib.pyplot as plt

#最大迭代次数
max_epcho=5000

#define a Perceptron感知机
class My_Per:
    def __init__(self,lr=0.2):
        self.lr=lr
        #偏置
        self.b=random.random()
        #权值矢量矩阵
        self.w=np.random.random(2)*(1-(-1))+(-1)
        
    #预测函数   
    def predict(self,Px):
        number=self.w[0]*Px[0]+self.w[1]*Px[1]+self.b
        return np.where(number >= 0,1,0)
    
    #训练函数
    def train(self,Px,t):
        #修正矢量
        update=self.lr*(t-self.predict(Px))
        #更新b
        self.b+=update
        #更新w
        self.w[0]+=update*Px[0]
        self.w[1]+=update*Px[1]
     
    
def main():
    #input
    P=[[-0.5,-0.5],[-0.5,0.5],[0.3,-0.5],[0,1]]
    #target
    T=[1,1,0,0]
    my_per=My_Per(0.1)
    
    plt.figure()
    plt.plot([-0.5,-0.5],[-0.5,0.5],'bo')
    plt.plot([0.3,0],[-0.5,1],'r.')
    
    #迭代
    for i in range(max_epcho):
        for i in range(4):
            Px=P[i]
            t=T[i]
            my_per.train(Px,t)
    
    x=np.arange(-1,2)
    y=-my_per.w[0]/my_per.w[1]*x-my_per.b/my_per.w[1]
    plt.plot(x,y)
    plt.show()
    print("b=",my_per.b)
    print('w[0]',my_per.w[0])
    print('w[1]',my_per.w[1])
       
if __name__ == "__main__":
    main()
    
    




      
        
        
        
        
        
        
        
        
        
        
        
        
        
        