# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:36:13 2021

@author: 成世杰
"""

##内置函数
#a=str(10)
#b=int(10)
#c=list([1,2,3,4])
#c=4.2315
#ab=['one','two','three']
##枚举返回索引和当前值
#for i,j in enumerate(ab):
#    print(i,j)
#    
#list1=['a','b','c']
#list2=[1,2,3]
##返回一个元组
#print(list(zip(list1,list2)))

#import math
##向下向上取整
#print(math.floor(4.7))
#print(math.ceil(4.7))

#自定义函数
#def avg(x):
#    return sum(x)/len(x)
#
#def ssn(n,beg=1):#第二个为默认参数
#    s=0
#    for i in range(beg,n+1):
#        s += i
#    return s
#
#
#print(avg([1,4,2,5,7]))
#print(ssn(100,2))
#print(ssn(10))

#高级函数

#g = lambda x,y:x**2+y#参数，返回值
#print(g(2,2))
#
#f1=lambda x:'A' if x==1 else 'B'
#print(f1(1.00))

#items=[1,2,3,4,5,6]
#def f(x):
#    return x**2
##map(function,input)#输出映射函数后的序列
#print(list(map(f,items)))


#from functools import reduce
##元素逐步迭代加工
#def f(x,y):
#    return x+y
#items=[1,2,3,4,5,6,7,8,9]
#result=reduce(f,items)
#print(result)
#
#def str_add(x,y):
#    return x+y
#items1=['a','b','sw']
#result=reduce(str_add,items1)
#print(result)

print(list(filter(lambda x:x%3==0,range(21))))
items1=['a','b','sw',1,2,3]
print(list(filter(lambda x: 1 if isinstance(x,int) else 0,items1)))



















