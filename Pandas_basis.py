# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:42:51 2021

@author: 成世杰
"""
import pandas as pd
import numpy as np

#创建Series对象

#series=pd.Series([2.3,4,5,1,2,6])
#print(series)
#print(type(series))
#
#series=pd.Series([2.3,4,5,1,2,6],
#                 index=['a','b','c','d','e','f'],
#                 name='this is an index')
#print(series)
#
#series1=pd.Series(np.array([2.3,4,5,1,2,6]),
#                  index=['a','b','c','d','e','f'],
#                  name='this is an index1'         )
#print(series1)

#series2=pd.Series({'beijing':3.,'shanghai':2,'shenzhen':1})
#print(series2)
#print(series2.values)
#print(series2.index)
#print(series2[0:2])
#print(series2['shanghai':'shenzhen'])
#




#DataFrame
list1=[['zhang',23,'m'],['li',24,'f'],['wang',23,'m']]
df1=pd.DataFrame(list1,columns=['name','age','sex'])
print(df1)

df2=pd.DataFrame({'name':['zhang','wang','li'],'age':[23,24,25],
                  'sex':['m','f','m']})
print(df2)

arr1=np.array(list1)
df3=pd.DataFrame(arr1,columns=['name','age','sex'],index=['a','b','c'])
print(df3)

print(df3.values)
print(df3.shape)
print(df3.dtypes)
print(df3.columns)
print(df3.size)
print(df3.index)



