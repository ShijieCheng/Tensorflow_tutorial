73# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:46:48 2021

@author: 成世杰
"""



#from scipy import io as sio
#a=np.zeros((3,3))
#sio.savemat('file.mat',{'a':a})
#data=sio.loadmat('file.mat',struct_as_record=True)
#print(data['a'])

#from scipy import linalg
#a=np.array([[1,0],[0,4]])
#deta=linalg.det(a)
#print(deta)
#inva=linalg.inv(a)
#print(inva)

#import numpy as np
#import matplotlib.pyplot as plt
#from scipy import fftpack
#
#a=np.zeros(1000)
#a[:100]=1
#a_fft=fftpack.fft(a)
#plt.figure()
#f=np.arange(-500,500,1)
#plt.subplot(221)
#plt.plot(f,abs(a_fft))
#a_fftshift=np.hstack((a_fft[500:],a_fft[:500]))
#plt.subplot(222)
#plt.plot(f,abs(a_fftshift))
#plt.show()

#import numpy as np
#import matplotlib.pyplot as plt
#from scipy import optimize
#
#def f(x):
#    return x**2
#
#x=np.arange(-10,10,0.1)
#optimize.fmin_bfgs(f,0)
#plt.plot(x,f(x))
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#生成正弦函数的数据
#measure_time = np.linspace(0,1,10)
#noise=(np.random.random(10)*2-1)*1e-1
#measures=np.sin(2*np.pi*measure_time)+noise


#构造线性插值函数
linear_interp=interp1d(measure_time,measures)
computed_time=np.linspace(0,1,50)
linear_results=linear_interp(computed_time)

plt.figure()
plt.plot(measure_time,measures,"bo")
plt.plot(computed_time,linear_results,'r*')

#import numpy as np
#import matplotlib.pyplot as plt
#from scipy import stats
#
##1000个正态分布的数
#a=np.random.normal(size=1000)
##统计均值，方差
#loc,std=stats.norm.fit(a)
#print(loc,std)
#
#plt.plot(a)
#plt.show()









