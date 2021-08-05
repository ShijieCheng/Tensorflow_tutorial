# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 19:03:37 2021

@author: 成世杰
"""

import h5py
import numpy as np
#HDF5的写入
imgData=np.zeros((30,3,128,256))
f=h5py.File('HDF5_FILE.h5','w')
f['data']=imgData
f['labels']=range(100)
f.close()

#读取
f=h5py.File('test_imgs_v2.hdf5','r')
for key in f.keys():
    print(f[key].name)
    if f[key].name == '/test_labels':
        print(f[key].value[0])
            