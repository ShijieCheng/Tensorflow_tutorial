# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:06:50 2021

@author: 成世杰
"""
import os
import numpy as np
import cv2
import keras
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D,BatchNormalization,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
resize=224
print(os.listdir("./train/"))
def load_data():
    imgs=os.listdir("./train/")
    train_data=np.empty((5000,resize,resize,3),dtype='int32')
    train_label=np.empty((5000),dtype='int32')
    test_data = np.empty((5000, resize, resize, 3), dtype="int32")
    test_label = np.empty((5000, ), dtype="int32")
    
    print('load training data')
    for i in range(5000):
        if i% 2:
            train_data[i]=cv2.resize(cv2.imread('./train/'+'dog.'+str(i)+'.jpg'),
                      (resize,resize))
            train_label[i]=1
        else:
            train_data[i] = cv2.resize(cv2.imread('./train/' + 'cat.' + str(i) + '.jpg'), 
                      (resize, resize))
            train_label[i] = 0
    
    print("\nload testing data")
    for i in range(5000, 10000):
        if i % 2:
            test_data[i-5000] = cv2.resize(cv2.imread('./train/' + 'dog.' + str(i) + '.jpg'),
                     (resize, resize))
            test_label[i-5000] = 1
        else:
            test_data[i-5000] = cv2.resize(cv2.imread('./train/' + 'cat.' + str(i) + '.jpg'), 
                     (resize, resize))
            test_label[i-5000] = 0
    return train_data, train_label, test_data, test_label


train_data, train_label, test_data, test_label=load_data()
train_data,test_data=train_data.astype('float32'),test_data.astype('float32')
train_data,test_data=train_data/255.0,test_data/255.0
# 变为 one-hot 向量
train_label=keras.utils.to_categorical(train_label,2)
test_label=keras.utils.to_categorical(test_label,2)

# 训练样本目录和测试样本目录
train_dir='./data/train/'
test_dir='./data/validation/'
# 对训练图像进行数据增强
train_pic_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# 对测试图像进行数据增强
test_pic_gen = ImageDataGenerator(rescale=1./255)
# 利用 .flow_from_directory 函数生成训练数据
train_flow=train_pic_gen.flow_from_directory(train_dir,
                                             target_size=(224,224),
                                             batch_size=64,
                                             class_mode='categorical')
# 利用 .flow_from_directory 函数生成测试数据
test_flow = test_pic_gen.flow_from_directory(test_dir,
                                             target_size=(224,224),
                                             batch_size=64,
                                             class_mode='categorical')
print(train_flow.class_indices)
