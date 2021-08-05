# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 22:48:43 2021

@author: 成世杰
"""
import tensorflow as tf 
tf.enable_eager_execution()
def extract_fn(data_record):
    features = {
        'int_list':tf.FixedLenFeature([],tf.int64),
        'float_list':tf.FixedLenFeature([],tf.float32),
        'str_list':tf.FixedLenFeature([],tf.string),
        # 如果不同的record中的大小不一样，则使用VarLenFeature
        'float_list2':tf.VarLenFeature(tf.float32)
    }
    sample = tf.parse_single_example(data_record,features)
    return sample
# 使用dataset模块读取数据
dataset = tf.data.TFRecordDataset(filenames=['example.tfrecord'])
# 对每一条record进行解析
dataset = dataset.map(extract_fn)
iterator = dataset.make_one_shot_iterator()
next_example = iterator.get_next()


# eager 模式下
#tf.enable_eager_execution()
try:
    while True:
        next_example = iterator.get_next()
        print(next_example)
except:
    pass

# 非eager模式
with tf.Session() as sess:
    try:
        while True:
            data_record = sess.run(next_example)
            print(data_record)
    except:
        pass