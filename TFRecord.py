# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:28:47 2021

@author: 成世杰
"""

# encoding:utf-8
import tensorflow as tf
import os
from glob import glob
import progressbar

class TFRecord():
    def __init__(self, labels, tfrecord_file):
        self.labels = labels
        self.tfrecord_file = tfrecord_file

    def _get_label_with_filename(self, filename):
        basename = os.path.basename(filename).split(".")[0]
        return self.labels[basename]

    def _convert_image(self, img_path, is_train=True):
        label = self._get_label_with_filename(img_path)
        filename = os.path.basename(img_path)
        with tf.gfile.FastGFile(img_path, 'rb') as fid:
            image_str = fid.read()
        if is_train:
            feature_key_value_pair = {
                'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_str])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
        else:
            feature_key_value_pair = {
                'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_str])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[-1]))
            }
        feature = tf.train.Features(feature=feature_key_value_pair)
        example = tf.train.Example(features=feature)
        return example

    def convert_image_folder(self, img_folder):
        img_paths = [img_path for img_path in glob(os.path.join(img_folder, '*'))]

        with tf.python_io.TFRecordWriter(self.tfrecord_file) as tfwriter:
            widgets = ['[INFO] write image to tfrecord: ', progressbar.Percentage(), " ",
                       progressbar.Bar(), " ", progressbar.ETA()]
            pbar = progressbar.ProgressBar(maxval=len(img_paths), widgets=widgets).start()
            for i, img_path in enumerate(img_paths):
                example = self._convert_image(img_path, is_train=True)
                tfwriter.write(example.SerializeToString())
                pbar.update(i)
            pbar.finish()

    def _extract_fn(self, tfrecord):
        # 解码器
        # 解析出一条数据，如果需要解析多条数据，可以使用parse_example函数
        # tf提供了两种不同的属性解析方法：
        ## 1. tf.FixdLenFeature:得到的是一个Tensor
        ## 2. tf.VarLenFeature:得到的是一个sparseTensor，用于处理稀疏数据
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

        sample = tf.parse_single_example(tfrecord, features)
        image = tf.image.decode_jpeg(sample['image'])
        image = tf.image.resize_images(image, (227, 227), method=1)
        label = sample['label']
        filename = sample['filename']
        return [image, label, filename]

    def extract_image(self, shuffle_size,batch_size):
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_fn)
        dataset = dataset.shuffle(shuffle_size).batch(batch_size)
        return dataset