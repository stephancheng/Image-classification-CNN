# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 22:08:28 2021

@author: Stephan
"""

import numpy as np
import mydata
import tensorflow as tf

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_TFRecord(images, labels, filename, fixed_size):
    n_samples = len(labels)
    TFWriter = tf.compat.v1.python_io.TFRecordWriter(filename)

    print('\nTransform start...')
    for i in np.arange(0, n_samples):
        try:
            image = mydata.create_feature(images.iloc[i], fixed_size)
            label = int(labels.iloc[i])

            #圖片轉為字串
            # DeprecationWarning: tostring() is deprecated. Use tobytes() instead.
            image_raw = image.tobytes()

            # 將 tf.train.Feature 合併成 tf.train.Features
            ftrs = tf.train.Features(
                    feature={'Label': int64_feature(label),
                             'image_raw': bytes_feature(image_raw)})

            # 將 tf.train.Features 轉成 tf.train.Example
            example = tf.train.Example(features=ftrs)

            # 將 tf.train.Example 寫成 tfRecord 格式
            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n')

    TFWriter.close()
    print('{}Transform done!'.format(filename))

Small_files = False
# all the data preprocessing function are stored in the file "mydata.py"
fixed_size  = tuple((250,250)) # size for image

# read the reference data filr, column name = ['img_link', 'class']
train_ref, val_ref, test_ref= mydata.readdata("train.txt"), mydata.readdata("val.txt"), mydata.readdata("test.txt")

if Small_files is True:
    # Optional: Seperate into 6 files so that each file is smaller
    train_ref_list = mydata.random_mini_batches_list(train_ref, mini_batch_size = 10600, one_batch=False)

    # -----------------------For training-------------------#

    for i in range(len(train_ref_list)):
        filename = 'mydata/Train{}.tfrecords'.format(i) # Output file name
        convert_to_TFRecord(train_ref_list[i]['img_link'],train_ref_list[i]['class'], filename, fixed_size)
else:
    # 把train data 打亂
    m = train_ref.shape[0]    # total number of training examples
    permutation = list(np.random.permutation(m))
    train_ref = train_ref.iloc[permutation]

    # -----------------------For training-------------------#

    filename = 'mydata/train.tfrecords' # Output file name
    # read the images of entire train data
    convert_to_TFRecord(train_ref['img_link'],train_ref['class'], filename, fixed_size)

# -----------------------For validation-------------------#
filename = 'mydata/val.tfrecords' # Output file name
# read the images of entire val data
convert_to_TFRecord(val_ref['img_link'],val_ref['class'], filename, fixed_size)

# -----------------------For testing-------------------#

filename = 'mydata/test.tfrecords' # Output file name
# read the images of entire test data
convert_to_TFRecord(test_ref['img_link'],test_ref['class'], filename, fixed_size)
