# -*- coding: utf-8 -*-

"""
Created on Wed Apr 28 22:08:28 2021

@author: Stephan
"""

import numpy as np
import pandas as pd
import cv2
import math
import tensorflow as tf

def readdata(path):
    # read the txt file with 1st column as image link and 2nd as class
    df = pd.read_csv(path, header=0)
    return df

def create_imglink(img_link, mode):
    # To handle the file name
    if mode == 'train':
        img_link = 'train_images/train_images/' + img_link
    elif mode == 'test':
        img_link = 'test_images/test_images/' + img_link
    return img_link

def create_feature(img_link, fixed_size, mode):
    # read image given the link to the file (where the image is located)
    img_link = create_imglink(img_link, mode)

    image = cv2.imread(img_link)
    if image is not None:
        # resize the image
        image = cv2.resize(image,fixed_size)
    return image

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_TFRecord(images, labels, filename, fixed_size, mode):
    n_samples = len(labels)
    TFWriter = tf.compat.v1.python_io.TFRecordWriter(filename)

    print('\nTransform start...')
    for i in np.arange(0, n_samples):
        try:
            # Read the image
            image = create_feature(images.iloc[i], fixed_size, mode)

            #圖片轉為字串
            # DeprecationWarning: tostring() is deprecated. Use tobytes() instead.
            image_raw = image.tobytes()

            if mode == 'train':
                label = int(labels.iloc[i])
                # 將 tf.train.Feature 合併成 tf.train.Features
                ftrs = tf.train.Features(
                        feature={'Label': int64_feature(label),
                                 'image_raw': bytes_feature(image_raw)})
            # No label for test data
            elif mode == 'test':
                ftrs = tf.train.Features(
                        feature={'image_raw': bytes_feature(image_raw)})

            # 將 tf.train.Features 轉成 tf.train.Example
            example = tf.train.Example(features=ftrs)

            # 將 tf.train.Example 寫成 tfRecord 格式
            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n')

    TFWriter.close()
    print('{}Transform done!'.format(filename))

# return random-shuffled mini-batches list(with link to the image and the respective class label)
def random_mini_batches_list(image_ref, mini_batch_size = 256, one_batch=False):
    '''
    input: image_ref (list of link to image and class label)
    Data size too big to read all the data at once
    1. shuffle
    2. get the image
    3. return image data
    '''
    m = image_ref.shape[0]    # total number of training examples
    mini_batches = []

    # Shuffle (image, label)
    permutation = list(np.random.permutation(m))
    shuffled = image_ref.iloc[permutation]

    # extract only one batch
    if one_batch:
        mini_batch_ref = shuffled.iloc[0: mini_batch_size]
        return mini_batch_ref

    # Partition (shuffled_image, shuffled_Y). Except the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch = shuffled.iloc[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch = shuffled.iloc[num_complete_minibatches * mini_batch_size : m]
        mini_batches.append(mini_batch)
    return mini_batches
small_files = False
fixed_size  = tuple((512,512)) # size for image, no resize

# read the reference data file, column name = ['img_link', 'class']
train_ref_all, test_ref= readdata("train.csv"), readdata("test.csv")

# 把train data 打亂
m = train_ref_all.shape[0]    # total number of training examples
permutation = list(np.random.permutation(m))
train_ref_all = train_ref_all.iloc[permutation]

# split into train and validation
msk = np.random.rand(m) < 0.9

train_ref = train_ref_all.iloc[msk]
val_ref = train_ref_all.iloc[~msk]

print(train_ref.shape[0])
print(val_ref.shape[0])

if small_files is True:
    # Optional: Seperate into 6 files for testing data
    test_ref_list = random_mini_batches_list(test_ref, mini_batch_size = 1700, one_batch=False)

    for i in range(len(test_ref_list)):
        filename = 'mydata/test{}.tfrecords'.format(i)
        convert_to_TFRecord(test_ref_list[i]['ID'],test_ref_list[i]['Label'], filename, fixed_size, mode = 'test')


# -----------------------For training-------------------#
# read the images of entire train data
filename = 'mydatatest/training.tfrecords'
convert_to_TFRecord(train_ref['ID'],train_ref['Label'], filename, fixed_size, mode = 'train')

# -----------------------For validation-------------------#
# read the images of entire train data
filename = 'mydatatest/val.tfrecords'
convert_to_TFRecord(val_ref['ID'],val_ref['Label'], filename, fixed_size, mode = 'train')


# -----------------------For testing-------------------#
# read the images of entire test data
filename = 'mydata/test.tfrecords'
convert_to_TFRecord(test_ref['ID'],test_ref['Label'], filename, fixed_size, mode = 'test')
