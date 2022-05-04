# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 21:04:32 2021

@author: Stephan
"""
import pandas as pd
import numpy as np
import cv2
import math
import h5py
import sys

def readdata(path):
    # read the txt file with 1st column as image link and 2nd as class
    df = pd.read_csv(path, sep= ' ', header=None, names=['img_link', 'class'])
    return df

def create_feature(img_link, fixed_size):
    # read image given the link to the file (where the image is located)
    image = cv2.imread(img_link) 
    if image is not None:
        # resize the image
        image = cv2.resize(image,fixed_size)
    return image

def import_feature(data_ref, fixed_size):    
    image = np.asarray([create_feature(i, fixed_size) for i in data_ref['img_link']])    
    label = data_ref['class']
    return image, label

def load_data(file_name, feature_name, label_name):
    # the images are already read into h5 format
    # feature name:'image_train'/ 'image_val' / 'image_test'
    # label name: 'label_train'/'label_val'/'label_test'
    try:
        f = h5py.File(file_name, 'r')
        image, label = f.get(feature_name)[()], f.get(label_name)[()]
    except:
        print("file or data not found")
        sys.exit()
    del f
    return image, label

def normalize(image, mode='lenet5'): 
    # (x - min) / max
    image -= image.min()
    image = image / image.max()
    # range = [0,1]
    if mode == '0p1':
        return image
    # range = [-1,1]
    elif mode == 'n1p1':
        image = image * 2 - 1
    # range = [-0.1,1.175]   
    elif mode == 'lenet5':
        image = image * 1.275 - 0.1
    return image

def get_one_batch(index, label_tranform, normalize_mode):
    # read one batch of training data
    file_name, feature_link, label_link = "data/{}.h5".format(index), 'image_train', 'label_train'
    batch_image, batch_label = load_data(file_name, feature_link, label_link)
    # normalise featrue and make one hot label
    batch_image = normalize(batch_image, normalize_mode)
    batch_label_ori = batch_label
    batch_label = label_tranform.turn_one_hot(batch_label)
    # ouput- batch_image: nX250X250X3, batch_label: nX50, batch_label_ori:ｎ
    return batch_image, batch_label, batch_label_ori

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

# --------------------------------------------------------------------#
'''
turn the class label(y value) into one hot vector for softmax
'''
class class_label():
    def __init__(self, num_class):                
        # Because total number of class label might not be 50 in batch, need to save the total number of class first
        if type(num_class) is not int: num_class = num_class[0] # fix error for differnet data type
        self.num_class = num_class
        
    def turn_one_hot(self, y_label):
        one_hot_labels = np.zeros((y_label.shape[0], self.num_class)) # create np array of number of data * number of class
        for i in range(y_label.shape[0]):
            one_hot_labels[i, y_label[i]] = 1
        return one_hot_labels

# return random-shuffled mini-batches
# not used because the data set is too big
def random_mini_batches(image, label, mini_batch_size = 256, one_batch=False):
    m = image.shape[0]                  # number of training examples
    mini_batches = []
    
    # Shuffle (image, label)
    permutation = list(np.random.permutation(m))
    shuffled_image = image[permutation,:,:,:]
    shuffled_label = label[permutation]
    
    # extract only one batch
    if one_batch:
        mini_batch_image = shuffled_image[0: mini_batch_size,:,:,:]
        mini_batch_label = shuffled_label[0: mini_batch_size]
        return (mini_batch_image, mini_batch_label)

    # Partition (shuffled_image, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_image = shuffled_image[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_label = shuffled_label[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_image = shuffled_image[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_label = shuffled_label[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    
    return mini_batches

