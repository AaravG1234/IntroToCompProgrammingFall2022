#Sources
#https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0
#https://towardsdatascience.com/pneumonia-detection-using-deep-learning-5dc66468eb9
#https://books.google.com/books?hl=en&lr=&id=ISBKDwAAQBAJ&oi=fnd&pg=PP1&dq=basics+of+expert+models+machine+learning&ots=R9R0Q1lpcA&sig=GfyYBZP8LDyAfZeb9EEXJqOhjD4
#https://www.sciencedirect.com/science/article/pii/B9780128129708000026


'''
import gdown
import zipfile

import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

#from sklearn.metrics import accuracy_score, confusion_matrix

#import tensorflow as keras
import keras.optimizers as optimizers
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
#from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121

'''
import pandas as pd
import numpy as np

class pkg:
#Downloading//loading the data
    def get_metadata(metadata_path, which_splits = ['train', 'test']):  
        metadata = pd.read_csv(metadata_path)
        keep_idx = metadata['split'].isin(which_splits)
        return metadata[keep_idx]

    def get_data_split(split_name, flatten, all_data, metadata, image_shape):
        sub_df = metadata[metadata['split'].isin([split_name])]
        index  = sub_df['index'].values
        labels = sub_df['class'].values
        data = all_data[index,:]
        if flatten:
            data = data.reshape([-1, np.product(image_shape)])
            return data, labels
    
    def get_train_data(flatten, all_data, metadata, image_shape):
        return get_data_split('train', flatten, all_data, metadata, image_shape)

    def get_test_data(flatten, all_data, metadata, image_shape):
         return get_data_split('test', flatten, all_data, metadata, image_shape)
 
    def get_field_data(flatten, all_data, metadata, image_shape):
        return get_data_split('field', flatten, all_data, metadata, image_shape)
