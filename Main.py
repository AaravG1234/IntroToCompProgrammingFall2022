#Aarav Garai

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
import matplotlib.pyplot as plt
import seaborn as sns

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


class Helpers:
#Plotting the data/results
    def plot_one_image(data, labels = [], index = None, image_shape = [64,64,3]):
        num_dims   = len(data.shape)
        num_labels = len(labels)

        if num_dims == 1:
            #data = data.reshape(target_shape)
            pass
        if num_dims == 2:
            data = data.reshape(np.vstack[-1, image_shape])
            num_dims   = len(data.shape)

    # check if single or multiple images
        if num_dims == 3:
            if num_labels > 1:
                print('Multiple labels does not make sense for single image.')
                return

            label = labels      
            if num_labels == 0:
                label = ''
            image = data

        if num_dims == 4:
            image = data[index, :]
            label = labels[index]

    # plot image of interest
        print('Label: %s'%label)
        plt.imshow(image)
        plt.show()

    def get_misclassified_data(data, labels, predictions):
        missed_index     = np.where(np.abs(predictions.squeeze() - labels.squeeze()) > 0)[0]
        missed_labels    = labels[missed_index]
        missed_data      = data[missed_index,:]
        predicted_labels = predictions[missed_index]
        return missed_data, missed_labels, predicted_labels, missed_index

    def combine_data(data_list, labels_list):
        return np.concatenate(data_list, axis = 0), np.concatenate(labels_list, axis = 0)

    def model_to_string(model):
        import re
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        sms = "\n".join(stringlist)
        sms = re.sub('_\d\d\d','', sms)
        sms = re.sub('_\d\d','', sms)
        sms = re.sub('_\d','', sms)  
        return sms

    def plot_acc(history, ax = None, xlabel = 'Epoch #'):
        history = history.history
        history.update({'epoch':list(range(len(history['val_accuracy'])))})
        history = pd.DataFrame.from_dict(history)

        best_epoch = history.sort_values(by = 'val_accuracy', ascending = False).iloc[0]['epoch']

        if not ax:
            f, ax = plt.subplots(1,1)
            sns.lineplot(x = 'epoch', y = 'val_accuracy', data = history, label = 'Validation', ax = ax)
            sns.lineplot(x = 'epoch', y = 'accuracy', data = history, label = 'Training', ax = ax)
            ax.axhline(0.5, linestyle = '--',color='red', label = 'Chance')
            ax.axvline(x = best_epoch, linestyle = '--', color = 'green', label = 'Best Epoch')  
            ax.legend(loc = 4)    
            ax.set_ylim([0.4, 1])

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Accuracy (Fraction)')
        
        plt.show()


