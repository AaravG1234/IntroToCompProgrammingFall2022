#Aarav Garai

#Sources
#https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0
#https://towardsdatascience.com/pneumonia-detection-using-deep-learning-5dc66468eb9
#https://books.google.com/books?hl=en&lr=&id=ISBKDwAAQBAJ&oi=fnd&pg=PP1&dq=basics+of+expert+models+machine+learning&ots=R9R0Q1lpcA&sig=GfyYBZP8LDyAfZeb9EEXJqOhjD4
#https://www.sciencedirect.com/science/article/pii/B9780128129708000026
#https://colab.research.google.com/drive/1_3rzBjtNn0G4sg0gARLAXddY1a8-vvn0#scrollTo=aSeClkWgIORK

#Importing all the pythonn libraries
import gdown
import zipfile
import wget
import pkg
import helpers

import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import keras.optimizers as optimizers
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet121

#Downloading and loading the given data
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

#receving the data    
def get_train_data(flatten, all_data, metadata, image_shape):
    return get_data_split('train', flatten, all_data, metadata, image_shape)

def get_test_data(flatten, all_data, metadata, image_shape):
    return get_data_split('test', flatten, all_data, metadata, image_shape)
 
def get_field_data(flatten, all_data, metadata, image_shape):
    return get_data_split('field', flatten, all_data, metadata, image_shape)


class Helpers:
#Code to plot the data/results
    def plot_one_image(data, labels = [], index = None, image_shape = [64,64,3]):
        num_dims   = len(data.shape)
        num_labels = len(labels)
#reshaping data
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

#defining accuracies of plot graphs
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

#building the neural networking models
class models:
#first model
    def DenseClassifier(hidden_layer_sizes, nn_params):
        model = Sequential()
        model.add(Flatten(input_shape = nn_params['input_shape']))
        model.add(Dropout(0.5))

        for ilayer in hidden_layer_sizes:
            model.add(Dense(ilayer, activation = 'relu'))
            model.add(Dropout(0.5))
    
        model.add(Dense(units = nn_params['output_neurons'], activation = nn_params['output_activation']))
        model.compile(loss=nn_params['loss'],
                optimizer= optimizers.SGD(learning_rate=1e-4, momentum=0.95),
                metrics=['accuracy'])
        return model
#second model
    def CNNClassifier(num_hidden_layers, nn_params):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=nn_params['input_shape'], padding = 'same', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        for i in range(num_hidden_layers-1):
            model.add(Conv2D(64, (3, 3), padding = 'same', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten()) 

        model.add(Dense(units = 128, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units = 64, activation = 'relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Dropout(0.5))

        model.add(Dense(units = nn_params['output_neurons'], activation = nn_params['output_activation']))

        opt = optimizers.RMSprop(learning_rate=1e-5, decay=1e-6)

        model.compile(loss=nn_params['loss'],
                  optimizer=opt,
                  metrics=['accuracy'])    
        return model
#third model (expert model)
    def TransferClassifier(name, nn_params, trainable = True):
        expert_dict = {'VGG16': VGG16, 
                    'VGG19': VGG19,
                    'DenseNet121':DenseNet121}

        expert_conv = expert_dict[name](weights = 'imagenet', 
                                                include_top = False, 
                                                input_shape = nn_params['input_shape'])
        for layer in expert_conv.layers:
            layer.trainable = trainable
        
        expert_model = Sequential()
        expert_model.add(expert_conv)
        expert_model.add(GlobalAveragePooling2D())

        expert_model.add(Dense(128, activation = 'relu'))
        expert_model.add(Dropout(0.5))

        expert_model.add(Dense(64, activation = 'relu'))
        expert_model.add(Dropout(0.5))

        expert_model.add(Dense(nn_params['output_neurons'], activation = nn_params['output_activation']))

        expert_model.compile(loss = nn_params['loss'], 
                    optimizer = optimizers.SGD(learning_rate=1e-4, momentum=0.9), 
                    metrics=['accuracy'])

        return expert_model

#dataset
metadata_url         = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/metadata.csv"
image_data_url       = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/image_data.npy'
image_data_path      = './image_data.npy'
metadata_path        = './metadata.csv'
image_shape          = (64, 64, 3)

# neural net parameters
nn_params = {}
nn_params['input_shape']       = image_shape
nn_params['output_neurons']    = 1
nn_params['loss']              = 'binary_crossentropy'
nn_params['output_activation'] = 'sigmoid'

#downloading data
url1 = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/metadata.csv"
ur12 = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/image_data.npy"

### pre-loading all data of interest
_all_data = np.load('image_data.npy')
_metadata = pkg.get_metadata(metadata_path, ['train','test','field'])


# downloading and loading data
get_data_split = pkg.get_data_split
get_metadata    = lambda :                 pkg.get_metadata(metadata_path, ['train','test'])
get_train_data  = lambda flatten = False : pkg.get_train_data(flatten = flatten, all_data = _all_data, metadata = _metadata, image_shape = image_shape)
get_test_data   = lambda flatten = False : pkg.get_test_data(flatten = flatten, all_data = _all_data, metadata = _metadata, image_shape = image_shape)
get_field_data  = lambda flatten = False : pkg.get_field_data(flatten = flatten, all_data = _all_data, metadata = _metadata, image_shape = image_shape)

# plotting
plot_one_image = lambda data, labels = [], index = None: helpers.plot_one_image(data = data, labels = labels, index = index, image_shape = image_shape)
plot_acc       = lambda history: helpers.plot_acc(history)

# querying and combining data
model_to_string        = lambda model: helpers.model_to_string(model)
get_misclassified_data = helpers.get_misclassified_data
combine_data           = helpers.combine_data

# models with input parameters
DenseClassifier     = lambda hidden_layer_sizes: models.DenseClassifier(hidden_layer_sizes = hidden_layer_sizes, nn_params = nn_params)
CNNClassifier       = lambda num_hidden_layers: models.CNNClassifier(num_hidden_layers, nn_params = nn_params)
TransferClassifier  = lambda name: models.TransferClassifier(name = name, nn_params = nn_params)

monitor = ModelCheckpoint('./model.h5', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

