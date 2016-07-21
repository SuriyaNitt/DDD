import cv2
import numpy as np
import os
import glob
import hickle
import warnings

warnings.filterwarnings("ignore")

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from keras.regularizers import l1, l2, l1l2, activity_l1, activity_l2, activity_l1l2
from keras.utils.visualize_util import plot

from keras import backend as K
from subprocess import call

from multiprocessing import Pool, Array
import threading
import multiprocessing

from load_avi import load_avi_into nparray
from read_config import read_config

np.random.seed(2016)

'''
    main CNN model
    Input
     |
    Conv3x3_32
     |
    Conv3x3_64
     |
    Conv3x3_128
     |
    FullyConnectedLayer_32
     |
    FullyConnectedLayer_2
'''

def CNN_model(frameHeight, frameWidth):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    
    model.add(Dense(32, W_regularizer=l2(1.26e-7)))
    model.add(Activation('relu'))     

    model.add(Dense(2, W_regularizer=l2(1e-0)))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')
    return model

'''
    There are 22 unique drivers.
    18 in training set and 
    4 in validation set
'''
def run_cross_validation(nfolds=11):
    frameHeight, frameWidth = read_config('frameHeight'), read_config('frameWidth')


