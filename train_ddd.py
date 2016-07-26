
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.visualize_util import plot

from multiprocessing import Pool, Array
import threading
import multiprocessing

import load_data
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
    
    plot(model, to_file='model.png')
    
    return model

'''
    There are 22 unique drivers.
    18 in training set and 
    4 in validation set
'''
def run_cross_validation(numFolds=11):
    frameHeight, frameWidth = read_config('frameHeight'), read_config('frameWidth')
    batchSize = read_config('batchSize')
    numEpochs = 50
    randomState = 51
    patienceFactor = 5
    foldNum = 0

    trainTarget, trainId, driverId, uniqueDrivers = load_data.read_train_targets()
    validationTarget, validationId = load_data.read_validation_targets()
    numUniqueDrivers = 18

    # Model Initialization
    cnnModel = Sequential()
    # Splitting the training data into train set and validation set in k combinations
    kf = KFold(numUniqueDrivers, n_folds=numFolds, shuffle=True, random_state=randomState)
 
    if not os.path.isdir('fold_loss'):
        os.mkdir('fold_loss')
    historyFileName = './fold_loss/history.txt'
    historyFile = open(historyFileName, 'w')
    historyFile.write('Fold, Loss\n')
    historyFile.close()
    
    for train_drivers, test_drivers in kf:
        cnnModel = CNN_model(frameHeight, frameWidth)
        
        uniqueListTrain = [uniqueDrivers[i] for i in train_drivers]
        crossTrainTarget, crossTrainId, crossTrainDriverId = load_data.copy_selected_drivers(trainTarget, trainId, driverId, uniqueListTrain)
        uniqueListValid = [uniqueDrivers[i] for i in test_drivers]
        crossValidTarget, crossValidId, crossValidDriverId = load_data.copy_selected_drivers(trainTarget, trainId, driverId, uniqueListValid)        
        
        print('\nStart KFold number {} from {}'.format(foldNum+1, numFolds))
        print('Split train: ', len(crossTrainTarget))
        print('Split valid: ', len(crossValidTarget))
        print('Train drivers: ', uniqueListTrain)
        print('Test drivers: ', uniqueListValid)
        
        for epoch in range(numEpochs):
            print ('Epoch {} of {}'.format(epoch+1, numEpochs))
