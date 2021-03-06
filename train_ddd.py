import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import scipy

from keras import backend as K
#from multiprocessing import Pool, Array
#import threading
#import multiprocessing

import math
import load_data
from read_config import read_config

class TrainingHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))

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
                            input_shape=(1, int(frameHeight), int(frameWidth))))
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

def two_inputs_cnn_model(frameHeight1, frameWidth1, frameHeight2, frameWidth2):

    # Convolutional stack for full frame
    model1 = Sequential()
    model1.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu',
                            input_shape=(1, int(frameHeight1), int(frameWidth1))))
    model1.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.1))

    model1.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.2))

    model1.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.2))

    model1.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(8, 8)))
    model1.add(Dropout(0.2))

    # Convolutional stack for face ROI
    model2 = Sequential()
    model2.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu',
                            input_shape=(1, int(frameHeight2), int(frameWidth2))))
    model2.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.1))

    model2.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))

    model2.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))

    model2.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(8, 8)))
    model2.add(Dropout(0.2))

    # FC stack with merged convolutional stacks
    model = Sequential()
    model.add(Merge([model1, model2], mode='concat', concat_axis=-1))

    model.add(Flatten())

    model.add(Dense(4096, W_regularizer=l2(1e-6)))
    model.add(Activation('relu'))

    model.add(Dense(1024, W_regularizer=l2(1e-5)))
    model.add(Activation('relu'))

    model.add(Dense(32, W_regularizer=l2(1e-4)))
    model.add(Activation('relu'))

    model.add(Dense(2, W_regularizer=l2(1e-0)))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=["accuracy"])

    plot(model, to_file='model.png')

    return model

def two_inputs_cnn_rnn_model(frameHeight1, frameWidth1, frameHeight2, frameWidth2):

    # Convolutional stack for full frame
    model1 = Sequential()
    model1.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu',
                            input_shape=(1, int(frameHeight1), int(frameWidth1))))
    model1.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.1))

    model1.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.2))

    model1.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.4))

    model1.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(8, 8)))
    #model1.add(Dropout(0.8))

    # Convolutional stack for face ROI
    model2 = Sequential()
    model2.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu',
                            input_shape=(1, int(frameHeight2), int(frameWidth2))))
    model2.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.1))

    model2.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))

    model2.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))

    model2.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(8, 8)))
    model2.add(Dropout(0.2))

    # FC stack with merged convolutional stacks
    model = Sequential()
    model.add(Merge([model1, model2], mode='concat', concat_axis=-1))
    model.add(Reshape((512, 4)))
    '''
    model.add(Flatten())

    model.add(Dense(4096, W_regularizer=l2(1e-6)))
    model.add(Activation('relu'))

    model.add(Dense(1024, W_regularizer=l2(1e-5)))
    model.add(Activation('relu'))

    model.add(Dense(32, W_regularizer=l2(1e-4)))
    model.add(Activation('relu'))

    model.add(Dense(2, W_regularizer=l2(1e-0)))
    model.add(Activation('softmax'))
    '''

    model.add(LSTM(32))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=["accuracy"])

    plot(model, to_file='model.png')

    return model

def corr(inputs):
    prev = inputs[0]
    curr = inputs[1]
    output = prev * curr
    output = output.sum(axis=2)
    output = output.sum(axis=2)
    print output.shape
    output = K.reshape(output, (output.shape[0], output.shape[1], 1))
    return output

def corr_shape(input_shape):
    'Merge output shape'
    shape = list(input_shape)
    outshape = (shape[0][0], shape[0][1], 1)
    return tuple(outshape)

def corr_model(frameHeight1, frameWidth1, frameHeight2, frameWidth2):

    # Convolutional stack for full frame
    model1 = Sequential()
    model1.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',
                            input_shape=(1, int(frameHeight1), int(frameWidth1))))
    model1.add(PReLU())
    model1.add(BatchNormalization(mode=2))
    model1.add(MaxPooling2D(pool_size=(2, 2)))

    model1.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model1.add(PReLU())
    model1.add(BatchNormalization(mode=2))
    model1.add(MaxPooling2D(pool_size=(2, 2)))

    model1.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model1.add(PReLU())
    model1.add(BatchNormalization(mode=2))
    model1.add(MaxPooling2D(pool_size=(2, 2)))

    model1.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal'))
    model1.add(PReLU())
    model1.add(BatchNormalization(mode=2))
    model1.add(MaxPooling2D(pool_size=(4, 4)))

    model1.add(Convolution2D(512, 3, 3, border_mode='same', init='he_normal'))
    model1.add(PReLU())
    model1.add(BatchNormalization(mode=2))
    model1.add(MaxPooling2D(pool_size=(4, 4)))

    model1.add(Reshape((512, 1)))

    # Convolutional stack for face ROI
    model2 = Sequential()
    model2.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',
                            input_shape=(1, int(frameHeight2), int(frameWidth2))))
    model2.add(PReLU())
    model2.add(BatchNormalization(mode=2))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model2.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model2.add(PReLU())
    model2.add(BatchNormalization(mode=2))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model2.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model2.add(PReLU())
    model2.add(BatchNormalization(mode=2))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model2.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal'))
    model2.add(PReLU())
    model2.add(BatchNormalization(mode=2))
    model2.add(MaxPooling2D(pool_size=(8, 8)))

    # Convolutional stack for face ROI
    model3 = Sequential()
    model3.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',
                            input_shape=(1, int(frameHeight2), int(frameWidth2))))
    model3.add(PReLU())
    model3.add(BatchNormalization(mode=2))
    model3.add(MaxPooling2D(pool_size=(2, 2)))

    model3.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model3.add(PReLU())
    model3.add(BatchNormalization(mode=2))
    model3.add(MaxPooling2D(pool_size=(2, 2)))

    model3.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model3.add(PReLU())
    model3.add(BatchNormalization(mode=2))
    model3.add(MaxPooling2D(pool_size=(2, 2)))

    model3.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal'))
    model3.add(PReLU())
    model3.add(BatchNormalization(mode=2))
    model3.add(MaxPooling2D(pool_size=(8, 8)))

    # FC stack with merged convolutional stacks
    model4 = Sequential()
    model4.add(Merge([model2, model3], mode=corr, output_shape=corr_shape))

    merged = Merge([model1, model4], mode='concat', concat_axis=1)
    
    modelForward = Sequential()
    modelForward.add(merged)
    modelForward.add(LSTM(32))

    modelBackward = Sequential()
    modelBackward.add(merged)
    modelBackward.add(LSTM(32, go_backwards=True))

    # FC stack with merged convolutional stacks
    model = Sequential()
    #model.add(merged)
    model.add(Merge([modelForward, modelBackward], mode='concat', concat_axis=1))

    #model.add(LSTM(32))
    #model.add(BatchNormalization(mode=2))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=1e-3), loss='binary_crossentropy', metrics=["accuracy"])

    plot(model, to_file='model.png')

    return model

def train(model, crossTrainTarget, crossTrainId, epoch):
    frameHeight, frameWidth = read_config('frameHeight'), read_config('frameWidth')
    frameHeightFace, frameWidthFace = read_config('frameHeightFace'), read_config('frameWidthFace')
    batchSize = read_config('batchSize')
    miniBatchSize = read_config('miniBatchSize')
    miniNumEpochs = read_config('miniNumEpochs')
    verbose = read_config('verbose')
    debugMode = read_config('debugMode')
    modelNo = read_config('modelNo')

    numBatches = math.ceil(len(crossTrainTarget)/batchSize)
    batchCount = 0
    itemsDone = 0
    xTrain, yTrain = [], []
    xFace = []
    xTrainFull = []
    batchList = []
    history = TrainingHistory()

    #plotting
    #plt.figure(figsize=(6, 3))
    '''
    plt.axis([0, numBatches, 0, 20])
    plt.ylabel('error')
    plt.xlabel('iteration')
    plt.title('training error')
    plt.ion()
    yPlotTrainLoss = []
    yPlotTrainAccuracy = []
    xPlot = []
    '''

    while itemsDone < len(crossTrainTarget):
        print('\nTraining Batch {} of {} and epoch {}'.format(batchCount+1, numBatches, epoch))
        if(len(crossTrainTarget) - itemsDone) < batchSize:
            batchList = crossTrainId[itemsDone:]
            yTrain = crossTrainTarget[itemsDone:]
            itemsDone = len(crossTrainTarget)
        else:
            batchList = crossTrainId[itemsDone:itemsDone+batchSize]
            yTrain = crossTrainTarget[itemsDone:itemsDone+batchSize]
            itemsDone += batchSize

        yTrain = np_utils.to_categorical(yTrain, 2)
        xTrain = load_data.read_train_data_avi(frameHeight, frameWidth, batchList)
        # zero mean
        mean = np.mean(xTrain, axis=0)
        xTrain = xTrain - mean
        xTrain = np.array(xTrain, dtype='int8')
        xTrain /= 127
        if modelNo == 1:
            xFace = load_data.read_train_data_img(frameHeightFace, frameWidthFace, batchList)
            # zero mean
            mean = np.mean(xFace, axis=0)
            xFace = xFace - mean
        elif modelNo == 2:
            xFace = load_data.read_train_data_img(frameHeightFace, frameWidthFace, batchList)
            #xFace1 = xFace
            xFace1 = np.ones((1, 1, frameHeightFace, frameWidthFace), dtype=xFace.dtype)
            xFace1 = np.append(xFace1, xFace[:-1, :, :, :], axis=0)

            # zero mean
            mean = np.mean(xFace, axis=0)
            xFace = xFace - mean
            mean = np.mean(xFace1, axis=0)
            xFace1 = xFace1 - mean
            xFace = np.array(xFace, dtype='int8')
            xFace1 = np.array(xFace1, dtype='int8')
            xFace /= 127
            xFace1 /= 127
            xTrainFull = [xTrain, xFace1, xFace]
        if modelNo == 0:
            xTrainFull = xTrain
        elif modelNo == 1:
            xTrainFull = [xTrain, xFace]

        print('Loaded data')
        if debugMode:
            print('xTrain shape:{}'.format(xTrainFull[0].shape + xTrainFull[1].shape + xTrainFull[2].shape))
            print('yTrain shape:{}'.format(yTrain.shape))

        model.fit(xTrainFull, yTrain, miniBatchSize, miniNumEpochs, verbose, callbacks=[history])
        # plotting
        loss = np.array(history.losses)
        accuracy = np.array(history.accuracy)
        '''
        xPlot.append(batchCount)
        yPlotTrainLoss.append(np.mean(loss))
        yPlotTrainAccuracy.append(0.69)
        plt.plot(xPlot, yPlotTrainLoss)
        plt.plot(xPlot, yPlotTrainAccuracy)
        plt.pause(0.05)
        '''

        batchCount += 1

    print('Training Epoch done')
    return model, history

def cross_validate(model, crossValidTarget, crossValidId, epoch):
    frameHeight, frameWidth = read_config('frameHeight'), read_config('frameWidth')
    frameHeightFace, frameWidthFace = read_config('frameHeightFace'), read_config('frameWidthFace')
    batchSize = read_config('batchSize')
    miniBatchSize = read_config('miniBatchSize')
    verbose = read_config('verbose')
    modelNo = read_config('modelNo')

    numBatches = math.ceil(len(crossValidTarget)/batchSize)
    itemsDone = 0
    batchCount = 0
    xValid, yValid = [], []
    xFace = []
    xValidFull = []
    batchList = []
    validationScore = 0

    while itemsDone < len(crossValidTarget):
        print('\nCross Validation Batch {} of {} and epoch {}'.format(batchCount+1, numBatches, epoch))
        if(len(crossValidTarget) - itemsDone) < batchSize:
            batchList = crossValidId[itemsDone:]
            yValid = crossValidTarget[itemsDone:]
            itemsDone = len(crossValidTarget)
        else:
            batchList = crossValidId[itemsDone:itemsDone+batchSize]
            yValid = crossValidTarget[itemsDone:itemsDone+batchSize]
            itemsDone += batchSize

        yValid = np_utils.to_categorical(yValid, 2)
        xValid = load_data.read_train_data_avi(frameHeight, frameWidth, batchList)
        # zero mean
        mean = np.mean(xValid, axis=0)
        xValid = xValid - mean
        xValid = np.array(xValid, dtype='int8')
        xValid /= 127

        if modelNo == 1:
            xFace = load_data.read_train_data_img(frameHeightFace, frameWidthFace, batchList)
            # zero mean
            mean = np.mean(xFace, axis=0)
            xFace = xFace - mean
        elif modelNo == 2:
            xFace = load_data.read_train_data_img(frameHeightFace, frameWidthFace, batchList)
            xFace1 = np.ones((1, 1, frameHeightFace, frameWidthFace), dtype=xFace.dtype)
            xFace1 = np.append(xFace1, xFace[:-1, :, :, :], axis=0)

            # zero mean
            mean = np.mean(xFace, axis=0)
            xFace = xFace - mean
            mean = np.mean(xFace1, axis=0)
            xFace1 = xFace1 - mean
            xFace = np.array(xFace, dtype='int8')
            xFace1 = np.array(xFace1, dtype='int8')
            xFace /= 127
            xFace1 /= 127
            xValidFull = [xValid, xFace1, xFace]

        if modelNo == 0:
            xValidFull = xValid
        elif modelNo == 1:
            xValidFull = [xValid, xFace]

        predictions = model.predict(xValidFull, miniBatchSize, verbose)
        validationScore += log_loss(yValid, predictions)
        batchCount += 1

    validationScore /= numBatches
    print('Cross Validation Score:{}'.format(validationScore))
    return validationScore

def validate(model, validationTarget, validationId):
    frameHeight, frameWidth = read_config('frameHeight'), read_config('frameWidth')
    frameHeightFace, frameWidthFace = read_config('frameHeightFace'), read_config('frameWidthFace')
    batchSize = read_config('batchSize')
    miniBatchSize = read_config('miniBatchSize')
    verbose = read_config('verbose')
    modelNo = read_config('modelNo')

    numBatches = math.ceil(len(validationTarget)/batchSize)
    itemsDone = 0
    xValid, yValid = [], []
    xFace = []
    xValidFull = []
    batchList = []
    validationScore = 0

    while itemsDone < len(validationTarget):
        if(len(validationTarget) - itemsDone) < batchSize:
            batchList = validationId[itemsDone:]
            yValid = validationTarget[itemsDone:]
            itemsDone = len(validationTarget)
        else:
            batchList = validationId[itemsDone:itemsDone+batchSize]
            yValid = validationTarget[itemsDone:itemsDone+batchSize]
            itemsDone += batchSize

        yValid = np_utils.to_categorical(yValid, 2)
        xValid = load_data.read_validation_data_mp4(frameHeight, frameWidth, batchList)
        # zero mean
        mean = np.mean(xValid, axis=0)
        xValid = xValid - mean
        xValid = np.array(xValid, dtype='int8')
        xValid /= 127

        if modelNo == 1:
            xFace = load_data.read_validation_data_img(frameHeightFace, frameWidthFace, batchList)
            # zero mean
            mean = np.mean(xFace, axis=0)
            xFace = xFace - mean
        elif modelNo == 2:
            xFace = load_data.read_validation_data_img(frameHeightFace, frameWidthFace, batchList)
            #xFace1 = xFace
            xFace1 = np.ones((1, 1, frameHeightFace, frameWidthFace), dtype=xFace.dtype)
            xFace1 = np.append(xFace1, xFace[:-1, :, :, :], axis=0)

            # zero mean
            mean = np.mean(xFace, axis=0)
            xFace = xFace - mean
            mean = np.mean(xFace1, axis=0)
            xFace1 = xFace1 - mean
            xFace = np.array(xFace, dtype='int8')
            xFace1 = np.array(xFace1, dtype='int8')
            xFace /= 127
            xFace1 /= 127
            xValidFull = [xValid, xFace1, xFace]

        if modelNo == 0:
            xValidFull = xValid
        elif modelNo == 1:
            xValidFull = [xValid, xFace]

        predictions = model.predict(xValidFull, miniBatchSize, verbose)
        validationScore += log_loss(yValid, predictions)

    validationScore /= numBatches
    print('Validation Score:{}'.format(validationScore))
    return validationScore

'''
    There are 18 unique drivers in training set
    16 for training and
    2 for cross-validation

    There are 4 unique drivers in evaluation set
'''
def run_cross_validation(numFolds=8, trainVar=1, validateVar=0):
    frameHeight, frameWidth = read_config('frameHeight'), read_config('frameWidth')
    frameHeightFace, frameWidthFace = read_config('frameHeightFace'), read_config('frameWidthFace')
    randomState = 51
    foldNum = 0
    patienceFactor = read_config('patienceFactor')
    numEpochs = read_config('numEpochs')
    debugMode = read_config('debugMode')
    modelNo = read_config('modelNo')

    driverId, trainId, trainTarget, uniqueDrivers = load_data.read_train_targets()
    validationId, validationTarget = load_data.read_validation_targets()
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

    if debugMode:
        print uniqueDrivers
        print('Training set length:{}'.format(len(trainTarget)))
        print('Validation set length:{}'.format(len(validationTarget)))

    if trainVar:
        for train_drivers, test_drivers in kf:
            if foldNum != 0:
                foldNum += 1
                continue
            print('ModelNo:{}'.format(modelNo))
            if modelNo == 0:
                cnnModel = CNN_model(frameHeight, frameWidth)
            elif modelNo == 1:
                cnnModel = two_inputs_cnn_rnn_model(frameHeight, frameWidth, frameHeightFace, frameWidthFace)
            elif modelNo == 2:
                cnnModel = corr_model(frameHeight, frameWidth, frameHeightFace, frameWidthFace)

            uniqueListTrain = [uniqueDrivers[i] for i in train_drivers]
            crossTrainTarget, crossTrainId, crossTrainDriverId = load_data.copy_selected_drivers(trainTarget, trainId, driverId, uniqueListTrain)
            uniqueListValid = [uniqueDrivers[i] for i in test_drivers]
            crossValidTarget, crossValidId, crossValidDriverId = load_data.copy_selected_drivers(trainTarget, trainId, driverId, uniqueListValid)

            print('\nStart KFold number {} from {}'.format(foldNum+1, numFolds))
            print('Split train: ', len(crossTrainTarget))
            print('Split valid: ', len(crossValidTarget))
            print('Train drivers: ', uniqueListTrain)
            print('Test drivers: ', uniqueListValid)

            prevScore = 10000
            minScore = 10000
            patienceCount = 0

            #plotting
            #plt.figure(figsize=(6, 3))
            plt.axis([1, numEpochs, 0, 20])
            plt.ylabel('error')
            plt.xlabel('iteration')
            plt.title('Training error')
            plt.ion()
            yPlotTrainLoss = []
            yPlotTrainAccuracy = []
            xPlot = []
            for epoch in range(int(numEpochs)):
                print ('Epoch {} of {}'.format(epoch+1, numEpochs))

                #if epoch == 0:
                #    cnnModel.load_weights('../weights/weight0.h5')

                cnnModel, history = train(cnnModel, crossTrainTarget, crossTrainId, epoch)
                # plotting
                loss = np.array(history.losses)
                accuracy = np.array(history.accuracy)
                yPlotTrainLoss.append(np.mean(loss))
                #yPlotTrainAccuracy.append(0.69)
                xPlot.append(epoch+1)
                plt.plot(xPlot, yPlotTrainLoss)
                #plt.plot(xPlot, yPlotTrainAccuracy)
                plt.pause(0.05)

                validationScore = cross_validate(cnnModel, crossValidTarget, crossValidId, epoch)

                if validationScore < minScore:
                    historyFile = open(historyFileName, 'a')
                    historyFile.write('{}, {}'.format(foldNum, minScore))
                    historyFile.close()
                    fileName = '../weights/weight' + str(foldNum) + '.h5'
                    if not os.path.isdir(os.path.dirname(fileName)):
                        os.mkdir(os.path.dirname(fileName))
                    cnnModel.save_weights(filepath=fileName, overwrite=True)
                    minScore = validationScore

                if validationScore > prevScore:
                    patienceCount += 1

                if patienceFactor < patienceCount:
                    historyFile = open(historyFileName, 'a')
                    historyFile.write('{}, {}'.format(foldNum, minScore))
                    historyFile.close()
                    break

                if epoch == numEpochs-1:
                    historyFile = open(historyFileName, 'a')
                    historyFile.write('{}, {}'.format(foldNum, minScore))
                    historyFile.close()

                prevScore = validationScore

            plt.savefig('training.png')
            foldNum += 1

    elif validateVar:
        validate(cnnModel, validationTarget, validationId)

    print('Exiting main function\n')
    
def run_test():
    frameHeight, frameWidth = read_config('frameHeight'), read_config('frameWidth')
    frameHeightFace, frameWidthFace = read_config('frameHeightFace'), read_config('frameWidthFace')
    randomState = 51
    debugMode = read_config('debugMode')
    modelNo = read_config('modelNo')
    miniBatchSize = read_config('miniBatchSize')
    verbose = read_config('verbose')

    # Model Initialization
    cnnModel = Sequential()

    if modelNo == 0:
        cnnModel = CNN_model(frameHeight, frameWidth)
    elif modelNo == 1:
        cnnModel = two_inputs_cnn_rnn_model(frameHeight, frameWidth, frameHeightFace, frameWidthFace)

    testPath = '../test'
    testVideos = glob.glob(os.path.join(testPath, '*.mp4'))
    xTest = []

    resultsDir = '../results'
    if not os.path.isdir(resultsDir):
        print('Creating results dir\n')
        os.mkdir(resultsDir)

    weightsDir = '../weights'
    if not os.path.isdir(weightsDir):
        print('No trained weights found, exiting..')
        return

    weights = glob.glob(os.path.join(weightsDir, '*.h5'))

    for testVideo in testVideos:
        fileBase = os.path.basename(testVideo)
        videoPath = os.path.join(testPath, testVideo)
        facesPath = os.path.join(testPath, fileBase.split('.')[0])
        faces = glob.glob(os.path.join(facesPath, '*.png'))
        numFrames = len(faces)
        videoNP = load_avi.load_avi_into_nparray(videoPath, frameHeight, frameWidth, 0, numFrames)
        facesNP = load_avi.load_face_into_nparray(facesPath, frameHeightFace, frameWidthFace, 0, numFrames)
        if modelNo == 0:
            xTest = videoNP
        elif modelNo == 1:
            xTest = [videoNP, facesNP]

        predictionsAll = np.empty((0,numFrames))
        for weight in weights:
            cnnModel.load_weights(weight)
            predictions = cnnModel.predict(xTest, miniBatchSize, verbose)
            predictionsAll = np.append(predictionsAll, predictions, axis=0)
        predictions = np.mean(predictionsAll, axis=0)
        predictions = np.round(predictions)
        resultFileName = os.path.join(resultsDir, fileBase.split('.')[0] + '.txt')
        resultFile = open(resultFileName, 'w')
        resultFile.write(predictions)
        resultFile.close()

if __name__ == '__main__':
    run_cross_validation(8, 1, 0)
    #run_test()

