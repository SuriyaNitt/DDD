
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.visualize_util import plot
from keras.utils import np_utils

#from multiprocessing import Pool, Array
#import threading
#import multiprocessing

import math
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
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.1))

    model1.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.2))

    model1.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model1.add(MaxPooling2D(pool_size=(8, 8)))
    model1.add(Dropout(0.2))

    # Convolutional stack for face ROI
    model2 = Sequential()
    model2.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu',
                            input_shape=(1, int(frameHeight2), int(frameWidth2))))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.1))

    model2.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))

    model2.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(8, 8)))
    model2.add(Dropout(0.2))

    # FC stack with merged convolutional stacks
    model = Sequential()
    model.add(Merge([model1, model2], mode='concat', concat_axis=-1))

    model.add(Flatten())

    model.add(Dense(32, W_regularizer=l2(1e-4)))
    model.add(Activation('relu'))     

    model.add(Dense(2, W_regularizer=l2(1e-0)))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=["accuracy"])
    
    plot(model, to_file='model.png')
    
    return model
    
def train(model, crossTrainTarget, crossTrainId):
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
            
    while itemsDone < len(crossTrainTarget):
        print('\n Batch {} of {}'.format(batchCount+1, numBatches))
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
        if modelNo == 1:
            xFace = load_data.read_train_data_img(frameHeightFace, frameWidthFace, batchList)
            # zero mean
            mean = np.mean(xFace, axis=0)
            xFace = xFace - mean
        if modelNo == 0:
            xTrainFull = xTrain
        elif modelNo == 1:
            xTrainFull = [xTrain, xFace]

        print('Loaded data')
        if debugMode:
            print('xTrain shape:{}'.format(xTrainFull[0].shape + xTrainFull[1].shape))
            print('yTrain shape:{}'.format(yTrain.shape))
        model.fit(xTrainFull, yTrain, miniBatchSize, miniNumEpochs, verbose)
        batchCount += 1

    print('Training Epoch done')
    return model
        
def cross_validate(model, crossValidTarget, crossValidId):
    frameHeight, frameWidth = read_config('frameHeight'), read_config('frameWidth')
    frameHeightFace, frameWidthFace = read_config('frameHeightFace'), read_config('frameWidthFace')
    batchSize = read_config('batchSize')
    miniBatchSize = read_config('miniBatchSize')
    verbose = read_config('verbose')
    modelNo = read_config('modelNo')
    
    numBatches = math.ceil(len(crossValidTarget)/batchSize)
    itemsDone = 0
    xValid, yValid = [], []
    xFace = []
    xValidFull = []
    batchList = []
    validationScore = 0
    
    while itemsDone < len(crossValidTarget):
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
        if modelNo == 1:
            xFace = load_data.read_train_data_img(frameHeightFace, frameWidthFace, batchList)
            # zero mean
            mean = np.mean(xFace, axis=0)
            xFace = xFace - mean
        if modelNo == 0:
            xValidFull = xValid
        elif modelNo == 1:
            xValidFull = [xValid, xFace]

        predictions = model.predict(xValidFull, miniBatchSize, verbose)
        validationScore += log_loss(yValid, predictions)
        
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
        if modelNo == 1:
            xFace = load_data.read_validation_data_img(frameHeightFace, frameWidthFace, batchList)
            # zero mean
            mean = np.mean(xFace, axis=0)
            xFace = xFace - mean
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
            if modelNo == 0:
                cnnModel = CNN_model(frameHeight, frameWidth)
            elif modelNo == 1:
                cnnModel = two_inputs_cnn_model(frameHeight, frameWidth, frameHeightFace, frameWidthFace)
 
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
        
            for epoch in range(int(numEpochs)):
                print ('Epoch {} of {}'.format(epoch+1, numEpochs))
            
                cnnModel = train(cnnModel, crossTrainTarget, crossTrainId)
                validationScore = cross_validate(cnnModel, crossValidTarget, crossValidId)

                if validationScore < minScore:
		    historyFile = open(historyFileName, 'a')
                    historyFile.write('{}, {}'.format(foldNum, minScore))
                    historyFile.close()
                    fileName = '../weights/weight' + str(num_fold) + '.h5'
                    if not os.path.isdir(os.path.dirname(fileName)):
                        os.mkdir(os.path.dirname(fileName))
                    model.save_weights(filepath=fileName, overwrite=True)
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
            
            foldNum += 1

    elif validateVar:
        validate(cnnModel, validationTarget, validationId)
        
    print('Exiting main function\n')
    
            
if __name__ == '__main__':
    run_cross_validation(8, 1, 0)
                
            
