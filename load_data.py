import os, glob, re
import numpy as np
import load_avi
from read_config import read_config

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def populate_train_info(basePath, driverFolder, glassFolder, videoName):
    driverId = []
    trainId = []
    trainTarget = []

    gndTrthFileName = str(driverFolder) + '_' + videoName.split('.')[0] + '_drowsiness.txt'
    gndTrthFilePath = os.path.join(basePath, driverFolder, glassFolder, gndTrthFileName)
    gndTrthFile = open(gndTrthFilePath, 'r')
    line = gndTrthFile.readline()
    frameNum = 0
    for character in line:
         trainTarget.append(atoi(character))
         trainId.append(str(driverFolder) + '_' + str(glassFolder) + '_' + \
                        videoName.split('.')[0] + '_' + str(frameNum))
         driverId.append(str(driverFolder))    
         frameNum += 1

    driverId = np.array(driverId)
    trainId = np.array(trainId)
    trainTarget = np.array(trainTarget)

    return driverId, trainId, trainTarget

def read_train_targets():
    debugMode = read_config('debugMode')
    driverId = np.empty((0), dtype='str')
    trainId = np.empty((0), dtype='str')
    trainTarget = np.empty((0), dtype='int8')
    uniqueDrivers = np.empty((0), dtype='str')
    # Check the existence of data directory
    if not os.path.isdir('../input/Training_Dataset'):
        print('Training Dataset is not found in path\n \
               Make sure that it is found in ../input/Training_Dataset folder\n')
    else:
        print('Training Dataset found!\n')
        path =  '../input/Training_Dataset'
        driverFolders = os.listdir(path)
        driverFolders = sorted(driverFolders, key=natural_keys)
        for driverFolder in driverFolders:
            #if debugMode:
            #    print driverFolder
            glassFolders = os.listdir(os.path.join(path, driverFolder))
            glassFolders = sorted(glassFolders, key = natural_keys)
            for glassFolder in glassFolders: 
                videos = glob.glob(os.path.join(path, driverFolder, glassFolder, '*.avi'))
                for video in videos:
                    fileBase = os.path.basename(video)
                    driverIdLocal, trainIdLocal, trainTargetLocal = populate_train_info(path, driverFolder, \
                                                                                        glassFolder, fileBase)
                driverId = np.append(driverId, driverIdLocal, axis=0)
                trainId = np.append(trainId, trainIdLocal, axis=0)
                trainTarget = np.append(trainTarget, trainTargetLocal, axis=0)
            uniqueDrivers = np.append(uniqueDrivers, [str(driverFolder)], axis=0)

    return driverId, trainId, trainTarget, uniqueDrivers

def populate_valid_info(basePath, driverFolder, videoName):
    validId = []
    validTarget = []

    gndTrthFileName = videoName.split('.')[0] + 'ing_drowsiness.txt'
    gndTrthFilePath = os.path.join(basePath, driverFolder, gndTrthFileName)
    gndTrthFile = open(gndTrthFilePath, 'r')
    line = gndTrthFile.readline()
    frameNum = 0
    for character in line:
         validTarget.append(atoi(character))
         validId.append(str(driverFolder) +  '_' + \
                        videoName.split('.')[0] + '_' + str(frameNum))
         frameNum += 1

    validId = np.array(validId)
    validTarget = np.array(validTarget)

    return validId, validTarget

def read_validation_targets():
    debugMode = read_config('debugMode')
    validId = np.empty((0), dtype='str')
    validTarget = np.empty((0), dtype='int8')
    # Check the existence of data directory
    if not os.path.isdir('../input/Evaluation_Dataset'):
        print('Evaluation Dataset is not found in path\n \
               Make sure that it is found in ../input/Evaluation_Dataset folder\n')
    else:
        print('Evaluation Dataset found!\n')
        path = '../input/Evaluation_Dataset'
        driverFolders = os.listdir(path)
        driverFolders = sorted(driverFolders, key=natural_keys)
        for driverFolder in driverFolders:
            #if debugMode:
            #    print driverFolder
            videos = glob.glob(os.path.join(path, driverFolder, '*.mp4'))
            for video in videos:
                fileBase = os.path.basename(video)
                validIdLocal, validTargetLocal = populate_valid_info(path, driverFolder, \
                                                                    fileBase)
                validId = np.append(validId, validIdLocal, axis=0)
                validTarget = np.append(validTarget, validTargetLocal, axis=0)

    return validId, validTarget

def read_train_data(frameHeight, frameWidth, batchList):    
    print('Loading training data\n')
    
    trainingData = np.empty((0, 1, frameHeight, frameWidth), dtype='float32')
    prevId = ''
    uniqueVideosList = []
    instanceCount = 0
    debugMode = read_config('debugMode')
    
    if debugMode:
        print'\n'
        print('BatchList length:{}'.format(len(batchList)))
        #print('BatchList:{}'.format(batchList))
    
    for instance in batchList:
        arr = instance.split('_')
        driverId = arr[0]
        glassInfo = arr[1]
        videoName = arr[2]
        
        id = driverId + glassInfo + videoName
        if id != prevId:
            uniqueVideosList.append(instanceCount)
        instanceCount += 1
        prevId = id
        
    if debugMode:
        print('Finished scanning batchList')
        print('uniqueVideosList length:{}'.format(len(uniqueVideosList)))
    
    for unqVidIdx1, unqVidIdx2 in zip(uniqueVideosList[:-1], uniqueVideosList[1:]):
        startFrame = batchList[unqVidIdx1].split('_')[3]
        endFrame = batchList[unqVidIdx2-1].split('_')[3]
        
        driverId = batchList[unqVidIdx1].split('_')[0]
        glassInfo = batchList[unqVidIdx1].split('_')[1]
        videoName = batchList[unqVidIdx1].split('_')[2]
        
        fileName = os.path.join('..', 'input', 'Training_Dataset', str(driverId), \
                                str(glassInfo), str(videoName) + '.avi')
        videoNP = load_avi.load_avi_into_nparray(fileName, frameHeight, frameWidth, int(startFrame), int(endFrame))
        trainingData = np.append(trainingData, videoNP, axis=0)
        
    startFrame = batchList[uniqueVideosList[-1]].split('_')[3]
    endFrame = batchList[-1].split('_')[3]
    
    driverId = batchList[uniqueVideosList[-1]].split('_')[0]
    glassInfo = batchList[uniqueVideosList[-1]].split('_')[1]
    videoName = batchList[uniqueVideosList[-1]].split('_')[2]
    
    fileName = os.path.join('..', 'input', 'Training_Dataset', str(driverId), \
                                str(glassInfo), str(videoName) + '.avi')
                                
    if debugMode:
        print 'Loading input data from avi file'
    try:
        startFrame = int(startFrame)
        endFrame = int(endFrame)
    except ValueError:
        print('StartFrame value:{}'.format(startFrame))
        print('EndFrame value:{}'.format(endFrame))
        print('batchList:{}'.format(batchList[1148]))

    videoNP = load_avi.load_avi_into_nparray(fileName, frameHeight, frameWidth, int(startFrame), int(endFrame))
    trainingData = np.append(trainingData, videoNP, axis=0)
        
    return trainingData    
    
def read_validation_data(frameHeight, frameWidth, batchList):
    print('Loading validation data\n')
    
    validationData = np.empty((0, frameHeight, frameWidth), dtype='float32')
    prevId = ''
    uniqueVideosList = []
    instanceCount = 0
    
    for instance in batchList:
        arr = instance.split('_')
        driverId = arr[0]
        videoName = arr[1]
        id = driverId + videoName
        if id != prevId:
            uniqueVideosList.append(instanceCount)
        instanceCount += 1
    
    for unqVidIdx1, unqVidIdx2 in zip(uniqueVideosList[:-1], uniqueVideosList[1:]):
        startFrame = batchList[unqVidIdx1].split('_')[2]
        endFrame = batchList[unqVidIdx2-1].split('_')[2]
        
        driverId = batchList[unqVidIdx1].split('_')[0]
        videoName = batchList[unqVidIdx1].split('_')[1]
        
        fileName = os.path.join('..', 'input', 'Evaluation_Dataset', str(driverId), \
                                str(videoName) + '.avi')
        videoNP = load_avi.load_avi_into_nparray(fileName, frameHeight, frameWidth, startFrame, endFrame)
        validationData = np.append(validationData, videoNP, axis=0)
        
    startFrame = batchList[uniqueVideosList[-1]].split('_')[2]
    endFrame = batchList[-1].split('_')[2]
    
    driverId = batchList[uniqueVideosList[-1]].split('_')[0]
    videoName = batchList[uniqueVideosList[-1]].split('_')[1]
    
    fileName = os.path.join('..', 'input', 'Evaluation_Dataset', str(driverId), \
                            str(videoName) + '.avi')
    videoNP = load_avi.load_avi_into_nparray(fileName, frameHeight, frameWidth, startFrame, endFrame)
    validationData = np.append(validationData, videoNP, axis=0)
        
    return validationData
    
def copy_selected_drivers(trainTarget, trainId, driverId, driverList):
    target = []
    trainIndex = []
    driverIndex = []
    debugMode = read_config('debugMode')
    if debugMode:
        print('\n')
        print('Length of driverId:{}'.format(len(driverId)))
        print('Length of driverList:{}'.format(len(driverList)))
        print driverId[0]
        
    for i in range(len(driverId)):
        if driverId[i] in driverList:
            target.append(trainTarget[i])
            trainIndex.append(trainId[i])
            driverIndex.append(driverId[i])
    target = np.array(target)
    trainIndex = np.array(trainIndex)
    driverIndex = np.array(driverIndex)
    return target, trainIndex, driverIndex