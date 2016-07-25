import os, glob, re
import numpy as np

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
    for character in len(line):
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
    driverId = np.empty((0), dtype='str')
    trainId = np.empty((0), dtype='str')
    trainTarget = np.empty((0), dtype='int8')
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
            glassFolders = os.listdir(os.path.join(path, driverFolder))
            glassFolders = sorted(glassFolders, key = natural_keys)
            for glassFolder in glassFolders: 
                videos = glob.glob(os.path.join(path, driverFolder, glassFolder, '*.avi'))
                for video in videos:
                    fileBase = os.path.basename(video)
                    driverIdLocal, trainIdLocal, trainTargetLocal = populate_train_info(path, driverFolder, \
                                                                                        glassFolder, fileBase)
                driverId = np.append(driverId, driverIdLocal, axis=0)
                trainId = np.append(trainId, trainIdLocal, axid=0)
                trainTarget = np.append(trainTarget, trainTargetLocal, axis=0)

    return driverId, trainId, trainTarget

def populate_valid_info(basePath, driverFolder, videoName):
    validId = []
    validTarget = []

    gndTrthFileName = str(driverFolder) + '_' + videoName.split('.')[0] + 'ing_drowsiness.txt'
    gndTrthFilePath = os.path.join(basePath, driverFolder, gndTrthFileName)
    gndTrthFile = open(gndTrthFilePath, 'r')
    line = gndTrthFile.readline()
    frameNum = 0
    for character in len(line):
         validTarget.append(atoi(character))
         validId.append(str(driverFolder) +  '_' + \
                        videoName.split('.')[0] + '_' + str(frameNum))
         frameNum += 1

    validId = np.array(validId)
    validTarget = np.array(validTarget)

    return validId, validTarget

def read_validation_targets():
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
            videos = glob.glob(os.path.join(path, driverFolder, '*.avi'))
            for video in videos:
                fileBase = os.path.basenane(video)
                validIdLocal, validTargetLocal = populate_valid_info(path, driverFolder, \
                                                                    fileBase)
                validId = np.append(validId, validIdLocal, axis=0)
                validTarget = np.append(validTarget, validTargetLocal, axis=0)

    return validId, validTarget
