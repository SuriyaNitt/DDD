import cv2
import load_avi
import os
import glob
import load_data
from read_config import read_config

def detect_save_face(path, driverFolder, glassFolder, fileBase):
    fileName = os.path.join(path, driverFolder, glassFolder, fileBase)
    destDir = os.path.join(path, driverFolder, glassFolder, fileBase.split('.')[0])
    if not os.path.isdir(destDir):
        os.mkdir(destDir)
    video = cv2.VideoCapture(fileName)
    numFrames = 0
    debugMode = read_config('debugMode')
    frameWidth = read_config('frameWidth')
    frameHeight = read_config('frameHeight')
    detectionListFilePath = os.path.join(destDir, 'detectionList.txt')
    detectionListFile = open(detectionListFilePath, 'w')

    while (video.isOpened()):
        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame, numFaces = load_avi.detectFace(frame)
        resized = cv2.resize(frame, (frameWidth, frameHeight), cv2.INTER_LINEAR)
        resized = resized.reshape(1, frameHeight, frameWidth)
        # if debugMode:
        #    print('resized shape:{}'.format(resized.shape))
        destName = os.path.join(destDir, str(numFrames) + '.png')
        detectionListFile.write(str(numFaces))
        detectionListFile.write('\n')
        cv2.imwrite(destName, frame)
        # if debugMode:
        #    print('videoNP shape:{}'.format(videoNP.shape))
        numFrames += 1

    detectionListFile.close()
    print('Num frames processed:{}'.format(numFrames))

def save_faces():
    driverFolderStart = 8#read_config('driverFolderStart')
    driverFolderEnd = 10#read_config('driverFolderEnd')
    if not os.path.isdir('../input/Training_Dataset'):
        print('Training Dataset is not found in path\n \
               Make sure that it is found in ../input/Training_Dataset folder\n')
    else:
        print('Training Dataset found!\n')
        path = '../input/Training_Dataset'
        driverFolders = os.listdir(path)
        driverFolders = sorted(driverFolders, key=load_data.natural_keys)
        driverFolders = driverFolders[driverFolderStart:driverFolderEnd]
        for driverFolder in driverFolders:
            # if debugMode:
            #    print driverFolder
            glassFolders = os.listdir(os.path.join(path, driverFolder))
            glassFolders = sorted(glassFolders, key=load_data.natural_keys)
            for glassFolder in glassFolders:
                videos = glob.glob(os.path.join(path, driverFolder, glassFolder, '*.avi'))
                for video in videos:
                    fileBase = os.path.basename(video)
                    # if debugMode:
                    #    print('Reading Video details:{}/{}/{}/{}'.format(path, driverFolder, glassFolder, fileBase))
                    detect_save_face(path, driverFolder, glassFolder, fileBase)


if __name__ == '__main__':
    save_faces()
