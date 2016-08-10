import cv2
import load_avi
import os
import glob
import load_data
from read_config import read_config
import sys
from tqdm import tqdm

def detect_save_face(path, driverFolder, glassFolder, fileBase):
    fileName = os.path.join(path, driverFolder, glassFolder, fileBase)
    destDir = os.path.join(path, driverFolder, glassFolder, fileBase.split('.')[0])
    if not os.path.isdir(destDir):
        os.mkdir(destDir)
    else:
        return
    video = cv2.VideoCapture(fileName)
    numFrames = 0
    debugMode = read_config('debugMode')
    frameWidth = read_config('frameWidth')
    frameHeight = read_config('frameHeight')
    detectionListFilePath = os.path.join(destDir, 'detectionList.txt')
    detectionListFile = open(detectionListFilePath, 'w')
    
    bar = tqdm(total=video.get(cv2.CAP_PROP_FRAME_COUNT))

    while (video.isOpened()):
        ret, frame = video.read()
        if ret:
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
        else:
            break
        bar.update(1)
        numFrames += 1
    
    bar.close()
    video.release()
    detectionListFile.close()
    print('Num frames processed:{}'.format(numFrames))

def save_faces_train(driverFolderStart, driverFolderEnd):
    debugMode = read_config('debugMode')
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
                    if debugMode:
                        print('Reading Video details:{}/{}/{}/{}'.format(path, driverFolder, glassFolder, fileBase))
                    detect_save_face(path, driverFolder, glassFolder, fileBase)

def save_faces_valid(driverFolderStart, driverFolderEnd):
    debugMode = read_config('debugMode')
    if not os.path.isdir('../input/Evaluation_Dataset'):
        print('Evaluation Dataset is not found in path\n \
               Make sure that it is found in ../input/Evaluation_Dataset folder\n')
    else:
        print('Evaluation Dataset found!\n')
        path = '../input/Evaluation_Dataset'
        driverFolders = os.listdir(path)
        driverFolders = sorted(driverFolders, key=load_data.natural_keys)
        driverFolders = driverFolders[driverFolderStart:driverFolderEnd]
        for driverFolder in driverFolders:
            videos = glob.glob(os.path.join(path, driverFolder, '*.mp4'))
            for video in videos:
                fileBase = os.path.basename(video)
                if debugMode:
                    print('Reading Video details:{}/{}/{}'.format(path, driverFolder, fileBase))
                detect_save_face(path, driverFolder, '', fileBase)

def save_faces_test(start, end):
    if not os.path.isdir('../test'):
        print('Testset is not found in path\n \
               Make sure that it is found in ../test folder\n')
    else:
        print('Testset found!\n')
        path = '../test'
        videos = glob.glob(ps.path.join(path, '*.mp4'))
        videos = videos[start:end]
        for video in videos:
            fileBase = os.path.basename(video)
            detect_save_face(path, '', '', fileBase)

if __name__ == '__main__':
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    #save_faces_train(start, end)
    #save_faces_valid(start, end)
    save_faces_test(start, end)
