import cv2
import numpy as np
from read_config import read_config
from tqdm import tqdm
import accv_utils
import os

'''
    Function to load avi file from disc in required frame resolution
'''

def load_avi_into_nparray(fileName, frameHeight, frameWidth, startFrame, endFrame):
    video = cv2.VideoCapture(fileName)
    numFrames = 0
    debugMode = read_config('debugMode')

    arr = fileName.split('/')
    arrLen = len(arr)
    cacheName = ''
    cachePath = ''
    if arrLen == 5:
        cacheName = arr[3] + '_' + arr[4].split('.')[0] + '_' + str(startFrame) + '_' + str(endFrame) + '.dat'
        cachePath = '../cache/' + cacheName
    elif arrLen == 6:
        cacheName = arr[3] + '_' + arr[4] + '_' + arr[5].split('.')[0] + '_' + str(startFrame) + '_' + str(endFrame) + '.dat'
        cachePath = '../cache/' + cacheName
    pbar = tqdm(total=(endFrame-startFrame+1))

    videoNP = np.empty((0, 1, frameHeight, frameWidth), dtype='float32')

    if os.path.isfile(cachePath):
        print('Restoring from cache')
        videoNP = accv_utils.restore_data(cachePath)
    else:
        while(video.isOpened()):
            ret, frame = video.read()
            if startFrame <= numFrames and endFrame >= numFrames:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(frame, (frameWidth, frameHeight), cv2.INTER_LINEAR)
                resized = resized.reshape(1, frameHeight, frameWidth)
                #if debugMode:
                #    print('resized shape:{}'.format(resized.shape))
                videoNP = np.append(videoNP, [resized], axis=0)
                #if debugMode:
                #    print('videoNP shape:{}'.format(videoNP.shape))
                pbar.update(1)
            if endFrame < numFrames:
                break
            numFrames += 1
        accv_utils.cache_data(videoNP, cachePath)

    pbar.close()
    print('Num frames loaded:{}'.format(endFrame - startFrame + 1))

    return videoNP

def detectFace(frame):
    #debugMode = read_config('debugMode')
    cascPath = '../input/cascades/haarcascades/haarcascade_frontalface_alt2.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    if len(faces) == 1:
        face = faces[0]
        x = face[0]
        y = face[1]
        w = face[2]
        h = face[3]
        detectedFace = frame[y:y+h, x:x+w]
        #if debugMode:
        #    cv2.imshow('Face Detected', detectedFace)
        #    cv2.waitKey(0)
        #    cv2.destroyAllWindows()
        return detectedFace, 1
    else:
        return frame, len(faces)

def load_processed_avi_into_nparray(fileName, frameHeight, frameWidth, startFrame, endFrame):
    arr = fileName.split('/')
    arrLen = len(arr)
    if arrLen == 5:
        cacheName = arr[3] + '_' + arr[4].split('.')[0] + '_' + str(startFrame) + '_' + str(endFrame) + '.dat'
        cachePath = '../cache/' + cacheName
    elif arrLen == 6:
        cacheName = arr[3] + '_' + arr[4] + '_' + arr[5].split('.')[0] + '_' + str(startFrame) + '_' + str(endFrame) + '.dat'
        cachePath = '../cache/' + cacheName

    video = cv2.VideoCapture(fileName)
    numFrames = 0
    debugMode = read_config('debugMode')

    videoNP = np.empty((0, 1, frameHeight, frameWidth), dtype='float32')

    if os.path.isfile(cachePath):
        print('Restoring from cache')
        videoNP = accv_utils.restore_data(cachePath)
    else:
	pbar = tqdm(total=(endFrame-startFrame+1))
        while (video.isOpened()):
            ret, frame = video.read()
            if startFrame <= numFrames and endFrame >= numFrames:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame, numFaces = detectFace(frame)
                resized = cv2.resize(frame, (frameWidth, frameHeight), cv2.INTER_LINEAR)
                resized = resized.reshape(1, frameHeight, frameWidth)
                # if debugMode:
                #    print('resized shape:{}'.format(resized.shape))
                videoNP = np.append(videoNP, [resized], axis=0)
                # if debugMode:
                #    print('videoNP shape:{}'.format(videoNP.shape))
                pbar.update(1)
            if endFrame < numFrames:
                break
            numFrames += 1
        pbar.close()
        print('Creating cache')
        accv_utils.cache_data(videoNP, cachePath)  

    print('Num frames loaded:{}'.format(endFrame - startFrame + 1))
    return videoNP

def load_face_into_nparray(filePath, frameHeight, frameWidth, startFrame, endFrame):
    debugMode = read_config('debugMode')

    if not os.path.isdir(filePath):
        print('FilePath:{} does not exist'.format(filePath))
        return

    arr = filePath.split('/')
    arrLen = len(arr)
    cacheName = ''
    cachePath = ''
    if arrLen == 5:
        cacheName = 'face_' + arr[3] + '_' + arr[4] + '_' + str(startFrame) + '_' + str(endFrame) + '.dat'
        cachePath = '../cache/' + cacheName
    elif arrLen == 6:
        cacheName = 'face_' + arr[3] + '_' + arr[4] + '_' + arr[5] + '_' + str(startFrame) + '_' + str(endFrame) + '.dat'
        cachePath = '../cache/' + cacheName

    pbar = tqdm(total=(endFrame-startFrame+1))
    videoNP = np.empty((0, 1, frameHeight, frameWidth), dtype='float32')
    if os.path.isfile(cachePath):
        print('Restoring from cache')
        videoNP = accv_utils.restore_data(cachePath)
    else:
        for i in range(startFrame, endFrame + 1):
            fileName = os.path.join(filePath, str(i) + '.png')
            frame = cv2.imread(fileName)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detected region is too big or small to be a face
            if frame.shape[0] > 420 and frame.shape[1] > 420 and frame.shape[0] < 150 and frame.shape[1] < 150:
                frame = np.zeros((250, 250), dtype='float32')

            resized = cv2.resize(frame, (frameWidth, frameHeight), cv2.INTER_LINEAR)
            resized = resized.reshape(1, frameHeight, frameWidth)
            #if debugMode:
            #    print('resized shape:{}'.format(resized.shape))
            videoNP = np.append(videoNP, [resized], axis=0)
            #if debugMode:
            #    print('videoNP shape:{}'.format(videoNP.shape))
            pbar.update(1)
            accv_utils.cache_data(videoNP, cachePath)
    pbar.close()
    print('Num frames loaded:{}'.format(endFrame - startFrame + 1))

    return videoNP

def load_frame(fileName, frameHeight, frameWidth, frameNumber):
    video = cv2.VideoCapture(fileName)
    numFrames = 0
    frameNP = np.empty((0, frameHeight, frameWidth))
    while(video.isOpened()):
        if numFrames == frameNumber:
            ret, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameNP = cv2.resize(frame, (frameWidth, frameHeight), cv2.INTER_LINEAR)
            break
        numFrames += 1

    return frameNP

'''
    TODO: Function to create jittered images for Data augmentation 
    eg. images with rotation, stretches, crops, flips, ZCA whitening
'''
