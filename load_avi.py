import cv2
import numpy as np

'''
    Function to load avi file from disc in required frame resolution
'''

def load_avi_into_nparray(fileName, frameHeight, frameWidth, startFrame, endFrame):
    video = cv2.VideoCapture(fileName)
    numFrames = 0
    
    videoNP = np.empty((0, frameHeight, frameWidth), dtype='float32')
    while(video.isOpened()):
        if startFrame <= numFrames and endFrame >= numFrames:
            ret, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(frame, (frameWidth, frameHeight), cv2.INTER_LINEAR)
            videoNP = np.append(videoNP, [resized], axis=0)
        numFrames += 1

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
