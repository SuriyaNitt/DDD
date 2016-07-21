import cv2
import numpy as np

'''
    Function to load avi file from disc in required frame resolution
'''

def load_avi_into_nparray(fileName, frameHeight, frameWidth):
    video = cv2.VideoCapture(fileName)
    numFrames = 1
    
    ret, frame = video.read()
    resized = cv2.resize(frame, (frameWidth, frameHeight), cv2.INTER_LINEAR)
    videoNP = resized
    while(video.isOpened()):
        ret, frame = video.read()
        resized = cv2.resize(frame, (frameWidth, frameHeight), cv2.INTER_LINEAR)
        videoNP = np.append(videoNP, resized)
        numFrames += 1

    print('Num frames loaded:{}'.format(numFrames))

    return videoNP

def load_frame(fileName, frameHeight, frameWidth, frameNumber):
    video = cv2.VideoCapture(fileName)
    numFrames = 0
    frameNP = np.empty((0, frameHeight, frameWidth))
    while(video.isOpened()):
        if numFrames == frameNumber:
            ret, frame = video.read()
            frameNP = cv2.resize(frame, (frameWidth, frameHeight), cv2.INTER_LINEAR)
            break
        numFrames += 1

    return frameNP

'''
    TODO: Function to create jittered images for Data augmentation 
    eg. images with rotation, stretches, crops, flips, ZCA whitening
'''
