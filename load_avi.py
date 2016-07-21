import cv2
import numpy as np

'''
    Function to load avi file from disc in required frame resolution
'''

def load_avi_into_nparray(fileName, frameHeight, frameWidth):
    video = cv2.VideoCapture(fileName)
    
    ret, frame = video.read()
    resized = cv2.resize(frame, (frameWidth, frameHeight), cv2.INTER_LINEAR)
    videoNP = resized
    while(video.isOpened()):
        ret, frame = video.read()
        resized = cv2.resize(frame, (frameWidth, frameHeight), cv2.INTER_LINEAR)
        videoNP = np.append(videoNP, resized)

    return videoNP
