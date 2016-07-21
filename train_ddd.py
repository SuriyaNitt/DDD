import cv2
import numpy as np
import os
import glob
import hickle
import warnings

warnings.filterwarnings("ignore")

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from keras.regularizers import l1, l2, l1l2, activity_l1, activity_l2, activity_l1l2
from keras.utils.visualize_util import plot

from keras import backend as K
from subprocess import call

from multiprocessing import Pool, Array
import threading
import multiprocessing

np.random.seed(2016)



