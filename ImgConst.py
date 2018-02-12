import glob
import os
import cv2
import numpy as np
import math
import time


from keras.callbacks import TensorBoard
from keras.models import load_model, Model
from keras.losses import binary_crossentropy, cosine_proximity
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D

import matplotlib.pyplot as plt

import tensorflow as tf




IMG_DIR = "/Users/jaydeep/jaydeep_workstation/ASU/Spring2018/MC_535/data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/"
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_COLOR = 1 #1D images

FILE_EXT = "*.png"

EPOCHS = 20
BATCH_SIZE = 64
TRNG_BATCH_SIZE = 1000

ACTV_FUNC_1 = "relu"
ACTV_FUNC_2 = "sigmoid"
LOSS_1 = "binary_crossentropy"
LOSS_2 = "cosine_proximity"
OPT_1 = "adadelta"
PAD = "same"

ENC_NM = "encoder"
MODEL_FL_PATH = '/Users/jaydeep/jaydeep_workstation/ASU/Spring2018/MC_535/data/autoencoder_{}.h5'

SAVE_FEATURE_PATH = '/Users/jaydeep/jaydeep_workstation/ASU/Spring2018/MC_535/data/img_featurer_{}.txt'
