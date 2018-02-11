import glob
import os
import cv2
import numpy as np
import sklearn.model_selection.train_test_split


IMG_DIR = "/home/local/ASUAD/jchakra1/jaydeep/MC_535/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/"
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_COLOR = 1 #1D images


def get_data():

    os.chdir(IMG_DIR)
    
    image_arr = []
    for file in glob.glob("*.png"):
        img = cv2.imread(IMG_DIR + file)
        img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
        image_arr.append(img)
        
    
    def imgToVac(image):
        img_vec = []
        for x in range(IMG_WIDTH):
            for y in range(IMG_HEIGHT):
                for z in range(IMG_COLOR):
                    img_vec.append(image[x][y][z])
                    
        return np.asanyarray(img_vec)
    
    img_vec_arr = np.array([imgToVac(image) for image in image_arr])
    
    return sklearn.model_selection.train_test_split(img_vec_arr)
