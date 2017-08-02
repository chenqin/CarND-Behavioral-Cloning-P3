import pandas
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from keras.backend import tf as ktf
import math
import cv2
import os

def my_resize_function(input):
    return ktf.image.resize_images(input, (64, 64))


def load_image(filepath):
	path = '/Users/chenqin/CarND-Behavioral-Cloning-P3/data/IMG/' + os.path.split(filepath)[1]
	return cv2.imread(path, 0)

# open as grayscale, crop non road portion of image, original image is 160x320
def crop_non_road(filepath):
    img = cv2.imread(filepath, 0)
    crop_img = img[50:130,:]
    return crop_img

# preprocessing image files
# flip image, and change steering to opposite direction
# open image as grey scale, resize to 64x64x1
# normalize image to (-1, 1)
def X_train_gen(trainning):
    centers = trainning['center'].values
    lefts = trainning['left'].values
    rights = trainning['right'].values
    steerings = trannings['steering'].values
    x = np.zeros((64, 64, 3))
    X_train = np.zeros(len(steerings), 64, 64, 3)
    y_train = np.zeros(len(steerings))
    for i in range(1, len(centers)):
        #overlap left , center, right into trainning set
        center = load_image(centers[i])[50:130, :]
        left = load_image(lefts[i])[50:130, :]
        right = load_image(rights[i])[50:130, :]
        x = (i, my_resize_function(left), my_resize_function(center), my_resize_function(right))
        y = float(steerings[i])
        X_train[i] = x
        y_train[i] = y
    return X_train, y_train

# load data
trannings = pandas.read_csv('/Users/chenqin/CarND-Behavioral-Cloning-P3/data/driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])
y_train_org = trannings['steering'].values

# drop 6/7 of straight moving examples, skip header index = 0
drop_rows = [i for i in range(1, len(y_train_org)) if math.fabs(float(y_train_org[i])) < 0.25 and randint(0, 6) != 0]
trannings.drop(drop_rows, inplace=True)



# make sure steering data has nice distribution
#plt.hist(pandas.to_numeric(trannings['steering'][1:]))
#plt.ylabel('steering values distribution')
#plt.show()

X_train, y_train = X_train_gen(trainning=trannings)

#test crop
#cv2.imshow("cropped", crop_non_road('/Users/chenqin/CarND-Behavioral-Cloning-P3/data/IMG/center_2017_07_30_13_22_07_801.jpg'))
#cv2.waitKey(0)

#TODO: transfer learning with nvidia model

