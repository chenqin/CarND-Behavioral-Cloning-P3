import pandas
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import random
from keras.backend import tf as ktf
import math
import cv2
import os
import json
from keras.layers import Dense, ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

def my_resize_function(input):
    return ktf.image.resize_images(input, (64, 64))


def load_image(filepath):
    path = '/Users/chenqin/CarND-Behavioral-Cloning-P3/data/IMG/' + os.path.split(filepath)[1]
    return cv2.imread(path)


# preprocessing image files
# flip image, and change steering to opposite direction
# open image as grey scale, resize to 64x64x1
# normalize image to (-1, 1)
def X_train_gen(trainning):
    centers = trainning['center'].values
    lefts = trainning['left'].values
    rights = trainning['right'].values
    steerings = trannings['steering'].values

    X_train = np.zeros((len(steerings), 66, 200, 3))
    y_train = np.zeros(len(steerings))

    for i in range(1, len(centers)):
        #overlap left , center, right into trainning set
        choice = randint(0,2)
        filepath = ""
        if choice == 0:
            filepath = centers[i]
            y_train[i] = float(steerings[i])
        elif choice == 1:
            filepath = lefts[i]
            y_train[i] = float(steerings[i]) + 0.3
        else:
            filepath = rights[i]
            y_train[i] = float(steerings[i]) - 0.3

        image = load_image(filepath)[50:130, :]
        #convert to YUV planes
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        #resize image to
        image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)

        # do random flip of 50% of images to avoid left turn bias
        if randint(0,1) == 1:
            image = cv2.flip(image, 1)
            y_train[i] = -y_train[i]

        X_train[i] = image
    return X_train, y_train

# load data
trannings = pandas.read_csv('/Users/chenqin/CarND-Behavioral-Cloning-P3/data/driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])
y_train_org = trannings['steering'].values

# drop 3/4 of straight moving examples, skip header index = 0
drop_rows = [i for i in range(1, len(y_train_org)) if math.fabs(float(y_train_org[i])) < 0.25 and randint(0, 3) != 0]
trannings.drop(drop_rows, inplace=True)

# exame distribution of steering bias
#plt.hist(y_train)
#plt.ylabel('steering values distribution')
#plt.show()

#TODO: transfer learning with nvidia model

def nvida_model(X_train, y_train):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(200,66,3)))
    model.add(Convolution2D(24, 5, 5, activation='relu'))
    model.add(Convolution2D(36, 5, 5, activation='relu'))
    model.add(Convolution2D(48, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    model.summary()
    # Save model to JSON
    with open('autopilot_basic_model.json', 'w') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))


X_train, y_train = X_train_gen(trainning=trannings)
nvida_model(X_train, y_train)