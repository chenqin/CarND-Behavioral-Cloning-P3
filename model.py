import pandas
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.backend import tf as ktf
import math
import cv2
import os
import json
from keras.layers import Dense, Cropping2D, Convolution2D, MaxPooling2D, Dropout, Lambda
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from collections import defaultdict

global stats
global trainnings, home_path, y_train_org

home_path = ''
stats = defaultdict(float)

def load_image(filepath):
    #for posix
    if len(filepath.split('/')) > 1:
        path = home_path + 'IMG/' + filepath.split('/')[-1]
    else:
        #for windows
        path = home_path + 'IMG/' + filepath.split('\\')[-1]
    return cv2.imread(path)

def randomize_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_brightness = .1 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

# preprocessing image files
# flip image, and change steering to opposite direction
# open image as grey scale, resize to 160x320(in case photo are taken with different size)
def X_gen(data, batch_size, is_valid=False):
    X_train = np.zeros((batch_size, 160, 320, 3), dtype=float)
    y_train = np.zeros(batch_size, dtype=float)
    # use left and right camera need to make correction due to geometry factor
    correction = 0.3
    data = shuffle(data)

    while True:
        centers = data['center'].values
        lefts = data['left'].values
        rights = data['right'].values
        steerings = data['steering'].values

        for index in range(0, batch_size):
            #random pick a pic
            i = np.random.randint(len(data))
            #overlap left , center, right into data set
            choice = np.random.randint(3)
            filepath = ""
            if choice == 0 or is_valid:
                filepath = centers[i]
                y_train[index] = float(steerings[i]) 
            elif choice == 1:
                filepath = lefts[i]
                y_train[index] = float(steerings[i]) + correction
            else:
                filepath = rights[i]
                y_train[index] = float(steerings[i]) - correction
            #load image and randomize brightness, avoid overfit
            image = randomize_brightness(load_image(filepath)) if not is_valid else load_image(filepath)

            # do random flip of 50% of images to avoid left turn bias
            if randint(0,1) == 1 and not is_valid:
                image = np.fliplr(image)
                y_train[index] = -y_train[index]

            # resize image to 160x320
            image = cv2.resize(image, (320, 160), interpolation=cv2.INTER_AREA)
            X_train[index] = image

        yield X_train, y_train


def modified_nvida_model():
    input_shape = (160, 320, 3)
    model = Sequential()
    #normalize cell value to [-0.5, 0.5]
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
    #crop top 50 and bottom 20
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='relu'))
    model.add(Convolution2D(48, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    #avoid OOM and remove first dense(1164)
    model.add(Dense(256, activation='relu'))
    #avoid overfit by dropout
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(1))
    return model


def visualize_steering(steerings):
    y, x = np.histogram(steerings)
    nbins = y.size
    plt.bar(x[:-1], y, width=x[1]-x[0], color='red', alpha=0.5)
    plt.hist(steerings, bins=nbins, alpha=0.5)
    plt.grid(True)
    plt.show()

# load data
trainnings = pandas.read_csv('driving_log.csv', skiprows=[0], names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])
y_train_org = trainnings['steering'].values

#drop some straight samples to avoid bais towards straight drive
drop_rows = [i for i in range(1, len(y_train_org)) if abs(float(y_train_org[i])) < 0.1 and np.random.randint(12) != 0]
trainnings.drop(drop_rows, inplace=True)
y_train_org = trainnings['steering'].values
visualize_steering(y_train_org)
model = modified_nvida_model()
model.summary()

model.compile(optimizer=Adam(lr=0.0001), loss="mse", metrics=['accuracy'])
history = model.fit_generator(X_gen(trainnings, batch_size=256), samples_per_epoch=len(y_train_org),
    validation_data=X_gen(trainnings, 256, True),nb_val_samples=len(y_train_org),nb_epoch=3,verbose=1)

print(history.history['loss'])

# Save model to JSON
with open('model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

#save model
model.save("model.h5")
