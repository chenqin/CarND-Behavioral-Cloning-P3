import pandas
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from keras.backend import tf as ktf
import math
import cv2
import os
import json
from keras.layers import Dense, Cropping2D, Convolution2D, MaxPooling2D, Dropout, Lambda
from keras.layers import Flatten
from keras.models import Sequential

home_path = '/home/carnd/CarND-Behavioral-Cloning-P3/'

def load_image(filepath):
    path = home_path + 'data/IMG/' + os.path.split(filepath)[1]
    return cv2.imread(path)


# preprocessing image files
# flip image, and change steering to opposite direction
# open image as grey scale, resize to 64x64x3
# normalize image to (-1, 1)
def X_train_gen(trainning, batch_size):
    X_train = np.zeros((batch_size, 160, 320, 3), dtype=float)
    y_train = np.zeros(batch_size, dtype=float)
    # use left and right camera need to make correction due to geometry factor
    correction = 0.3

    while True:
        trainning = shuffle(trainning)
        centers = trainning['center'].values
        lefts = trainning['left'].values
        rights = trainning['right'].values
        steerings = trannings['steering'].values

        for i in range(0, batch_size):
            #overlap left , center, right into trainning set
            choice = randint(0,2)
            filepath = ""
            if choice == 0:
                filepath = centers[i]
                y_train[i] = float(steerings[i])
            elif choice == 1:
                filepath = lefts[i]
                y_train[i] = float(steerings[i]) + correction
            else:
                filepath = rights[i]
                y_train[i] = float(steerings[i]) - correction

            image = load_image(filepath)
            #convert to YUV planes
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

            # do random flip of 50% of images to avoid left turn bias
            if randint(0,1) == 1:
                image = np.fliplr(image)
                y_train[i] = -y_train[i]

            # resize image to 160x320
            image = cv2.resize(image, (320, 160), interpolation=cv2.INTER_AREA)
            X_train[i] = image

        yield X_train, y_train

def X_valid_gen(validation, batch_size):
    X_valid = np.zeros((batch_size, 160, 320, 3), dtype=float)
    y_valid = np.zeros(batch_size, dtype=float)
    while True:
        validation = shuffle(validation)
        centers = validation['center'].values
        steerings = validation['steering'].values
        
        for i in range(0, batch_size):
            #validation only use center image
            filepath = centers[i]
            image = load_image(filepath)
            image = randomize_brightness(image)
            #convert to YUV planes
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            #resize image to
            image = cv2.resize(image, (320, 160), interpolation=cv2.INTER_AREA)
            X_valid[i] = image
            y_valid[i] = float(steerings[i])
        yield X_valid, y_valid

# load data
trannings = pandas.read_csv(home_path+'data/driving_log.csv', skiprows=[0], names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])
y_train_org = trannings['steering'].values

# drop 3/4 of straight moving examples, skip header index = 0
drop_rows = [i for i in range(1, len(y_train_org)) if math.fabs(float(y_train_org[i])) < 0.1 and randint(0, 3) != 0]
trannings.drop(drop_rows, inplace=True)
y_train_org = trannings['steering'].values

# exame distribution of steering bias
#plt.hist(y_train_org)
#plt.ylabel('steering values distribution')
#plt.show()

def randomize_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_brightness = .1 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def nvida_model():
    input_shape = (160, 320, 3)
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
    #crop top 60 and bottom 20
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='relu'))
    model.add(Convolution2D(48, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model


model = nvida_model()
model.summary()

history = model.fit_generator(X_train_gen(trainning=trannings, batch_size=512), samples_per_epoch=len(y_train_org), validation_data=X_valid_gen(validation=trannings, batch_size=512),nb_val_samples=len(y_train_org)/5,nb_epoch=10,verbose=1)

#print(history.history['loss'])
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

# Save model to JSON
with open('model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

model.save_weights("model.h5")
