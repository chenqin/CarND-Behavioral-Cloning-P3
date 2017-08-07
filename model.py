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
from keras.optimizers import Adam
from keras.regularizers import l2

home_path = '/Users/chenqin/CarND-Behavioral-Cloning-P3/'

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

    while True:
        trainning = shuffle(trainning)
        centers = trainning['center'].values
        lefts = trainning['left'].values
        rights = trainning['right'].values
        steerings = trannings['steering'].values

        for i in range(1, batch_size+1):
            #overlap left , center, right into trainning set
            choice = randint(0,2)
            filepath = ""
            if choice == 0:
                filepath = centers[i]
                y_train[i-1] = float(steerings[i])
            elif choice == 1:
                filepath = lefts[i]
                y_train[i-1] = float(steerings[i]) + 0.3
            else:
                filepath = rights[i]
                y_train[i-1] = float(steerings[i]) - 0.3

            image = load_image(filepath)
            #convert to YUV planes
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

            # do random flip of 50% of images to avoid left turn bias
            if randint(0,1) == 1:
                image = np.fliplr(image)
                y_train[i-1] = -y_train[i-1]

            # resize image to 160x320
            image = cv2.resize(image, (320, 160), interpolation=cv2.INTER_AREA)
            X_train[i-1] = image

        yield X_train, y_train

def X_valid_gen(validation, batch_size):
    X_valid = np.zeros((batch_size, 160, 320, 3), dtype=float)
    y_valid = np.zeros(batch_size, dtype=float)
    while True:
        validation = shuffle(validation)
        centers = validation['center'].values
        lefts = validation['left'].values
        rights = validation['right'].values
        steerings = validation['steering'].values
        
        for i in range(1, batch_size+1):
            choice = randint(0,2)
            filepath = ""
            if choice == 0:
                filepath = centers[i]
            elif choice == 1:
                filepath = lefts[i]
            else:
                filepath = rights[i]
            image = load_image(filepath)
            #convert to YUV planes
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            #resize image to
            image = cv2.resize(image, (160, 320), interpolation=cv2.INTER_AREA)
            X_valid[i-1] = image
            y_valid[i-1] = float(steerings[i])
        yield X_valid, y_valid

# load data
trannings = pandas.read_csv(home_path+'data/driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])
y_train_org = trannings['steering'].values

# drop 3/4 of straight moving examples, skip header index = 0
#drop_rows = [i for i in range(1, len(y_train_org)) if math.fabs(float(y_train_org[i])) < 0.25 and randint(0, 3) != 0]
#trannings.drop(drop_rows, inplace=True)

# exame distribution of steering bias
#plt.hist(y_train)
#plt.ylabel('steering values distribution')
#plt.show()

def nvida_model():
    input_shape = (160, 320, 3)
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
    model.add(Cropping2D(cropping=((64, 64), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Convolution2D(24, 5, 5, activation='relu', W_regularizer = l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='relu', W_regularizer = l2(0.001)))
    model.add(Convolution2D(48, 3, 3, activation='relu', W_regularizer = l2(0.001)))
    model.add(Convolution2D(64, 3, 3, activation='relu', W_regularizer = l2(0.001)))
    model.add(Convolution2D(64, 3, 3, activation='relu', W_regularizer = l2(0.001)))
    model.add(Flatten())
    # OOM in aws g2
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model


model = nvida_model()
model.summary()

history = model.fit_generator(X_train_gen(trainning=trannings, batch_size=256),
                              samples_per_epoch=256,
                              validation_data=X_valid_gen(validation=trannings, batch_size=256),
                              nb_val_samples=256,nb_epoch=10,
                              verbose=1)

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# Save model to JSON
with open('model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

model.save("model.h5")
