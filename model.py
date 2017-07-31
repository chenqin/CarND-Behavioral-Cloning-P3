import pandas
from random import randint
import matplotlib.pyplot as plt
import ipdb
import math
import cv2
import os

# load data
trannings = pandas.read_csv('data/driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])
y_train = trannings['steering'].values

# drop 6/7 of straight moving examples, skip header index = 0
drop_rows = [i for i in range(1, len(y_train)) if math.fabs(float(y_train[i])) < 0.25 and randint(0, 6) != 0]
trannings.drop(drop_rows, inplace=True)

#flip steering value of original tranning data to avoid direction bias
flip_trainnings = trannings.copy()

for i in range(1, len(flip_trainnings['steering'].values)):
	#img = load_image(flip_trainnings['center'].values[i])
	#flipimg = cv2.flip(img, 1)
	# TODO: save fliped image or keep in batch
	flip_trainnings['steering'].values[i] = -float(flip_trainnings['steering'].values[i])


# merge fliped tranning data with original tranning data
trannings.append(flip_trainnings)
y_train = trannings['steering'].values

# make sure steering data has nice distribution
#plt.hist(pandas.to_numeric(trannings['steering'][1:]))
#plt.ylabel('steering values distribution')
#plt.show()

def load_image(filepath):
	path = 'data/IMG/' + os.path.split(filepath)[1]
	return cv2.imread(path, 0)

# open as grayscale, crop non road portion of image, original image is 160x320
def crop_non_road(filepath):
    img = cv2.imread(filepath, 0)
    crop_img = img[50:130,:]
    return crop_img

#test crop
cv2.imshow("cropped", crop_non_road('data/IMG/center_2017_07_30_13_22_07_801.jpg'))
cv2.waitKey(0)

#TODO: transfer learning with nvidia model
