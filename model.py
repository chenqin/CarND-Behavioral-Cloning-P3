import pandas
from random import randint
import matplotlib.pyplot as plt
import ipdb
import math

# load data
trannings = pandas.read_csv('sample/driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])
y_train = trannings['steering'].values

# drop 80% of straight moving examples, skip header index = 0
drop_rows = [i for i in range(1, len(y_train)) if math.fabs(float(y_train[i])) < 0.25 and randint(0, 6) != 0]
trannings.drop(drop_rows, inplace=True)

#flip steering value of original tranning data to avoid direction bias
flip_trainnings = trannings.copy()

for i in range(1, len(flip_trainnings['steering'].values)):
    flip_trainnings['steering'].values[i] = -float(flip_trainnings['steering'].values[i])

trannings.append(flip_trainnings)
y_train = trannings['steering'].values
# make sure steering data has nice distribution
plt.hist(pandas.to_numeric(trannings['steering'][1:]))
plt.ylabel('steering values distribution')
plt.show()

