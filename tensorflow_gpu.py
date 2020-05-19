from PIL import Image
import tensorflow as tf
import numpy as np
import csv
import re
import cv2
from keras import layers, Sequential
import keras
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import time

#将训练集图片转成数组
train_path = "C:\\Users\\liuyuntao\\Desktop\\final2\\"
def train_img(filename):
    img1 = Image.open(train_path+filename+'.png').convert('L')
    img = np.array(img1)

    return img
x_train = []
#将所有的训练图片转成数组
y_train = []
csv_reader = csv.reader(open("E:\\图像处理\\aptos2019-blindness-detection\\train.csv"))
i = 0
for row in csv_reader:
    if i == 0:
        i += 1
        continue
    img = train_img(row[0])
    x_train.append(img)
    y_train.append(row[1])
    i += 1
X_train = np.array(x_train)
y_train = np.array(y_train)


# 数据预处理
X_train = X_train / 255.
X_train = X_train.reshape(-1, 256, 256, 1)
y_train = tf.keras.utils.to_categorical(y_train, 5)

#Conv2D参数调整，activation激活函数"relu","sigmoid","selu","tanh",Dropout防止过拟合，参数可调整

with tf.device('/gpu:0'):
	start_time = time.time()
	model = Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
	model.add(layers.Conv2D(32, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(layers.Dropout(0.25))

	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(layers.Dropout(0.25))

	model.add(layers.Flatten())
	model.add(layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(5, activation="sigmoid"))
	model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])

	X_t, X_te, y_t, y_te = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
	print('start')
	model.summary()
	model.fit(X_t, y_t, batch_size=8, epochs=10, validation_data=(X_te,y_te))
	end_time = time.time()
	total_time = end_time - start_time
print(total_time)