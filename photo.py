import numpy as np
import pandas as pd
import pymysql
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
# from keras.layers import Activation, Dense, Dropout, BatchNormalization
# from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split

conn = pymysql.connect(host="172.22.14.51", port=8097, user="root", password="system", db="caijingbo")
data = pd.read_sql("select * from data_7_30_3 where is_buyer<=2", conn)

data.pop('age')
data.pop('id')
data.pop('six_id')
data.pop('tdc_id')
data.pop('name')
data.pop('tel')
data.pop('province')
data.pop('area')
data.pop("age_check")
data.pop('Questionnaire_family_money')
data.pop('Questionnaire_married')
data.pop('Questionnaire_fuel_vehicle')
data.pop('Store_app_activate_time_check')
data.pop('Integral_total_integral')
data.head(10)
print(data.shape)
df = np.array(data)
X = df[:, :35]
y = df[:, 35:36]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = tf.keras.Sequential()
model.add(layers.Dense(128, input_shape=(35,), activation='relu'))
# model.add(layers.Dense(64, activation='sigmoid'))
# model.add(layers.GaussianNoise(0.1))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
# model.summary()

model.compile(loss='mae', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=35, epochs=10)
loss, accuracy = model.evaluate(X_train, y_train)
print(loss)
print(accuracy)