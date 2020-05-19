# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:44:47 2019

@author: liuyuntao
"""

from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("C:\\Users\\liuyuntao\\Desktop\\train.csv", nrows=1000)
print(data.head(5))

X = data.acoustic_data
y = data.time_to_failure

model = Sequential()

model.add(Dense(units=1, input_dim=1))

model.compile(loss="mse", optimizer="sgd")

for step in range(501):
    cost = model.train_on_batch(X, y)
    if step % 50 == 0:
        print("After %d trainings, the cost: %f" % (step, cost))
    
cost = model.evaluate(X, y, batch_size=40)
print(cost)

Y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, Y_pred)
plt.show()