# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:39:07 2019

@author: liuyuntao
"""

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np


data=[
    [0.067732,3.176513],[0.427810,3.816464],[0.995731,4.550095],[0.738336,4.256571],[0.981083,4.560815],
    [0.526171,3.929515],[0.378887,3.526170],[0.033859,3.156393],[0.132791,3.110301],[0.138306,3.149813],
    [0.247809,3.476346],[0.648270,4.119688],[0.731209,4.282233],[0.236833,3.486582],[0.969788,4.655492],
    [0.607492,3.965162],[0.358622,3.514900],[0.147846,3.125947],[0.637820,4.094115],[0.230372,3.476039],
    [0.070237,3.210610],[0.067154,3.190612],[0.925577,4.631504],[0.717733,4.295890],[0.015371,3.085028],
    [0.335070,3.448080],[0.040486,3.167440],[0.212575,3.364266],[0.617218,3.993482],[0.541196,3.891471]
]

print(type(data))
data = np.array(data)
print(type(data))
X = data[:, 0:1]
y = data[:, 1]

model = Sequential()

model.add(Dense(units=1, input_dim=1))

# 选择loss
model.compile(loss="mse", optimizer="sgd")

#model.fit(X, y)
print("Traing----------------")
for step in range(501):
    cost = model.train_on_batch(X, y)
    if step % 50 == 0:
        print("After %d trainings, the cost: %f" % (step, cost))

cost = model.evaluate(X, y, batch_size=40)
print("测试值：",cost)

Y_pred = model.predict(X)
plt.scatter(X, y)
plt.plot(X, Y_pred)
plt.show()
