# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:11:34 2019

@author: liuyuntao
"""

import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from sklearn import datasets
from sklearn.model_selection import train_test_split

bosten = datasets.load_boston()
X= bosten.data
y = bosten.target

#数据标准化
mean = X.mean(axis=0)
print(mean, "1111111111")
std = X.std(axis=0)

X = (X-mean) / std
print(X)
#print(X)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#model = Sequential()
#
#model.add(Dense(32, input_shape=(13,), activation='relu'))
#model.add(Dense(16, activation='relu'))
#model.add(Dense(1))
#
#model.compile(loss='mse', optimizer='sgd')

#for step in range(501):
#    cost = model.train_on_batch(X, y)
#    if step % 50 == 0:
#        print("After %d trainings, the cost: %f" % (step, cost))
        
#cost = model.evaluate(X, y)
#print("the cost:", cost)

#plt.scatter(X, y)
#plt.show()
