# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:21:12 2019

@author: liuyuntao
"""
import pymysql
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

# 读取数据库
conn = pymysql.connect(host='127.0.0.1', user='root', password='123456', db='111')
sql = 'select * from data'
data = pd.read_sql(sql, conn)
data.to_csv('C:\\Users\\liuyuntao\\Desktop\\sql.csv')

data = pd.read_csv('C:\\Users\\liuyuntao\\Desktop\\2.csv', nrows=10000)
X = data.id
y = data.order_no
print(X, y)

#建立模型
model = Sequential()

model.add(Dense(input_dim=1, units=1))
model.add(Activation("relu"))

model.compile(loss='mse', optimizer='sgd')

model.fit(X, y)

plt.scatter(X, y)
plt.show()