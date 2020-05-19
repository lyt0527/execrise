# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:18:02 2019

@author: liuyuntao
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("C:\\Users\\liuyuntao\\Desktop\\数据.xlsx")
new_data = data.dropna(axis=0)
#print(new_data)
X_train = new_data.发送时间
list = []
for t in X_train:
	t1 = float(t.split('-')[2])
	list.append(t1)
    
X_train = pd.DataFrame(list)
y_train = new_data.发送数量

model = Sequential()

# 将一些网络层通过.add()堆叠起来，构成了模型
model.add(Dense(input_dim=1, units=1))
#model.add(Activation("relu"))
#model.add(Dense(units=10))
#model.add(Activation("softmax"))

# 模型构造完成后，使用.compile()方法进行编译(损失函数和优化器)：
model.compile(loss="mse", optimizer="sgd")

# 在训练数据上按batch进行一定次数的迭代来训练网络
#model.fit(X_train, y_train, batch_size=32)

# 通过手动将一个个batch的数据送入网络中训练，则使用：
for step in range(501):
    cost = model.train_on_batch(X_train, y_train)
    if step % 50 == 0:
        print("After %d trainings, the cost: %f" % (step, cost))

# 使用一行代码对我们模型进行评估，看模型是否满足：
cost = model.evaluate(X_train, y_train)
print("test cost:", cost)

# 最后使用模型进行预测：
Y_pred = model.predict(X_train)

plt.scatter(X_train, y_train)
plt.plot(X_train, Y_pred)
plt.show()


