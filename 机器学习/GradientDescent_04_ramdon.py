
# coding: utf-8

# ### 随机梯度下降法

# In[1]:


import numpy as np
from sklearn import datasets


# In[2]:


boston = datasets.load_boston()
X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]


# In[3]:


from sklearn.cross_validation import train_test_split


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


stand_scaler1 = StandardScaler()
stand_scaler1.fit(x_train)
x_train_stand = stand_scaler1.transform(x_train)
stand_scaler2 = StandardScaler()
stand_scaler2.fit(x_test)
x_test_stand = stand_scaler2.transform(x_test)


# In[7]:


from sklearn.linear_model import SGDRegressor


# In[8]:


sgd_reg = SGDRegressor(max_iter=5, n_iter=100)
sgd_reg.fit(x_train_stand, y_train)
sgd_reg.score(x_test_stand, y_test)

