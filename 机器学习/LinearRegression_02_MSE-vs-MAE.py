
# coding: utf-8

# ## 衡量回归算法的标准

# In[56]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# ## 波士顿房产数据

# In[57]:


boston = datasets.load_boston()
print(boston.DESCR)


# In[58]:


boston.feature_names


# In[59]:


x = boston.data[:,5]


# In[60]:


x.shape


# In[61]:


y = boston.target


# In[62]:


y.shape


# In[63]:


plt.scatter(x, y)
plt.show()


# In[64]:


np.max(y)


# In[65]:


x = x[y < 50.]
y = y[y < 50.]


# In[66]:


plt.scatter(x, y)
plt.show()


# In[67]:


from sklearn.cross_validation import train_test_split


# In[68]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)


# In[69]:


x_train.shape


# In[70]:


x_train_mean = np.mean(x_train)
x_test_mean = np.mean(x_test)
y_train_mean = np.mean(y_train)
y_test_mean = np.mean(y_test)


# In[71]:


numerator = (x_train - x_train_mean).dot(y_train - y_train_mean)
denominator = (x_train - x_train_mean).dot(x_train - x_train_mean)
a = numerator / denominator
b = y_train_mean - a * x_train_mean


# In[72]:


print(a)


# In[73]:


print(b)


# In[74]:


plt.scatter(x_train, y_train)
plt.plot(x_train, a*x_train + b, color='r')
plt.show()


# In[75]:


y_predict = a * x_test + b


# ## 编写公式实现MSE、RMSE和MAE

# In[81]:


# MSE
mse_test = np.sum((y_predict - y_test)**2) / len(y_test)
print(mse_test)


# In[82]:


# MAE
mae_test = np.sum(np.absolute(y_test - y_predict)) / len(y_test)
print(mae_test)


# ### 在scikit-learn中实现MSE、RMSE和MAE

# In[96]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[97]:


# MSE
mse_test = np.sum((y_predict - y_test)**2) / len(y_test)
print(mse_test)


# ## 自行编写公式求R方

# In[100]:


def mean_squared_error(y_true, y_predict):
	"""计算y_true和y_predict之间的MSE"""
	assert len(y_true) == len(y_predict),	"the size of y_true must be equal to the size of y_predict"
	return np.sum(y_true - y_predict)**2 / len(y_true)
print(y_predict,y_test)
r_square = (1 - (mse_test(y_test, y_predict) / np.var(y_test)))
print(r_square)
print(1 - mean_squared_error(y_test, y_predict) / np.var(y_test))


# ## 在scikit-learn中实现R方

# In[99]:


from sklearn.metrics import r2_score
r_square = r2_score(y_test, y_predict)
print(r_square)


# In[ ]:




