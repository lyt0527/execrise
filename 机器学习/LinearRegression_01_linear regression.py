
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt


# In[14]:


x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 2., 3., 4., 5.])


# In[15]:


plt.scatter(x, y)
plt.axis([0, 6, 0, 6])
plt.show()


# In[21]:


x_mean = np.mean(x)
y_mean = np.mean(y)


# In[22]:


num = 0.0
d = 0.0
for x_i, y_i in zip(x, y):
    num += (x_i - x_mean) * (y_i - y_mean)
    d += (x_i - x_mean) ** 2


# In[23]:


a = num / d
b = y_mean - a * x_mean


# In[24]:


a


# In[25]:


b


# In[31]:


y_hat = a * x + b


# In[32]:


plt.scatter(x, y)
plt.plot(x, y_hat, color='red')
plt.axis([0, 6, 0, 6])
plt.show()


# In[33]:


x_predict = 6
y_predict = a * x_predict + b


# In[34]:


y_predict 


# ## 向量化实现的性能测试

# In[38]:


m = 1000000


# In[39]:


big_x = np.random.random(size=m)


# In[44]:


big_y = big_x * 2. + 3. + np.random.normal(size=m)


# In[ ]:


get_ipython().magic(u'timeit reg1.fit(big_x, big_y)')
get_ipython().magic(u'timeit reg2.fit(big_x, big_y)')

