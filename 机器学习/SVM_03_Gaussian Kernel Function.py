
# coding: utf-8

# ### 高斯核函数

# In[36]:


import numpy as np
import matplotlib.pyplot as plt


# In[37]:


x = np.arange(-4, 5, 1)
x


# In[38]:


y = np.array((x >= -2) & (x <= 2), dtype='int')


# In[39]:


y


# In[40]:


plt.scatter(x[y==0], [0] * len(x[y==0]))
plt.scatter(x[y==1], [0] * len(x[y==1]))
plt.show()


# In[41]:


# 因为本例中，x和l均为一个数，所以||x - l||即为(x - l)，不用再花精力求||x-l||的值
def gaussian(x, l):
    gamma = 1.0
    return np.exp(-gamma * (x - 1)**2)


# In[48]:


l1, l2 = -1, 1

X_new = np.empty((len(x), 2))
for i, data in enumerate(x):
    X_new[i, 0] = gaussian(data, l1)
    X_new[i, 1] = gaussian(data, l2)
    print(X_new)


# In[43]:


plt.scatter(X_new[y==0, 0], X_new[y==0, 1])
plt.scatter(X_new[y==1, 0], X_new[y==1, 1])
plt.show()

