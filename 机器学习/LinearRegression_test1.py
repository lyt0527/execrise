
# coding: utf-8

# ## 衡量回归算法的标准

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# In[4]:


bosten = datasets.load_boston()


# In[5]:


print(bosten.DESCR)


# In[6]:


bosten.feature_names


# In[7]:


x = bosten.data[:, 5]


# In[8]:


x.shape


# In[9]:


y = bosten.target


# In[10]:


y.shape


# In[11]:


plt.scatter(x, y)
plt.show()


# In[12]:


np.max(y)


# In[13]:


x = x[y < 50.0]
y = y[y < 50.0]


# In[14]:


plt.scatter(x, y)
plt.show()


# ## 使用简单线性回归法

# In[ ]:


from 

