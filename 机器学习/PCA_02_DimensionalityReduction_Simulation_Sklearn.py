
# coding: utf-8

# ### 从高维数据向低维数据的映

# In[9]:


import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100, 2))
X[:, 0] = np.random.uniform(0., 100., size=100)
X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0., 10., size=100)


# In[10]:


plt.scatter(X[:, 0], X[:, 1])
plt.show()


# In[12]:


from sklearn.decomposition import PCA


# In[13]:


# 求主成分，n_components为几就是求前几个主成分
pca = PCA(n_components=1)
pca.fit(X)


# In[14]:


pca.components_


# In[16]:


# 降维，本例中，将X从二维降至一维
x_reduction = pca.transform(X)


# In[18]:


x_reduction.shape


# In[20]:


# 升维，将刚才被降成一维的X重新升至二维
x_restore = pca.inverse_transform(x_reduction)


# In[22]:


x_restore.shape


# In[23]:


plt.scatter(X[:, 0], X[:, 1])
plt.scatter(x_restore[:, 0], x_restore[:, 1], color='r', alpha=0.5)
plt.show()

