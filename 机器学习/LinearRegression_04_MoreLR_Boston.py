
# coding: utf-8

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


from sklearn.linear_model import LinearRegression


# In[5]:


lin_reg = LinearRegression()
lin_reg.fit(X, y)


# In[6]:


lin_reg.coef_


# In[7]:


np.argsort(lin_reg.coef_)


# In[8]:


boston.feature_names


# In[9]:


boston.feature_names[np.argsort(lin_reg.coef_)]


# In[10]:


print(boston.DESCR)

