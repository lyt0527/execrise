
# coding: utf-8

# ### 决策树解决回归问题

# In[8]:


import numpy as np
import matplotlib.pyplot as plt


# In[9]:


from sklearn import datasets

bosten = datasets.load_boston()
X = bosten.data
y = bosten.target


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


# ### Decision Tree Regreeor

# In[16]:


from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)


# In[17]:


dt_reg.score(X_test, y_test)


# In[18]:


dt_reg.score(X_train, y_train)

