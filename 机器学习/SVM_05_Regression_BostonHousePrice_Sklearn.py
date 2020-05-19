
# coding: utf-8

# ### SVM 思想解决回归问题

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


# In[6]:


from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def StandardLinearSVR(epsilon=1.0):
    return Pipeline([
        ('std_scalar', StandardScaler()),
        ('lin_svr', LinearSVR(epsilon=epsilon))
    ])


# In[7]:


svr = StandardLinearSVR(epsilon=1.0)
svr.fit(X_train, y_train)


# In[8]:


svr.score(X_test, y_test)

