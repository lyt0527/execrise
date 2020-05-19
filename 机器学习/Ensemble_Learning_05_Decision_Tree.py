
# coding: utf-8

# ### 随机森林和Extra-Trees

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)


# In[4]:


plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()


# ### 随机森林

# In[6]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, random_state=666, oob_score=True, n_jobs=-1)
rf_clf.fit(X, y)


# In[8]:


rf_clf.oob_score_


# In[9]:


rf_clf2 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=666, oob_score=True, n_jobs=-1)
rf_clf2.fit(X, y)


# In[10]:


rf_clf2.oob_score_


# ### 使用Extra-Trees

# In[11]:


from sklearn.ensemble import ExtraTreesClassifier

et_clf = ExtraTreesClassifier(n_estimators=500, bootstrap=True, oob_score=True, random_state=666)
et_clf.fit(X, y)


# In[12]:


et_clf.oob_score_


# ### 集成学习解决回归问题

# In[13]:


from sklearn.ensemble import BaggingRegressor, BaggingClassifier, ExtraTreesRegressor

