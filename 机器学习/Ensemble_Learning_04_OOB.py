
# coding: utf-8

# ### oob和更多的Bagging相关

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


# ### 使用oob

# In[6]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                               n_estimators=500, max_samples=100,
                               bootstrap=True, oob_score=True)


# In[7]:


bagging_clf.fit(X, y)


# In[8]:


bagging_clf.oob_score_


# ### n_jobs

# In[9]:


get_ipython().run_cell_magic(u'time', u'', u'bagging_clf = BaggingClassifier(DecisionTreeClassifier(),\n                               n_estimators=500, max_samples=100,\n                               bootstrap=True, oob_score=True)\nbagging_clf.fit(X, y)')


# In[12]:


get_ipython().run_cell_magic(u'time', u'', u'bagging_clf = BaggingClassifier(DecisionTreeClassifier(),\n                               n_estimators=500, max_samples=100,\n                               bootstrap=True, oob_score=True,\n                               n_jobs=-1)\nbagging_clf.fit(X, y)')


# ### bootstrap_features

# In[16]:


random_subspaces_clf = BaggingClassifier(DecisionTreeClassifier(),
                               n_estimators=500, max_samples=500,
                               bootstrap=True, oob_score=True,
                               n_jobs=-1,
                               max_features=1, bootstrap_features=True)
random_subspaces_clf.fit(X, y)
random_subspaces_clf.oob_score_


# In[18]:


random_patches_clf = BaggingClassifier(DecisionTreeClassifier(),
                               n_estimators=500, max_samples=100,
                               bootstrap=True, oob_score=True,
                               n_jobs=-1,
                               max_features=1, bootstrap_features=True)
random_patches_clf.fit(X, y)
random_patches_clf.oob_score_

