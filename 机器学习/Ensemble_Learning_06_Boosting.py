
# coding: utf-8

# ### Boosting

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)


# In[3]:


plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


# ### AdaBoosting

# In[5]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=500)
ada_clf.fit(X_train, y_train)


# In[6]:


ada_clf.score(X_test, y_test)


#  ### Gradient Boosting

# In[12]:


from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(max_depth=2, n_estimators=30)
gb_clf.fit(X_train, y_train)


# In[13]:


gb_clf.score(X_test, y_test)


# ### Boosting解决回归问题

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

