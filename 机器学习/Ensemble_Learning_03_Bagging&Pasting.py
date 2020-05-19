
# coding: utf-8

# ### Bagging 和 Pasting

# In[20]:


import numpy as np
import matplotlib.pyplot as plt


# In[21]:


from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)


# In[22]:


plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()


# In[23]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# ### 使用Bagging

# In[24]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                               n_estimators=500, max_samples=100,
                               bootstrap=True)


# In[25]:


bagging_clf.fit(X_train, y_train)


# In[26]:


bagging_clf.score(X_test, y_test)


# In[27]:


bagging_clf2 = BaggingClassifier(DecisionTreeClassifier(),
                               n_estimators=5000, max_samples=100,
                               bootstrap=True)
bagging_clf2.fit(X_train, y_train)


# In[30]:


bagging_clf2.score(X_test, y_test)

