
# coding: utf-8

# ### scikit-learn中的PCA

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# In[4]:


digits = datasets.load_digits()
X = digits.data
y = digits.target


# In[6]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


# In[7]:


X_train.shape


# In[9]:


X_test.shape


# In[11]:


get_ipython().run_cell_magic(u'time', u'', u'from sklearn.neighbors import KNeighborsClassifier\n\nknn_clf = KNeighborsClassifier()\nknn_clf.fit(X_train, y_train)')


# In[12]:


knn_clf.score(X_test, y_test)


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

