
# coding: utf-8

# ### 多分类问题中的混淆矩阵

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)


# In[5]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)


# In[6]:


y_predict = log_reg.predict(X_test)


# In[7]:


# 查看混淆矩阵，并用灰度显示可以看出，对于总体样例来说，预测对的情况还是占大部分的
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_predict)


# In[9]:


cfm = confusion_matrix(y_test, y_predict)
plt.matshow(cfm, cmap=plt.cm.gray)
plt.show()


# In[10]:


row_sums = np.sum(cfm, axis=1)
err_matrix = cfm / row_sums
np.fill_diagonal(err_matrix, 0)
err_matrix


# In[11]:


plt.matshow(err_matrix, cmap=plt.cm.gray)
plt.show()

