
# coding: utf-8

# ### 验证数据集调整超参数使用

# ### 交叉验证

# In[1]:


import numpy as np
from sklearn import datasets


# In[2]:


digits = datasets.load_digits()
X = digits.data
y = digits.target


# ### 测试train_test_split

# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=666)


# In[4]:


from sklearn.neighbors import KNeighborsClassifier

best_score, best_p, best_k = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_score, best_p, best_k = score, p, k
            
print("Best K = ", best_k)
print("Best P = ", best_p)
print("Best Score = ", best_score)
print("1111")


# ### 使用交叉验证

# In[5]:


from sklearn.model_selection import cross_val_score

knn_clf = KNeighborsClassifier()
cross_val_score(knn_clf, X_train, y_train)


# In[6]:


best_score, best_p, best_k = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        score = cross_val_score(knn_clf, X_train, y_train)
        score = np.mean(score)
        if score > best_score:
            best_score, best_p, best_k = score, p, k
            
print("Best K = ", best_k)
print("Best P = ", best_p)
print("Best Score = ", best_score)


# In[7]:


best_knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=2, p=2)


# In[8]:


best_knn_clf.fit(X_train, y_train)
best_knn_clf.score(X_test, y_test)

