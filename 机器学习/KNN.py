
# coding: utf-8

# ### KNN算法

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.363808831, 3.369360954],
              [3.582294042, 4.699179110],
              [2.280362439, 2.866990263],
              [7.423636942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792753481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0 , 1, 1, 1, 1 ,1]


# In[8]:


X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)


# In[9]:


X_train


# In[10]:


y_train


# In[12]:


plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], color='g')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color='r')
plt.show()


# In[13]:


x = np.array([8.093607318, 3.365731514])


# In[14]:


plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], color='g')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color='r')
plt.scatter(x[0], x[1], color='b')
plt.show()


# ## KNN过程

# In[15]:


from math import sqrt
distance = []
for x_train in X_train:
    d = sqrt(np.sum((x_train - x)**2))
    distance.append(d)


# In[16]:


distance


# In[17]:


distance = [sqrt(np.sum((x_train - x)**2))for x_train in X_train]


# In[18]:


distance


# In[19]:


np.sort(distance)


# In[23]:


nearest = np.argsort(distance)


# In[24]:


k = 6


# In[25]:


topK_y = [y_train[i] for i in nearest[:k]]


# In[26]:


topK_y


# In[27]:


from collections import Counter
Counter(topK_y)


# In[28]:


votes = Counter(topK_y)


# In[31]:


votes.most_common(1)[0][0]


# In[34]:


predict_y = votes.most_common(1)[0][0]


# In[35]:


predict_y


# ## 使用scikit-learn中KNN

# In[36]:


from sklearn.neighbors import KNeighborsClassifier


# In[37]:


KNN_classifier = KNeighborsClassifier(n_neighbors=6)


# In[38]:


KNN_classifier.fit(X_train, y_train)


# In[40]:


## KNN_classifier.predict(x)


# In[41]:


X_predict = x.reshape(1, -1)


# In[42]:


X_predict


# In[45]:


y_predict = KNN_classifier.predict(X_predict)


# In[46]:


y_predict


# In[48]:


y_predict[0]

