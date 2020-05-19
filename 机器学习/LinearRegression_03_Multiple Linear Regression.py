
# coding: utf-8

# ### 实现多元线性回归模型

# In[12]:


import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split


# In[13]:


bosten = datasets.load_boston()

X = bosten.data
y = bosten.target


# In[14]:


X.shape


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=666)
print(x_train.shape)


# ### 使用scikit-learn进行线性回归

# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


lin_reg = LinearRegression()


# In[18]:


lin_reg.fit(x_train, y_train)


# In[19]:


lin_reg.coef_


# In[20]:


lin_reg.intercept_


# In[21]:


lin_reg.score(x_test, y_test)


# ### 使用K（聚类）KNN回归

# In[24]:


from sklearn.neighbors import KNeighborsRegressor


# In[25]:


knn_reg = KNeighborsRegressor()


# In[27]:


knn_reg.fit(x_train, y_train)
knn_reg.score(x_test, y_test)


# In[31]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]
    },
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        "p": [i for i in range(1, 6)]
    }
]
knn_reg = KNeighborsRegressor()
grid_search = GridSearchCV(knn_reg, param_grid, n_jobs=-1, verbose=1)
grid_search.fit(x_train, y_train)


# In[32]:


grid_search.best_params_


# In[33]:


grid_search.best_score_


# In[34]:


grid_search.best_estimator_.score(x_test, y_test)

