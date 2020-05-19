
# coding: utf-8

# ### 使用梯度上升法求解主成分

# In[40]:


import numpy as np
import matplotlib.pyplot as plt


# In[41]:


X = np.empty((100, 2))
X[:, 0] = np.random.uniform(0., 100., size=100)
X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)


# In[42]:


plt.scatter(X[:, 0], X[:, 1])
plt.show()


# ### demean

# In[43]:


def demean(X):
    return X - np.mean(X, axis=0)


# In[44]:


X_demean = demean(X)


# In[45]:


plt.scatter(X_demean[:, 0], X_demean[:, 1])
plt.show()


# In[46]:


np.mean(X_demean[:, 0])


# In[47]:


np.mean(X_demean[:, 1])


# ### 梯度上升法求主成分（第一主成分）

# In[48]:


def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X) 


# In[49]:


def df(w, X):
    return (2. / len(X)) * X.T.dot(X.dot(w))

def df_math(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)


# In[50]:


#测试df_math求出的结果是否正确
def df_debug(w, X, epsilon=0.0001):
    res = np.empty(len(X))
    for i in range(len(w)):
        w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X) - f(w_2, X)) / (2 * epsilon)
    return res


# In[51]:


# 将w向量的模化为1
def direction(w):
    return w / np.linalg.norm(w)

def first_component(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    w = direction(initial_w)
    cur_iter = 0
    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)
        if(abs(f(w, X) - f(last_w, X)) < epsilon):
            break
            
        cur_iter += 1
        
    return w


# In[52]:


initial_w = np.random.random(X.shape[1]) #注意：不能用0向量开始
initial_w


# In[53]:


eta = 0.001


# In[54]:


## 注意：不能使用StandarScalar标准化数据 


# In[55]:


first_component(df, X_demean , initial_w, eta)


# In[56]:


w = first_component(df_math, X_demean, initial_w, eta)
plt.scatter(X_demean[:, 0], X_demean[:, 1])
plt.plot([0, w[0]*30], [0, w[1]*30], color='r')
plt.show()


# ### 求第二主成分

# In[57]:


X2 = np.empty(X.shape)
for i in range(len(X)):
    X2[i] = X[i] - X[i].dot(w) * w
# 以上也可以直接写成：X2 = X - X.dot(w).reshape(-1, 1) * w


# In[58]:


plt.scatter(X2[:, 0], X2[:, 1])
plt.show()


# In[61]:


w2 = first_component(df, X2, initial_w, eta)
w2


# In[62]:


w.dot(w2)


# ### 综上，求前n个主成分

# In[36]:


def first_n_components(n, X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
    X_pca = X.copy()
    X_pca = demean(X_pca)
    res = []
    for i in range(n):
        initial_w = np.random.random(X_pca.shape[1])
        w = first_component(df, X_pca, initial_w, eta)
        res.append(w)
        X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
    return res


# In[38]:


#尝试求前n个主成分（注意，由于本例中X只存在两个维度的数据，所以最多只能求2个主成分）
res = first_n_components(2, X, initial_w, eta)
res


# In[63]:


res = first_n_components(3, X, initial_w, eta)
res


# In[64]:


res = first_n_components(4, X, initial_w, eta)
res

