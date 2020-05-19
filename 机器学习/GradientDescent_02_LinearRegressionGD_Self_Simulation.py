
# coding: utf-8

# ### 在线性回归模型中使用梯度下降

# In[22]:


import numpy as np
import matplotlib.pyplot as plt


# In[23]:


x = 2 * np.random.random(size=100)
y = x * 3. + 4. + np.random.normal(size=100)


# In[24]:


X = x.reshape(-1, 1)
X.shape


# In[25]:


plt.scatter(x, y)
plt.show()


# ### 使用梯度下降进行训练

# ![](image\GradientDescent.png)

# In[26]:


def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta))**2) / len(X_b)
    except:
        return float('inf')


# In[35]:


def dJ(theta, X_b, y):
    res = np.empty(len(theta))
    res[0] = np.sum(X_b.dot(theta) - y)
    for i in range(1, len(theta)):
        res[i] = np.sum((X_b.dot(theta) - y).dot(X_b[:, i]))
    return res * 2 / len(X_b)


# In[36]:


def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon = 1e-8):
    theta = initial_theta
    i_iter = 0
    
    while i_iter < n_iters:
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta - eta * gradient
        
        if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
            break
            
        i_iter += 1
        
    return theta


# In[37]:


X_b = np.hstack([np.ones((len(X), 1)), X]) # 注意np.ones使用了两个()，是因为整体的(len(X), 1)作为np.ones的第一个参数，少了括号的话意思就不对了
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01


# In[38]:


theta = gradient_descent(X_b, y, initial_theta, eta)


# In[39]:


theta


# In[ ]:




