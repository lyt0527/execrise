
# coding: utf-8

# ### 梯度下降法的向量化

# In[1]:


import numpy as np
from sklearn import datasets


# In[2]:


boston = datasets.load_boston()
X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]


# In[3]:


from sklearn.cross_validation import train_test_split


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# In[5]:


from sklearn.linear_model import LinearRegression

lin_reg1 = LinearRegression()
get_ipython().magic(u'time lin_reg1.fit(x_train, y_train)')
lin_reg1.score(x_test, y_test)


# ### 使用梯度下降法

# In[8]:


def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta))**2) / len(X_b)
    except:
        return float('inf')
    
def dJ(theta, X_b, y):
    # res = np.empty(len(theta))
    # res[0] = np.sum(X_b.dot(theta) - y)
    # for i in range(1, len(theta)):
    #     res[i] = np.sum((X_b.dot(theta) - y).dot(X_b[:, i])) # 不写np.sum也可以，因为共识最后的Xn是一个一行一列的数据，点乘之后只有一个数
    # return res * 2 / len(X_b)
    
    # 使用梯度下降向量化
    return 2. / len(X_b) * X_b.T.dot(X_b.dot(theta) - y)

def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon = 1e-8):
    theta = initial_theta
    i_iter = 0
    
    while i_iter < n_iters:
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta - eta * gradient
        if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
            break;
        i_iter += 1
    return theta


# In[9]:


X_b = np.hstack([np.ones((len(X), 1)), X]) # 注意np.ones使用了两个()，是因为整体的(len(X), 1)作为np.ones的第一个参数，少了括号的话意思就不对了
initial_theta = np.zeros(X_b.shape[1])
eta = 0.000001 # 注意：eta设太大（比如0.01）就会造成报错，因为输入数据大小不一，学习率太高会造成数据过大


# In[10]:


get_ipython().magic(u'time theta = gradient_descent(X_b, y, initial_theta, eta)')


# In[11]:


theta


# ### 使用梯度下降法前将数据归一化

# In[12]:


from sklearn.preprocessing import StandardScaler


# In[16]:


standardScaler = StandardScaler()
standardScaler.fit(X_b)


# In[20]:


x_stand_scaler = stand_scaler.transform(X_b)


# In[21]:


get_ipython().magic(u'time theta = gradient_descent(X_b, y, initial_theta, eta, n_iters=1e6)')

