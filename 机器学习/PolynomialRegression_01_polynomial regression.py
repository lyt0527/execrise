
# coding: utf-8

# ## 多项式回归

# In[1]:


import numpy as np
import matplotlib.pylab as plt


# In[2]:


x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)


# In[3]:


y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)


# In[4]:


plt.scatter(x, y)
plt.show()


# ## 解决方案，添加一个特征

# In[5]:


(X**2).shape


# In[13]:


X2 = np.hstack([X, X**2])


# In[15]:


X2.shape


#  ## scikit-learn中的多项式回归和Pipeline

# In[11]:


import numpy as np
import matplotlib.pylab as plt


# In[12]:


x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)


# In[13]:


from sklearn.preprocessing import PolynomialFeatures


# In[14]:


poly = PolynomialFeatures(degree=2)
poly.fit(X)
X2 = poly.transform(X)


# In[15]:


X2.shape


# In[16]:


X2[:5,:]


# In[17]:


X[:5,:]


# In[29]:


from sklearn.linear_model import LinearRegression


# In[30]:


lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)


# In[32]:


plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()


# In[33]:


lin_reg2.coef_


# In[34]:


lin_reg2.intercept_


# ## 关于PolynomialFeatures

# In[30]:


X = np.arange(1, 11).reshape(-1, 2)


# In[31]:


X.shape


# In[32]:


X


# In[36]:


poly = PolynomialFeatures(degree=3)
poly.fit(X)
X3 = poly.transform(X)


# In[37]:


X3.shape


# In[38]:


X3


# ## Pipeline

# In[54]:


x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)


# In[55]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

poly_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("std_scaler", StandardScaler()),bubu
    ("lin_reg", LinearRegression())
])


# In[56]:


poly_reg.fit(X, y)
y_predict = poly_reg.predict(X)


# In[58]:


plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
plt.show()

