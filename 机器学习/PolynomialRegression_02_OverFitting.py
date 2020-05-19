
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[5]:


# 在-3~3之间随机取100个值，这100个值符合均匀分布
x = np.random.uniform(-3.0, 3.0, size=100)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100) 


# In[6]:


plt.scatter(x, y)


# In[8]:


X = x.reshape(-1, 1)
X.shape


# ## 使用线性回归

# In[9]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.score(X, y)


# In[18]:


y_predict = lin_reg.predict(X)
plt.scatter(X, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
plt.show()


# In[19]:


from sklearn.metrics import mean_squared_error

y_predict = lin_reg.predict(X)
mean_squared_error(y, y_predict)


# ## 使用多项式回归

# In[20]:


# 将多项式回归（多项式特征提取、数据标准化、线性回归）封装成函数
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

def polynomial_regression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scalar', StandardScaler()),
        ('lin_reg', LinearRegression())
    ])


# In[23]:


poly2_reg = polynomial_regression(degree=2)
poly2_reg.fit(X, y)


# In[24]:


y2_predict = poly2_reg.predict(X)
mean_squared_error(y, y2_predict)


# In[25]:


plt.scatter(x, y)
plt.plot(np.sort(x), y2_predict[np.argsort(x)], color='r')
plt.show()


# ### 使用10次幂进行多项式回归（略微过拟合）

# In[26]:


poly_reg10 = polynomial_regression(10)
poly_reg10.fit(X ,y)
y_predict10 = poly_reg10.predict(X)


# In[28]:


plt.scatter(x, y)
plt.plot(np.sort(x), y_predict10[np.argsort(x)], color='r')
plt.show()


# ### 使用100次幂进行多项式回归（过拟合）

# In[29]:


poly_reg100 = polynomial_regression(100)
poly_reg100.fit(X, y)
y_predict100 = poly_reg100.predict(X)


# In[30]:


plt.scatter(x, y)
plt.plot(np.sort(x), y_predict100[np.argsort(x)], color='r')
plt.show()


# In[31]:


x_plot = np.linspace(-3, 3, 100).reshape(100, 1)
y_plot = poly_reg100.predict(x_plot)

plt.scatter(x, y)
plt.plot(x_plot[:,0], y_plot, color='r')
plt.axis([-3, 3, -1, 10])
plt.show()

