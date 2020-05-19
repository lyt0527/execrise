
# coding: utf-8

# ## 岭回归

# In[15]:


import numpy as np
import matplotlib.pyplot as plt


# In[23]:


np.random.seed(42)
x = np.random.uniform(-3., 3., size=100)
X = x.reshape(-1, 1)
y = 0.5 * x + 3. + np.random.normal(0, 1, size=100)


# In[30]:


plt.scatter(x, y)
plt.show()


# In[31]:


# 将多项式回归（多项式特征提取、数据标准化、线性回归）封装成函数
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

def polynomial_regression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scalar", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])


# In[32]:


from sklearn.model_selection import train_test_split

np.random.seed(666)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[33]:


from sklearn.metrics import mean_squared_error

poly20_reg = polynomial_regression(degree=20)
poly20_reg.fit(X_train, y_train)

y20_predict = poly20_reg.predict(X_test)
mean_squared_error(y_test, y20_predict)


# In[36]:


# 将会之函数封装成函数，以便后面频繁使用
def plot_model(model):
    X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
    y_plot = model.predict(X_plot)
    
    plt.scatter(x, y)
    plt.plot(X_plot[:, 0], y_plot, color='r')
    plt.axis([-3., 3., 0., 6.])
    plt.show()


# In[37]:


plot_model(poly20_reg)


# ### 对该过拟合进行岭回归

# In[52]:


# 复用以上多项式回归函数，并进行修改
from sklearn.linear_model import Ridge

def ridge_regression(degree, alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scalar", StandardScaler()),
        ("ridge_reg", Ridge(alpha=alpha))
    ])


# In[53]:


# 使用岭回归
ridge1 = ridge_regression(20, 0.0001)
ridge1.fit(X_train, y_train)

y1_predict = ridge1.predict(X_test)
mean_squared_error(y_test, y1_predict)


# In[54]:


plot_model(ridge1)


# In[55]:


# 使用岭回归，0取100000
ridge4 = ridge_regression(20, 100000)
ridge4.fit(X_train, y_train)

y4_predict = ridge4.predict(X_test)
mean_squared_error(y_test, y4_predict)


# In[56]:


plot_model(ridge4)

