
# coding: utf-8

# ## LASSO回归

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


np.random.seed(42)
x = np.random.uniform(-3., 3., size=100)
X = x.reshape(-1, 1)
y = 0.5 * x + 3. + np.random.normal(0, 1, size=100)


# In[4]:


plt.scatter(x, y)
plt.show()


# In[5]:


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


# In[6]:


from sklearn.model_selection import train_test_split

np.random.seed(666)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[7]:


from sklearn.metrics import mean_squared_error

poly20_reg = polynomial_regression(degree=20)
poly20_reg.fit(X_train, y_train)

y20_predict =poly20_reg.predict(X_test)
mean_squared_error(y_test, y20_predict)


# In[8]:


# 将绘制图形封装成函数，以便后面频繁调用
def plot_model(model):
    X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
    y_plot = model.predict(X_plot)

    plt.scatter(x, y)
    plt.plot(X_plot[:, 0], y_plot, color='r')
    plt.axis([-3., 3., 0., 6.])
    plt.show()


# In[9]:


plot_model(poly20_reg)


# ### 对过拟合进行LASSO回归

# In[10]:


from sklearn.linear_model import Lasso

def LassoRegression(degree, alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lasso_reg", Lasso(alpha=alpha))
    ])


# In[11]:


lasso1_reg = LassoRegression(20, 0.01)
lasso1_reg.fit(X_train, y_train)

y1_predict = lasso1_reg.predict(X_test)
mean_squared_error(y_test, y1_predict)


# In[13]:


plot_model(lasso1_reg)


# In[14]:


lasso2_reg = LassoRegression(20, 0.1)
lasso2_reg.fit(X_train, y_train)

y2_predict = lasso2_reg.predict(X_test)
mean_squared_error(y_test, y2_predict)


# In[15]:


plot_model(lasso2_reg)


# In[16]:


lasso3_reg = LassoRegression(20, 1)
lasso3_reg.fit(X_train, y_train)

y3_predict = lasso3_reg.predict(X_test)
mean_squared_error(y_test, y3_predict)


# In[17]:


plot_model(lasso3_reg)

