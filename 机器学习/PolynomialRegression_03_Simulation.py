
# coding: utf-8

# ### -学习曲线-

# In[5]:


import numpy as np
import matplotlib.pyplot as plt


# In[6]:


np.random.seed(666)
x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size=100)


# In[7]:


plt.scatter(X, y)
plt.show()


# ### 学习曲线

# In[21]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)


# In[22]:


X_train.shape


# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_score = []
test_score = []
for i in range(1, 76):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train[:i], y_train[:i])
    
    y_train_predict = lin_reg.predict(X_train[:i])
    train_score.append(mean_squared_error(y_train[:i], y_train_predict))
    
    y_test_predict = lin_reg.predict(X_test)
    test_score.append(mean_squared_error(y_test, y_test_predict))


# In[24]:


plt.plot([i for i in range (1, 76)], np.sqrt(train_score), label='train')
plt.plot([i for i in range (1, 76)], np.sqrt(test_score), label='test')
plt.legend()
plt.show()


# In[29]:


# 将学习率曲线做成函数
def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
    
    train_score = []
    test_score = []
    for i in range(1, len(X_train)+1):
        algo.fit(X_train[:i], y_train[:i])
        
        y_train_predict = algo.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))
        
        y_test_predict = algo.predict(X_test)
        test_score.append(mean_squared_error(y_test, y_test_predict))
        
    plt.plot([i for i in range(1, len(X_train)+1)], np.sqrt(train_score), label="train")
    plt.plot([i for i in range(1, len(X_train)+1)], np.sqrt(test_score), label="test")
    
    plt.legend()
    plt.axis([0, len(X_train)+1, 0, 4])
    plt.show()
    
plot_learning_curve(LinearRegression(), X_train, X_test, y_train, y_test)


# In[36]:


# 同样，将多项式回归（多项式特征提取、数据标准化、线性回归）封装成函数
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])


# In[37]:


poly2_reg = PolynomialRegression(degree=2)
plot_learning_curve(poly2_reg, X_train, X_test, y_train, y_test)


# In[38]:


# 过拟合
poly2_reg = PolynomialRegression(degree=20)
plot_learning_curve(poly2_reg, X_train, X_test, y_train, y_test)

