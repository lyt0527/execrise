
# coding: utf-8

# ### Sigmoid函数

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 


# In[2]:


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# In[3]:


x = np.linspace(-10, 10, 500)
y = sigmoid(x)

plt.plot(x, y)
plt.show()


# In[4]:


# 值域在(0, 1)


# ### 逻辑回归中添加多项式特征

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# 设置一个二分类，分类边界为抛物线形
np.random.seed(666)
X = np.random.normal(0, 1, size=(200, 2))
y = np.array(X[:, 0]**2 + X[:, 1] < 1.5, dtype='int')
# 手工添加噪音，即把其中20个点强制变为1
for _ in range(20):
    y[np.random.randint(200)] = 1


# In[3]:


plt.scatter(X[y==0,0], X[y==0,1], color='blue')
plt.scatter(X[y==1,0], X[y==1,1], color='red')
plt.show()


# In[4]:


y


# In[6]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


# ### 使用scikit-sklearn中的逻辑回归

# In[7]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# In[8]:


log_reg.score(X_test, y_test)


# In[15]:


# 封装一个函数，可绘制决策边界的情况
def plot_decision_boindary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1)  
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
                    
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


# In[16]:


plot_decision_boindary(log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()


# ### 使用多项式回归进行逻辑回归

# In[57]:


from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

def Polynomial_logistic_regression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])


# In[58]:


## 将degree传入2，发现效果良好
poly_log_reg = polynomial_logistic_regression(degree=2)
poly_log_reg.fit(X_train, y_train)


# In[59]:


poly_log_reg.score(X_test, y_test)


# In[60]:


plot_decision_boindary(poly_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()


# ### degree传入20，出现过拟合

# In[77]:


poly_log_reg20 = polynomial_logistic_regression(degree=20)
poly_log_reg20.fit(X_train, y_train)


# In[78]:


poly_log_reg20.score(X_test, y_test)


# In[79]:


plot_decision_boindary(poly_log_reg20, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()


# ### 使用超参数C，penalty进行正则化

# In[80]:


# 修改polynomial_logistic_regression函数，先只传入C
def polynomial_logistic_regression2(degree, C):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C))
    ])


# In[81]:


# C传入0.1，则分类函数起的作用只有0.1，而正则化起的作用只有0.9
poly_log_reg2 = polynomial_logistic_regression2(degree=20, C=0.1)
poly_log_reg2.fit(X_train, y_train)


# In[82]:


poly_log_reg2.score(X_test, y_test)


# In[83]:


plot_decision_boindary(poly_log_reg2, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()


# In[88]:


def polynomial_logistic_regression3(degree, C, penalty):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C, penalty=penalty))
    ])


# In[102]:


poly_log_reg3 = polynomial_logistic_regression3(degree=20, C=0.1, penalty='l1')
poly_log_reg3.fit(X_train, y_train)


# In[103]:


poly_log_reg3.score(X_test, y_test)


# In[104]:


# 事实证明，因为degree太大（为20），所以使用l1正则化能够使得多个无需的特征项的值为0，效果更好
plot_decision_boindary(poly_log_reg3, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()

