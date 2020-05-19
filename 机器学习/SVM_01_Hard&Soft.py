
# coding: utf-8

# ### scikit-learn中的SVM

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# 因为SVM只能处理二分类问题，因此该样例只能取两个分类来训练
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

X = X[y < 2, :2]
y = y[y < 2]


# In[22]:


plt.scatter(X[y==0, 0], X[y==0, 1], color='r')
plt.scatter(X[y==1, 0], X[y==1, 1], color='b')
plt.show()


# In[5]:


from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X)
X_standard = standardScaler.transform(X)


# In[6]:


from sklearn.svm import LinearSVC

svc = LinearSVC(C=1e9)
svc.fit(X_standard, y)


# In[7]:


# 封装一个函数，可绘制出决策边界情况
def plot_decision_boundary(model, axis):
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


# In[9]:


plot_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0, 0], X_standard[y==0, 1])
plt.scatter(X_standard[y==1, 0], X_standard[y==1, 1])
plt.show()


# In[24]:


svc2 = LinearSVC(C=0.01)
svc2.fit(X_standard, y)


# In[25]:


plot_decision_boundary(svc2, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0, 0], X_standard[y==0, 1])
plt.scatter(X_standard[y==1, 0], X_standard[y==1, 1])
plt.show()


# In[26]:


svc.coef_


# In[27]:


svc.intercept_


# In[28]:


def plot_svc_decision_boundary(model, axis):
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
    
    w = model.coef_[0]
    b = model.intercept_[0]
    
    # 有了w、b，决策边界就是w0 * x0 + w1 * x1 + b = 0
    # 即 x1 = -w0/w1 * x0 - b/w1
    plot_x = np.linspace(axis[0], axis[1], 200)
    # 同理，决策边界的上支撑边界就是w0 * x0 + w1 * x1 + b = 1
    # 即 x1 = -w0/w1 * x0 - b/w1 + 1/w1
    up_y = -w[0]/w[1] * plot_x - b/w[1] + 1/w[1]
    # 同理，决策边界的下支撑边界就是w0 * x0 + w1 * x1 + b = -1
    # 即 x1 = -w0/w1 * x0 - b/w1 - 1/w1
    down_y = -w[0]/w[1] * plot_x - b/w[1] - 1/w[1]
    
    # 做一下过滤，只绘制在axis[2]~axis[3]范围内的点
    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])
    
    plt.plot(plot_x[up_index], up_y[up_index], color='black')
    plt.plot(plot_x[down_index], down_y[down_index], color='black')


# In[29]:


plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0, 0], X_standard[y==0, 1])
plt.scatter(X_standard[y==1, 0], X_standard[y==1, 1])
plt.show()


# In[30]:


plot_svc_decision_boundary(svc2, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0, 0], X_standard[y==0, 1])
plt.scatter(X_standard[y==1, 0], X_standard[y==1, 1])
plt.show()

