
# coding: utf-8

# ### 梯度下降法模拟

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


plot_x = np.linspace(-1, 6, 141)
plot_x


# In[3]:


plot_y = (plot_x - 2.5)**2 - 1


# In[4]:


plt.plot(plot_x, plot_y)
plt.show()


# In[5]:


# 使用θ对J进行求导，在该例中，J即为plot_y，θ即为plot_x，因此本例是使用plot_x对plot_y进行求导
def dJ(theta):
    return 2 * (theta - 2.5)


# In[6]:


# 定义J函数，本例中即为plot_y
def J(theta):
    return (theta - 2.5)**2 - 1


# In[7]:


# 定义θ初始值、学习率、最小差值(10的-8次方)
theta = 0.0
eta = 0.1
epsilon = 1e-8
theta_history = [theta]
while True:
    gradient = dJ(theta)
    theta_last = theta
    theta = theta - eta * gradient
    theta_history.append(theta)
    if abs(J(theta) - J(theta_last)) < epsilon:
        break


# In[8]:


print(theta)
print(J(theta))


# In[9]:


plt.plot(plot_x, J(plot_x))
plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
plt.show()


# In[10]:


print(len(theta_history))


# ### 将以上代码进行封装，即可得到梯度下降算法功能

# In[11]:


# 参数为初始θ，学习率，最小差值，最多计算次数
def gradient_descent(initial_theta, eta, epsilon=1e-8, n_iters=1e4):
    theta = initial_theta
    theta_history.append(initial_theta)
    i = 0
    
    while i < n_iters:
        gradient = dJ(theta)
        theta_last = theta
        theta = theta - (eta * gradient)
        theta_history.append(theta)
        if abs(J(theta) - J(theta_last)) < epsilon:
            break
        i += 1

def plot_theta_history():
    plt.plot(plot_x, J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
    plt.show()


# In[12]:


#eta = 0.01
theta_history = []
gradient_descent(0, 0.01)
plot_theta_history()
print(len(theta_history))


# In[13]:


theta_history = []
gradient_descent(0, 0.8)
plot_theta_history()
print(len(theta_history))

