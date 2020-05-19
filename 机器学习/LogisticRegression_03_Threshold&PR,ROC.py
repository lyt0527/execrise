
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target==9] = 1
y[digits.target!=9] = 0


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


# In[5]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_predict = log_reg.predict(X_test)


# In[6]:


# 计算混淆矩阵
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_predict)


# In[7]:


# 计算精准率
from sklearn.metrics import precision_score

precision_score(y_test, y_predict)


# In[8]:


# 计算召回率
from sklearn.metrics import recall_score

recall_score(y_test, y_predict)


# In[9]:


# 计算f1分数
from sklearn.metrics import f1_score

f1_score(y_test, y_predict)


# In[10]:


# 查看测试集中每组数据计算出来的最终结果
log_reg.decision_function(X_test)


# In[12]:


# 将这些结果存起来
decision_scores = log_reg.decision_function(X_test)
np.min(decision_scores)


# In[13]:


np.max(decision_scores)


# In[14]:


# 重新制定threshold取5，看看混淆矩阵，精准率，召回率，f1分数
y_predict2 = np.array(decision_scores >= 5, dtype='int')
print(confusion_matrix(y_test, y_predict2))
print(precision_score(y_test, y_predict2))
print(recall_score(y_test, y_predict2))
print(f1_score(y_test, y_predict2))


# In[15]:


# 重新制定threshold取-5，看看混淆矩阵，精准率，召回率，f1分数
y_predict2 = np.array(decision_scores >= -5, dtype='int')
print(confusion_matrix(y_test, y_predict2))
print(precision_score(y_test, y_predict2))
print(recall_score(y_test, y_predict2))
print(f1_score(y_test, y_predict2))


# ### scikit-learn中的Precision-Recall曲线

# In[16]:


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, decision_scores)


# In[17]:


precisions.shape


# In[18]:


recalls.shape


# In[19]:


thresholds.shape


# In[20]:


# 绘制P-R曲线，注意因为threshold比precisions,recalls少一个元素，所以precisions和recalls需要去掉一个元素
plt.plot(thresholds, precisions[:-1])
plt.plot(thresholds, recalls[:-1])
plt.show()


# In[21]:


plt.plot(precisions, recalls)
plt.show()


# In[22]:


# 绘制ROC曲线
from sklearn.metrics import roc_curve

fprs, tprs, thresholds = roc_curve(y_test, decision_scores)
plt.plot(fprs, tprs)
plt.show()


# In[23]:


plt.plot(thresholds, fprs)
plt.plot(thresholds, tprs)
plt.show()

