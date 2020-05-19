
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# In[2]:


data_type={'acoustic_data': np.int16, 'time_to_failure': np.float64}
#data = pd.read_csv("../../data/LANLEarthquakePrediction/train.csv", nrows=8000000, dtype=data_type)
#data.to_csv("../data/train.csv", encoding="utf-8", index=False)
data = pd.read_csv("../../train.csv", dtype=data_type)


# In[3]:


data.index


# In[20]:


data.shape


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.plot(data.index, data.time_to_failure, color="r")
ax.set_title("8 million")
ax.set_xlabel("Index")
ax.set_ylabel("time_to_failure")


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.plot(data.index, data.acoustic_data, color="b")
ax.set_title("8 million")
ax.set_xlabel("Index")
ax.set_ylabel("acoustic_data")


# In[18]:


fig, ax = plt.subplots(1, 1, figsize=(20, 5))
ax.plot(data.index[:50000], data.time_to_failure[:50000], color="r")
plt.legend(["time_to_failure"])
ax.set_title("50000")
ax.set_xlabel("Index")
ax.set_ylabel("time_to_failure")


# In[49]:


# fig, ax = plt.subplots(1, 1, figsize=(20, 5))
# ax.plot(data.index[:49999], np.diff(data.time_to_failure[:50000]), color="b")
# ax.set_title("50000")
# ax.set_xlabel("Index")
# ax.set_ylabel("time_to_failure")


# In[17]:


fig, ax = plt.subplots(1, 1, figsize=(20, 5))
ax.plot(data.index[:4000], data.time_to_failure[:4000], color="green")
plt.legend(["time_to_failure"])
ax.set_title("4000")
ax.set_xlabel("Index")
ax.set_ylabel("time_to_failure")


# In[32]:


np.diff(data.acoustic_data[:49999])


# In[23]:


data.index[:50000]


# In[16]:


fig, ax = plt.subplots(1, 1, figsize=(20, 5))
ax.plot(data.index[:4000], data.acoustic_data[:4000], color="b")
plt.legend(["acoustic_data"])
ax.set_title("50000")
ax.set_xlabel("Index")
ax.set_ylabel("acoustic_data")


# In[9]:


#data1 = pd.read_csv("../../data/LANLEarthquakePrediction/test/seg_0012b5.csv", nrows=10000, dtype=data_type)
#data1.shape


# In[14]:


plt.scatter(data.index[:8000000], data.time_to_failure[:8000000])
plt.show()


# In[5]:


data.describe()


# In[10]:


data.head(5)


# In[32]:


test_path = "../../data/LANLEarthquakePrediction/test/"
files = os.listdir(test_path)
print(files[:4])
len(files)


# In[30]:


data1 = pd.read_csv('../../data/LANLEarthquakePrediction/sample_submission.csv')
data1.head(5)


# In[31]:


len(data1)


# In[3]:


data["per_25"] = data.acoustic_data.rolling(window=50).quantile(0.25)
data["per_50"] = data.acoustic_data.rolling(window=50).quantile(0.5)
data["per_75"] = data.acoustic_data.rolling(window=50).quantile(0.75)
data["iqr"] = data.per_75 - data.per_25
data["min"] = data.acoustic_data.rolling(window=50).min()
data["max"] = data.acoustic_data.rolling(window=50).max()
data["skewness"] = data.acoustic_data.rolling(window=50).skew()
data["kurtosis"] = data.acoustic_data.rolling(window=50).kurt()


# In[38]:


X_train_25 = np.array(data["per_25"][50:1000001]).reshape(-1, 1)
X_train_25.shape
y_train = np.array(data["time_to_failure"][50:1000001]).reshape(-1, 1)
y_train.shape


# In[41]:


plt.scatter(X_train_25, y_train, color='r')
plt.scatter(X_train_min, y_train, color='b')
plt.show()


# In[15]:


from sklearn.preprocessing import StandardScaler

sds = StandardScaler()
sds.fit(X_train_25, y_train)


# In[17]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, random_state=666, oob_score=True, n_jobs=-1)
rf_clf.fit(X_train_25, y_train.astype("int"))


# In[18]:


rf_clf.oob_score_


# In[20]:


rf_clf2 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=666, oob_score=True, n_jobs=-1)
rf_clf2.fit(X_train_25, y_train.astype("int"))


# In[21]:


rf_clf2.oob_score_


# In[25]:


X_train_min = np.array(data["min"][50:1000001]).reshape(-1 ,1)
X_train_min.shape


# In[26]:


sds.fit(X_train_min, y_train)


# In[27]:


rf_clf.fit(X_train_min, y_train.astype("int"))


# In[28]:


rf_clf.oob_score_


# ### 逻辑回归

# In[33]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit()

