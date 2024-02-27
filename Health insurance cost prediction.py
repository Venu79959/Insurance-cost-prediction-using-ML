#!/usr/bin/env python
# coding: utf-8

# # 2. Health Insurance cost prediction 

# In[2]:


import pandas as pd


# In[3]:


data=pd.read_csv('insurance.csv')


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.shape


# In[7]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# In[8]:


data.info()


# In[17]:


data.isnull()


# In[14]:


data.describe()


# In[15]:


data.describe(include='all')


# In[18]:


data.head()


# In[19]:


data['sex'].unique()


# In[27]:


data['sex'] = data['sex'].map({'female':0,'male':1})


# In[28]:


data.head()


# In[29]:


data['smoker']=data['smoker'] = data['smoker'].map({'no':0,'yes':1})


# In[30]:


data.head()


# In[31]:


data['region']=data['region'].map({'southwest':1,'southeast':2,'northwest':3,'northeast':4})


# In[32]:


data.head()


# In[34]:


X=data.drop(['charges'],axis=1)
y=data['charges']


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split (X,y,test_size=0.2,random_state=42)


# In[36]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[38]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[39]:


svm=SVR()
svm.fit(X_train,y_train)


# In[40]:


rf=RandomForestRegressor()
rf.fit(X_train,y_train)


# In[41]:


gr=GradientBoostingRegressor()
gr.fit(X_train,y_train)


# In[42]:


y_pred1=lr.predict(X_test)
y_pred2=svm.predict(X_test)
y_pred3=rf.predict(X_test)
y_pred4=gr.predict(X_test)
df1=pd.DataFrame({'Actual':y_test,'Lr':y_pred1,'svm':y_pred2,'rf':y_pred3,'gr':y_pred4})


# In[43]:


df1


# In[44]:


import matplotlib.pyplot as plt


# In[47]:


plt.subplot(221)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['Lr'].iloc[0:11],label='Lr')
plt.legend()

plt.subplot(222)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['svm'].iloc[0:11],label='svm')
plt.legend()

plt.subplot(223)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['rf'].iloc[0:11],label='rf')
plt.legend()

plt.subplot(224)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['gr'].iloc[0:11],label='gr')
plt.tight_layout()
plt.legend()


# In[48]:


from sklearn import metrics
score1=metrics.r2_score(y_test,y_pred1)
score2=metrics.r2_score(y_test,y_pred2)
score3=metrics.r2_score(y_test,y_pred3)
score4=metrics.r2_score(y_test,y_pred4)
print(score1,score2,score3,score4)


# In[50]:


s1=metrics.mean_absolute_error(y_test,y_pred1)
s2=metrics.mean_absolute_error(y_test,y_pred2)
s3=metrics.mean_absolute_error(y_test,y_pred3)
s4=metrics.mean_absolute_error(y_test,y_pred4)
print(s1,s2,s3,s4)


# In[104]:


data=({'age':40,
       'sex':1,
       'bmi':40.30,
       'children':4,
       'smoker':1,
       'region':2})


# In[105]:


data


# In[108]:


import numpy as np
df = pd.DataFrame(data,index=[0])
df


# In[109]:


new_pred=gr.predict(df)
print(new_pred)


# In[111]:


gr=GradientBoostingRegressor()
gr.fit(X,y)
import joblib
joblib.dump(gr,'model_joblib_gr')
model=joblib.load('model_joblib_gr')
model.predict(df)


# In[ ]:




