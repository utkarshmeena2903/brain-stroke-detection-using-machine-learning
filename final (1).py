#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings  
warnings.filterwarnings('ignore')


# In[ ]:





# In[70]:


data=pd.read_csv(r'D:\python\stroke prediction\healthcare-dataset-stroke-data.csv')
data


# In[ ]:





# In[71]:


data=data.drop(['id'],axis=1)
data


# # MISSING

# In[72]:


data.info()


# In[73]:


data.describe()


# In[74]:


data['bmi'].value_counts()


# In[75]:


data['bmi'].fillna(data['bmi'].mean(),inplace=True)
data


# # OUTLINER	

# In[76]:


plt.rcParams["figure.figsize"]=(10,10)
data.plot(kind='box')
plt.show()


# In[ ]:





# # LABEL ENCODING

# In[77]:


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()


# In[78]:


gender=enc.fit_transform(data['gender'])
smoking_status=enc.fit_transform(data['smoking_status'])
work_type=enc.fit_transform(data['work_type'])
ever_married=enc.fit_transform(data['ever_married'])
Residence_type=enc.fit_transform(data['Residence_type'])


# In[79]:


data['work_type']=work_type
data['smoking_status']=smoking_status
data['gender']=gender
data['Residence_type']=Residence_type
data['ever_married']=ever_married
data


# # SPLITTING

# X ---train_X,test_X
# Y ---train_Y,test_Y
# RATIO= 80/20
# 

# In[80]:


X=data.drop('stroke',axis=1)


# In[81]:


X


# In[82]:


Y=data['stroke']


# In[83]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=20)


# In[ ]:





# In[84]:


X_train


# # normalise

# In[85]:


data.describe()


# In[86]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()


# In[87]:


X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)


# In[88]:


import pickle
import os


# In[89]:


scaler_path=os.path.join('D:/python/stroke prediction/','models/scaler.pkl')
with open(scaler_path,'wb') as scaler_file:
    pickle.dump(std,scaler_file)


# In[ ]:





# # saving the scalor object

# In[90]:


import pickle
import os


# In[91]:


scaler_path=os.path.join('D:/python/stroke prediction/','models/scaler.pkl')
with open(scaler_path,'wb') as scaler_file:
    pickle.dump(std,scaler_file) 


# In[ ]:





# In[92]:


X_train


# In[93]:


X_train_std


# In[94]:


X_test_std


# # training data

# # DECISION TREE

# In[95]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[96]:


dt.fit(X_train_std,Y_train)


# In[97]:


dt.feature_importances_


# In[98]:


X_train.columns


# In[99]:


Y_pred=dt.predict(X_test_std)


# In[100]:


from sklearn.metrics import accuracy_score


# In[101]:


acur_DT=accuracy_score(Y_test,Y_pred)


# In[102]:


acur_DT


# In[103]:


import joblib
model_path=os.path.join('D:/python/stroke prediction/','models/dt.sav')
joblib.dump(dt,model_path)


# # LOGISTIC REGRESSION 

# In[104]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[105]:


lr.fit(X_train_std,Y_train)


# In[51]:


Y_pred=lr.predict(X_test_std)


# In[52]:


acur_LR=accuracy_score(Y_test,Y_pred)


# In[53]:


acur_LR


# # KNN(K-NEARNEST NEIGHBOUR )

# In[54]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[55]:


knn.fit(X_train_std,Y_train)


# In[56]:


Y_pred=knn.predict(X_test_std)


# In[57]:


acur_knn=accuracy_score(Y_test,Y_pred)


# In[58]:


acur_knn


# # RANDOM FOREST

# In[59]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[60]:


rf.fit(X_train_std,Y_train)


# In[61]:


Y_pred=rf.predict(X_test_std)


# In[62]:


acur_rf=accuracy_score(Y_test,Y_pred)


# In[63]:


acur_rf


# In[ ]:





# In[ ]:





# In[64]:


plt.bar(['Decision Tree','Logistic Regression','KNN','RANDOM FOREST'],[acur_DT,acur_LR,acur_knn,acur_rf])
plt.xlabel("algorithms")
plt.ylabel("accuracy")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




