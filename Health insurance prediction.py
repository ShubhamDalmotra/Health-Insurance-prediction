#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


os.chdir(r'F:\Data science\Machine learning\Health Insurance Prediction')


# In[3]:


df=pd.read_csv('insurance.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.shape


# In[8]:


df.describe(include ='all')


# # Convert string columns to numerical values

# In[9]:


df['sex'].unique()


# In[10]:


df['sex']=df['sex'].map({'female':0, 'male':1})


# In[11]:


df.head()


# In[12]:


df['smoker'].unique()


# In[13]:


df['smoker']=df['smoker'].map({'no':0, 'yes':1})


# In[14]:


df.head()


# In[15]:


df['region'].unique()


# In[16]:


df['region']=df['region'].map({'southwest':1, 'southeast':2,'northwest':3,'northeast':4})


# In[17]:


df.head()


# Store feature matix in X and Y----
# Y is dependent variable and X is independent variables

# In[18]:


y= df['charges']


# In[19]:


x=df.drop(['charges'], axis =1)


# In[20]:


x


# In[21]:


y


# # train test split 

# In[45]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.20, random_state=42)


# In[54]:


y_train


# In[46]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()


# In[47]:


reg.fit(x_train,y_train)


# In[50]:


reg.intercept_


# In[51]:


from sklearn.metrics import r2_score


# In[52]:


r2_score(y,reg.predict(x))


# # Stats regression model

# In[27]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
import statsmodels.api as sm


# In[25]:


X_sm= sm.add_constant(x)


# In[26]:


X_sm


# In[29]:


reg=sm.OLS(y,X_sm).fit()


# In[30]:


print(reg.summary())


# In[62]:


user_input = [[19 ,1 ,22.705 ,1 ,1, 2]]


# In[63]:


predicted_value = reg.predict(user_input)


# In[64]:


print("Predicted value:", predicted_value)


# In[65]:


df


# In[ ]:




