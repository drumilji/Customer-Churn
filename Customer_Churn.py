#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df=pd.read_csv("C:/Users/abcdr/OneDrive/Desktop/DataSets/customer_churn.csv")


# In[5]:


df.head(4)


# In[6]:


gend=df['gender'].value_counts()


# In[7]:


customer_5=df.iloc[:,5]


# In[8]:


customer_5.head(5)


# In[9]:


customer_15=df.iloc[:,15]


# In[10]:


customer_15.head()


# In[11]:


senior_male_electronic=df[(df['gender']=='Male') & (df['SeniorCitizen']==1) & (df['PaymentMethod']=='Electronic check')]


# In[12]:


senior_male_electronic.head()


# In[13]:


customer_total_tenure=df[(df['tenure']>70)| (df['MonthlyCharges']>100)]


# In[14]:


customer_total_tenure.head()


# In[15]:


two_mail_yes=df[(df['Contract']=='Two year')&(df['PaymentMethod']=='Mailed check')&(df['Churn']=='Yes')]


# In[16]:


two_mail_yes


# In[17]:


customer_333=df.sample(n=333)


# In[18]:


customer_333.head()


# In[19]:


df['Churn'].value_counts()


# In[20]:


plt.bar(df['InternetService'].value_counts().keys().tolist(),df['InternetService'].value_counts().tolist(),color='Red')
plt.xlabel("Categories OF INTERNET SERVICE")
plt.ylabel("COUNTS")
plt.title("Distribution Of Internet Service")
plt.grid='TRUE'


# In[21]:


plt.hist(df['tenure'],bins=30,color='red')
plt.title("Distribution Of Tenure")


# In[22]:



help(plt.boxplot)


# In[23]:


plt.scatter(x=df['tenure'],y=df['MonthlyCharges'])
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.title("MonthlyCharges vs Tenure")


# In[24]:


df.boxplot(column=['tenure'],by=['Contract'])


# In[25]:


plt.scatter(x=df['tenure'],y=df['MonthlyCharges'])


# In[26]:


from sklearn.model_selection import train_test_split
X=df[['tenure']]
Y=df[['MonthlyCharges']]


# In[27]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


# In[28]:


X_train.shape,Y_train.shape


# In[29]:


Y_test.shape,X_test.shape


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


lm=LinearRegression()


# In[32]:


lm.fit(X_train,Y_train)


# In[33]:


lm.fit_intercept


# In[34]:


lm.coef_


# In[35]:


y_pred=lm.predict(X_test)


# In[36]:


from sklearn.metrics import mean_squared_error


# In[37]:


mse=mean_squared_error(Y_test,y_pred)


# In[38]:


np.sqrt(mse)


# In[39]:


x=df[['MonthlyCharges']]
y=df[['Churn']]


# In[40]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35)


# In[41]:


x_train.shape


# In[42]:


x_test.shape


# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


glm=LogisticRegression()


# In[45]:


glm.fit(x_train,y_train)


# In[46]:


y_pred=glm.predict(x_test)


# In[47]:


from sklearn.metrics import confusion_matrix


# In[48]:


confusion_matrix(y_test,y_pred)


# In[49]:


from sklearn.metrics import accuracy_score


# In[50]:


accuracy_score(y_test,y_pred)


# In[51]:


x=df[['tenure','MonthlyCharges']]


# In[52]:


y=df[['Churn']]


# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[55]:


from sklearn.linear_model import LogisticRegression


# In[56]:


glm=LogisticRegression()


# In[57]:


glm.fit(x_train,y_train)


# In[58]:


y_pred=glm.predict(x_test)


# In[59]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[60]:


accuracy_score(y_test,y_pred)


# In[61]:


confusion_matrix(y_test,y_pred)


# In[62]:


(913+188)/(913+114+188+194)


# In[63]:


x=df[['tenure']]


# In[64]:


y=df[['Churn']]


# In[65]:


from sklearn.model_selection import train_test_split


# In[66]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[67]:


from sklearn.tree import DecisionTreeClassifier


# In[68]:


treee=DecisionTreeClassifier()


# In[69]:


treee.fit(x_train,y_train)


# In[70]:


treee.predict(x_test)


# In[71]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[72]:


accuracy_score(y_test,treee.predict(x_test))


# In[73]:


x=df[['tenure','MonthlyCharges']]


# In[74]:


y=df[['Churn']]


# In[75]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[76]:


x_train.shape


# In[77]:


y_test.shape


# In[78]:


from sklearn.ensemble import RandomForestClassifier


# In[79]:


forest=RandomForestClassifier()


# In[80]:


forest.fit(x_train,y_train)


# In[81]:


forest.predict(x_test)


# In[82]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[83]:


confusion_matrix(y_test,forest.predict(x_test))


# In[84]:


accuracy_score(y_test,forest.predict(x_test))


# In[85]:


import pickle
filename = 'customer.pkl'
pickle.dump(glm, open(filename, 'wb'))


# In[ ]:




