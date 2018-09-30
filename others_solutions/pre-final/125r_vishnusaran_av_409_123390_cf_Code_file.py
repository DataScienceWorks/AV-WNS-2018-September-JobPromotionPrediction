
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing train test dataset

# In[41]:

train = pd.read_csv("train_LZdllcl.csv")
test = pd.read_csv("test_2umaH9m.csv")


# In[42]:

train.shape, test.shape


# Missing values

# In[43]:

train.isnull().sum()


# In[44]:

test.isnull().sum()


# In[45]:

train.education.fillna("not_mentioned", inplace=True)


# In[46]:

test.education.fillna("not_mentioned", inplace=True)


# In[47]:

train.previous_year_rating.fillna(0, inplace=True)
test.previous_year_rating.fillna(0, inplace=True)


# Categorical variables

# In[48]:

train.ix[train["gender"]=="m", "gender"] = 0
train.ix[train["gender"]=="f", "gender"] = 1


# In[49]:

test.ix[test["gender"]=="m", "gender"] = 0
test.ix[test["gender"]=="f", "gender"] = 1


# In[50]:

train.dtypes


# In[51]:

test.dtypes


# In[52]:

train['gender']=train['gender'].astype(int)
test['gender']=test['gender'].astype(int)


# In[53]:

test.dtypes


# In[54]:

df_train=pd.get_dummies(train, columns=["department", "region","education","recruitment_channel"], prefix=["department", "region","education","recruitment_channel"])


# In[55]:

df_test=pd.get_dummies(test, columns=["department", "region","education","recruitment_channel"], prefix=["department", "region","education","recruitment_channel"])


# In[56]:

df_train.head()


# In[57]:

df_test.head()


# In[58]:

x_train = df_train.drop(["employee_id","is_promoted"],axis=1).values


# In[59]:

y_train = df_train.ix[:,"is_promoted"]


# In[61]:

x_test = df_test.drop(["employee_id"],axis=1).values


# In[65]:

df_train.is_promoted.value_counts()


# In[67]:

(y_train==1).sum()


# # Model training

# After different interations and parameter tuning I select lightgbm as classifier with n_estimators=85 
# and probability thresold = 0.25

# In[68]:

import lightgbm as lgb


# In[191]:

model = lgb.LGBMClassifier(n_estimators=85,min_child_samples=20)
model.fit(x_train,y_train)


# In[192]:

pred_prob = model.predict_proba(x_test)[:,1]


# In[198]:

y_pred = np.where(pred_prob > 0.25,1,0)


# In[199]:

Submission = pd.DataFrame({"employee_id":df_test.ix[:,"employee_id"].values,"is_promoted":y_pred})


# In[200]:

Submission.head()


# In[201]:

Submission.shape


# In[202]:

Submission.is_promoted.value_counts()


# In[203]:

Submission.to_csv("Submission15.csv")


# In[ ]:




