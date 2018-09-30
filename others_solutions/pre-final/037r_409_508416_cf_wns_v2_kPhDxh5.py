
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time, re, string
import warnings
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm_notebook as tqdm

warnings.filterwarnings('ignore')
pd.options.display.max_columns = 100


# In[2]:


train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
sub = pd.read_csv('input/sample_submission.csv')
train.head()


# In[3]:


train.isnull().sum()


# In[4]:


train.nunique()


# In[5]:


def categorical_data(df,col):
    dff = df.copy()
    dff.drop('employee_id',axis=1,inplace=True)
    dff['dept&edu'] = dff['department'] + dff['education']
    dff['gender+edu'] = dff['gender'] + dff['education']
    dff = pd.DataFrame({col: dff[col].astype('category').cat.codes for col in dff}, index=dff.index)
    dff['awards_per_years'] = dff['awards_won?']/dff['length_of_service']
    dff['total_score'] = dff['avg_training_score']*dff['no_of_trainings']
    return dff


# In[6]:


col = ['department','region','education','gender','recruitment_channel','dept&edu','gender+edu','dept&reg']
train_df = categorical_data(train,col)
test_df = categorical_data(test,col)
train_df.head()


# In[7]:


train_df.nunique()


# In[9]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb

X_trn, X_val, Y_trn, Y_val = train_test_split(train_df.loc[:, train_df.columns != 'is_promoted'], train_df.is_promoted, test_size=0.1, shuffle=True, 
                                              random_state=42)

print ("Train_shape: " + str(X_trn.shape))
print ("Val_shape: " + str(X_val.shape))
print ("No of positives in train: " + str(Y_trn.sum()))
print ("No of positives in val: " + str(Y_val.sum()))


# In[ ]:


from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)
    return 'f1', f1_score(y_true, y_hat), True

lgb_train = lgb.Dataset(X_trn, Y_trn)
lgb_eval = lgb.Dataset(X_val, Y_val)
evals_result = {} 

params = {
    'task': 'train',
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 11,
    'learning_rate': 0.01,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
}

print('Start training...')

gbm = lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=[lgb_eval, lgb_train], 
                valid_names=['val', 'train'], early_stopping_rounds=2000, feval=lgb_f1_score,
                evals_result=evals_result, verbose_eval=500)


# In[65]:


lgb.plot_metric(evals_result, metric='f1')


# In[13]:


import matplotlib.pyplot as plt
# %matplotlib inline

# print('Plot metrics during training...')
# ax = lgb.plot_metric(evals_result, metric='auc', figsize=(7, 5))
# plt.show()


# In[44]:


print('Plot feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=15, figsize=(10, 6))
plt.show()


# In[45]:


y_trn_pred = gbm.predict(X_trn, num_iteration=gbm.best_iteration)
y_val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
y_train_pred = gbm.predict(train_df.loc[:, train_df.columns != 'is_promoted'], num_iteration=gbm.best_iteration)


# In[46]:


y_pred = pd.DataFrame(y_train_pred)
y_pred.describe()


# In[47]:


thresholds = np.linspace(0, 1, 50)
ious = np.array([f1_score(Y_val, np.int32(y_val_pred > threshold)) for threshold in thresholds])

threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("f1")
plt.title("Threshold vs f1 ({}, {})".format(threshold_best, iou_best))
plt.legend()


# In[48]:


TH = threshold_best

for i in range(0, len(y_trn_pred)):
    if y_trn_pred[i] < TH:
        y_trn_pred[i] = 0
    else:
        y_trn_pred[i] = 1

for i in range(0, len(y_val_pred)):
    if y_val_pred[i] < TH:
        y_val_pred[i] = 0
    else:
        y_val_pred[i] = 1

for i in range(0, len(y_train_pred)):
    if y_train_pred[i] < TH:
        y_train_pred[i] = 0
    else:
        y_train_pred[i] = 1


# In[49]:


from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
print ('Training Set')
print ('Precision: ', precision_score(Y_trn, y_trn_pred))
print ('Recall: ', recall_score(Y_trn, y_trn_pred))
print ('F1 Score: ', f1_score(Y_trn, y_trn_pred))

print ('Validation Set')
print ('Precision: ', precision_score(Y_val, y_val_pred))
print ('Recall: ', recall_score(Y_val, y_val_pred))
print ('F1 Score: ', f1_score(Y_val, y_val_pred))


# In[50]:


print('Start predicting...')
y_pred_test = gbm.predict(test_df, num_iteration=gbm.best_iteration)
for i in range(0, len(y_pred_test)):
    if y_pred_test[i] < TH:
        y_pred_test[i] = 0
    else:
        y_pred_test[i] = 1


# In[51]:


sub.shape


# In[52]:


sub['is_promoted'] = y_pred_test
sub['is_promoted'].astype('int', inplace=True)
sub.head()


# In[53]:


sub.to_csv('0.526_lgbm.csv', index=False)


# In[54]:


sub.describe()
