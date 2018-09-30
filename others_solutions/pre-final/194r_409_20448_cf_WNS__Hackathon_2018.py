
# coding: utf-8

# ### Import Libraries and Set Options

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from itertools import product
from sklearn import model_selection
import lightgbm as lgb
from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

# Set all options
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-notebook')
plt.rcParams["figure.figsize"] = (20, 3)
pd.options.display.float_format = '{:20,.4f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set(context="paper", font="monospace")


# ### User Defined Functions

# In[2]:


def convert_categorical_to_dummies(d_convert):
    

    
    df = d_convert.copy()
    list_to_drop = []
    for col in df.columns:
        if df[col].dtype == 'object':
            list_to_drop.append(col)
            df = pd.concat([df,pd.get_dummies(df[col],prefix=col,prefix_sep='_', drop_first=False)], axis=1)    
    df = df.drop(list_to_drop,axis=1)
    return df


def chk_dtype_nunique(df):


    
    df.T.apply(lambda x: x.nunique(), axis=1)
    display(pd.DataFrame({'dtype':df.dtypes,'nunique':df.T.apply(lambda x: x.nunique(), axis=1)}))

    
def chk_missing_data(df):
    

    
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print("Missing Information :")
    display(missing_data)
    
    
def object_count_plot(df):
    

    
    for var in df.columns:
        if df[var].dtype == 'object':
            print(df[var].value_counts())
            plt.figure(figsize=(12,5))
            g = sns.countplot(x=var,data=df)
            g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
            plt.tight_layout()
            plt.show()
            
def numeric_distribution_plot(df):
    

    for col in df.columns:
        if df[col].dtype != 'object':
            print(df[col].describe())
            plt.figure(figsize=(12,5))
            plt.title("Distribution of "+col)
            ax = sns.distplot(df[col].dropna())
            plt.tight_layout()
            plt.show()
            
    
def score_on_test_set(model, file_name, out_name):
    
    test_data = pd.read_csv(file_name)
    
    # Treating the missing values of education as a separate category
    test_data['education'] = test_data['education'].replace(np.NaN, 'NA')
    
    # Treating the missing values of education as a separate category
    test_data['previous_year_rating'] = test_data['previous_year_rating'].fillna(0)
    
    # Creating dummy variables for all the categorical columns, droping that column
    master_test_data = convert_categorical_to_dummies(test_data)
    
    # Removing the id attributes
    df_test_data = master_test_data.drop(['employee_id'],axis=1)
    if out_name == "submission_lightgbm.csv":
        y_pred = model.predict_proba(df_test_data.values, num_iteration=model.best_iteration_)
    else:        
        y_pred = model.predict_proba(df_test_data.values)
    submission_df = pd.DataFrame({'employee_id':master_test_data['employee_id'],'is_promoted':y_pred[:,1]})
    submission_df.to_csv(out_name, index=False)
    
    score = model.predict_proba(df_test_data.values)
    return test_data,score
 
    
def score_on_test_set_d(model, file_name, out_name):
    
    test_data = pd.read_csv(file_name)
    
    # Treating the missing values of education as a separate category
    test_data['education'] = test_data['education'].replace(np.NaN, 'NA')
    
    # Treating the missing values of education as a separate category
    test_data['previous_year_rating'] = test_data['previous_year_rating'].fillna(0)
    
    # Creating dummy variables for all the categorical columns, droping that column
    master_test_data = convert_categorical_to_dummies(test_data)
    
    # Removing the id attributes
    df_test_data = master_test_data.drop(['employee_id'],axis=1)
    if out_name == "submission_lightgbm.csv":
        y_pred = model.predict(df_test_data.values, num_iteration=model.best_iteration_)
    else:        
        y_pred = model.predict(df_test_data.values)
    submission_df = pd.DataFrame({'employee_id':master_test_data['employee_id'],'is_promoted':y_pred})
    submission_df.to_csv(out_name, index=False)
    
    score = model.predict_proba(df_test_data.values)
    return test_data,score
    
    
def std_scalor(X):
    

    
    std = StandardScaler()
    std.fit(X)
    X_new = std.transform(X)
    return X_new    

def min_max_scalor(X):
    

    mms = MinMaxScaler()
    mms.fit(X)
    X_new = mms.transform(X)    
    return X_new  
   


# In[3]:


def create_interaction(df,var1,var2):
    name = var1 + "_" + var2
    if var2 == 'education':
        zz = (df[var2] == "Master's & above")
        df[name] = pd.Series(df[var1] * zz, name=name)
    else:
        df[name] = pd.Series(df[var1] * df[var2], name=name)
    return df

def create_specific_interaction(df,var1,var2):
    name = var1 + "_" + var2
    yy =  (df[var2] == 5.0)
    df[name] = pd.Series(df[var1] * yy, name=name)
    return df

def create_overall_interaction(df,var1,var2,var3):
    name = var1 + "_" + var2 + "_" + var3
    yy =  (df[var2] == 5.0)
    zz = (df[var3] == "Master's & above")
    df[name] = pd.Series(df[var1] * yy * zz, name=name)
    return df


# ### Load data

# In[4]:


data = pd.read_csv("train_wns_data.csv")
print(data.shape)
data.head()


# In[5]:


plt.figure(figsize=(6,3))
sns.countplot(x='is_promoted',data=data)
plt.show()

# Checking the event rate : event is when claim is made
data['is_promoted'].value_counts()


# In[6]:


# Checking the attribute names
pd.DataFrame(data.columns) 


# In[7]:


# checking missing data
chk_missing_data(data)


# In[8]:


# Treating the missing values of education as a separate category
data['education'] = data['education'].replace(np.NaN, 'NA')

# Treating the missing values of education as a separate category
data['previous_year_rating'] = data['previous_year_rating'].fillna(0)


# In[9]:


# checking missing data again
chk_missing_data(data)


# In[10]:


# Checking number of unique values in each column, just to confirm if there are multiple values in it.
chk_dtype_nunique(data)


# In[11]:


object_count_plot(data)


# In[12]:


# Checking the distribution of numeric features
numeric_distribution_plot(data)


# In[13]:


# Creating dummy variables for all the categorical columns, droping that column
master_data = convert_categorical_to_dummies(data)
chk_dtype_nunique(master_data)
print("Total shape of master data :",master_data.shape)


# In[14]:


# dropping the target from dataset
labels = np.array(master_data['is_promoted'].tolist())

# Removing the id attributes
df_data = master_data.drop(['is_promoted','employee_id'],axis=1)
print("Shape of Data:",df_data.shape)
df = df_data.values


# ### Model  - LightGBM

# In[15]:


gbm_model = lgb.LGBMClassifier(objective='binary')
print(gbm_model)
f1_scores = cross_val_score(gbm_model, df, labels, cv=5, scoring='f1',n_jobs=3)
print(f1_scores," Mean = ",np.mean(f1_scores))
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.10, stratify=labels)
gbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=150)
test_data,score_lgbm_tuned = score_on_test_set(gbm_model,"test_wns_data.csv","submission_lightgbm.csv")    

