import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def less_important_features(csvdata,testdata,boost_model):
    rem_cols = boost_model.feature_importances_ > 0.000001
    colnames = list(csvdata)
    for k in range(len(rem_cols)) :
       print boost_model.feature_importances_[k],colnames[k]
       if rem_cols[k] == False or k == 0:
          csvdata.drop(columns=colnames[k],inplace=True)
          testdata.drop(columns=colnames[k],inplace=True)
          print "removing " + colnames[k]
    return csvdata,testdata

def find_feature_importance(csvdata,test) :
    print np.shape(csvdata)
    print np.shape(testdata)
    xgbmodel = xgb.XGBClassifier(tree_method= 'exact',n_jobs=64,max_depth=3, n_estimators=50, learning_rate=0.2,random_state=1234).fit(csvdata,test)
    return xgbmodel

def cross_validate(csvdata) :
    kf=KFold(n_splits=2, random_state=123, shuffle=True)
    for train_index, test_index in kf.split(csvdata):
        X_train, X_test = csvdata.iloc[train_index], csvdata.iloc[test_index]
        y_train, y_test = test.iloc[train_index], test.iloc[test_index]
        xgbmodel = xgb.XGBRegressor(tree_method= 'exact',n_jobs=64,max_depth=6, n_estimators=50, learning_rate=0.3,random_state=14).fit(X_train,y_train)
        #y_pred = xgbmodel.predict(X_test)[:,1]
        y_pred = xgbmodel.predict(X_test) 
     	print(roc_auc_score(y_test,y_pred))    
        y_pred[y_pred < 0.25]  = 0
        y_pred[y_pred != 0 ]  = 1
        print  f1_score(y_test, y_pred, average='binary')
    #Print Average Cross Validation Score
    return xgbmodel


csvdata = pd.read_csv('train_LZdllcl.csv',engine='python')
testdata = pd.read_csv('test_2umaH9m.csv',engine='python')

csvdata = csvdata.append(testdata).reset_index()


csvdata , cat_cols = one_hot_encoder(csvdata, True)



# CONVERT ALL COLUMNS TO NUMERIC

colnames = list(csvdata)
for k in colnames :
  csvdata[k] = pd.to_numeric(csvdata[k],errors='coerce')

df  = csvdata[csvdata['is_promoted'].notnull()]
dt = csvdata[csvdata['is_promoted'].isnull()]

print "Finding Importance Features "

df = df.sample(frac=15,replace=True).reset_index(drop=True)

test = df['is_promoted']
df = df.drop(['is_promoted'],axis=1)
dt = dt.drop(['is_promoted'],axis=1)

df = df.drop(['employee_id'],axis=1)
mytestid = dt['employee_id']
dt = dt.drop(['employee_id'],axis=1)

# Find Feature Importance
print np.shape(csvdata)
xgb_feature_imp = find_feature_importance(df,test)

# Drop Features of less Importance

print "Drop less important Features"

csvdata,testdata = less_important_features(df,dt,xgb_feature_imp)
print np.shape(df)
print np.shape(dt)

# FILL MISSING VALUES WITH 0 - Do Feature Engg for Accuracy improvements

csvdata = csvdata.fillna(0)
testdata = testdata.fillna(0)

# FIND hyper-paramaters that results in higher average CROSS VALIDATION score
print "Crossvalidating "

xgbmodel = cross_validate(df)

# PREDICT ON TEST DATA
print "Inferencing "

#y_pred = xgbmodel.predict_proba(dt)[:,1]
y_pred = xgbmodel.predict(dt)
#print y_pred

# CREATE DATAFRAME FOR SUBMISSION

submission_df = pd.DataFrame(data=mytestid,columns=['employee_id'])
y_pred[y_pred < 0.22]  = 0
y_pred[y_pred != 0 ]  = 1

submission_df['is_promoted'] = y_pred

# WRITE RESULTS TO CSV FILE

submission_df.to_csv("results.csv", sep=',',index=False)
