# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 14:07:15 2018

@author: Himansh
"""

from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
seed = 7
num_trees = 70
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(learning_rate=0.12,n_estimators=num_trees, random_state=seed,min_samples_split=700,min_samples_leaf=45,max_depth=14,max_features='log2',subsample=0.80)


import pandas as pd
import numpy as np
from sklearn.utils import shuffle

dataset = pd.read_csv('F:\kaggle\wns\OutlierTrain.csv')
X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 9].values

X, y = shuffle(X, y, random_state=123)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search


#param_test1 = {'n_estimators':np.arange(20,81,10)}
#gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
#param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch1.fit(X,y)
#gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#



#param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
#gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10), 
#param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch2.fit(X,y)
#gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
#
#
#
#param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
#gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,max_features='sqrt', subsample=0.8, random_state=10), 
#param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch3.fit(X,y)
#gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#
#
#
#param_test4 = {'max_features':range(7,20,2)}
#gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10),
#param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch4.fit(X,y)
#gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#
#
#
#
#param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
#gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10,max_features=7),
#param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch5.fit(X,y)
#gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_






#mod = model.fit(X,y)














dataset2 = pd.read_csv('F:\kaggle\wns\wns\\X_test.csv')
X_test_new = dataset2.iloc[:, 2:11].values


mod = model.fit(X,y)
res = mod.predict(X_test_new)
print(np.unique(res,return_counts=True))

dfTest = pd.read_csv("featuresUpdatedTesting.csv")
X_test_emp = dfTest.iloc[:, 0].values
count=0
arr=[]
emp=[]
for i in res:
    emp.append(X_test_emp[count])
    count+=1
    if(i > 0.5):
        arr.append(1)
    else:
        arr.append(0)


#dfTest.insert(loc=13,column='employee_id',value=emp)        
#dfTest.insert(loc=14,column='is_promoted',value=arr)        

newDF = pd.DataFrame()
newDF.insert(loc=0,column='employee_id',value=emp)        
newDF.insert(loc=1,column='is_promoted',value=arr)        

newDF.to_csv('submissions/Submission_svm2.csv')