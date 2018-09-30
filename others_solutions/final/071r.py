# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

def f_1(preds, train_data):
    labels = train_data.get_label()
    return 'error', f1_score(labels,np.round(preds+0.20482185831793379)), True

# load or create your dataset
print('Load data...')
df_train = pd.read_csv("dados/train.csv")
print(df_train.head())
df_test = pd.read_csv('dados/test.csv')
print(df_test.head())

y_train = df_train['is_promoted'].values

print(df_train.dtypes)
print(df_train["department"].value_counts())
print(df_train["region"].value_counts())
print(df_train["education"].value_counts()) #ok
print(df_train["gender"].value_counts()) #ok
print(df_train["recruitment_channel"].value_counts()) #ok


cleanup_nums = {"gender":     {"m": 1, "f": 0},
                "recruitment_channel": {"other": 0, "sourcing": 0.5, "referred": 1 },
                "education": {"Below Secondary": 0, "Bachelor's": 0.5, "Master's & above": 1}
                }

df_train.replace(cleanup_nums, inplace=True)
df_test.replace(cleanup_nums, inplace=True)


#df_train = pd.get_dummies(df_train, columns=["recruitment_channel"], prefix=["rec"])
#df_test = pd.get_dummies(df_test, columns=["recruitment_channel"], prefix=["rec"])

#--------------removed region-------------------------------
#df_train["region_mod"] = df_train["region"].astype('category').cat.codes
#df_test["region_mod"] = df_test["region"].astype('category').cat.codes

df_train = pd.get_dummies(df_train, columns=["department"], prefix=["dep"])
df_test = pd.get_dummies(df_test, columns=["department"], prefix=["dep"])

print(df_train.head())

X_train2 = df_train.drop(['region','is_promoted','employee_id'], axis=1).values
X_test2 = df_test.drop(['region','employee_id'], axis=1).values

print(df_train.head())
print(df_test.head())

#X_test = df_test.drop(-1, axis=1).values

# create dataset for lightgbm

#lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

#---------------------- xentropy
# 0.528026315285
#> BESTSOLUTION: [535.0150005238936, 30.72836434271078, 0.02658081962125346, 0.5056964347836906, 0.9801601137516736, 27.229693508611938, 16.74614782175742, 9.171316414725663, -0.22141196223925164]

#-------------------train with  is_unbalanced
# F1 0.527685997769
# [266.3228855629336, 26.054721397916115, 0.05741583423216279, 0.7504840072191963, 0.921411252360875, 30, 10.073567903501592, 0.49463853383320544, 0.2739374011944594]


#-------------------train with no is_unbalanced
#F1 - 0.528957277863
#BEST SOLUTION: [442.7231570628395, 15.0, 0.05168752585799009, 0.7998240215022054, 0.71736958445406, 15.307364888355693, 10.401569226196223, 2.3515002255557746, -0.2034666409356397]

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'xentropy',#'binary',
    'num_leaves': 38,
    'learning_rate': 0.05774820675369787,
    'feature_fraction': 0.6325216833967215,
    'bagging_fraction': 0.8752006932474125,
    'bagging_freq': 1,
    'max_depth' : 11,
    'min_data_in_leaf':13,
    'lambda_l2' : 18,
    'is_unbalance' : True,
    'verbose': 0
}

print('Start training...')
# train



kf = KFold(n_splits=5,random_state=42)


y_pred = np.zeros(X_test2.shape[0])
y_predf = np.zeros(X_test2.shape[0])

for train_index, test_index in kf.split(X_train2):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_train2[train_index], X_train2[test_index]
    y_trainA, y_testA = y_train[train_index], y_train[test_index]

    lgb_train = lgb.Dataset(X_train, y_trainA)
    gbm = lgb.train(params, lgb_train, num_boost_round=723)

    lgb_test = lgb.Dataset(X_test, y_testA)
       #gbm = lgb.train(params, lgb_train, num_boost_round=int(x[0])  ) #x[0]

    gbm = lgb.train(params, lgb_train, num_boost_round=656,valid_sets = [lgb_test],early_stopping_rounds=30,feval = f_1, verbose_eval= 0)     #x[0]
    y_pred += np.round(gbm.predict(X_test2,num_iteration=gbm.best_iteration)+0.20482185831793379)
    print(y_pred[1:100])






a = df_test['employee_id']
b = pd.DataFrame(np.round(y_pred/5) )
#print(a)
#print(b)
df = pd.concat( [a ,b], axis=1)

#print(df)
df.columns = ['employee_id', 'is_promoted']
#df['employee_id'].astype(int)
df['is_promoted'].astype(int)

df.to_dense().to_csv('sub.csv', index = False, sep=',')
#0.923824952281
#BEST SOLUTION: [763.1145472317621, 30.218236711498427, 0.04253683748516579, 0.05, 0.9240741175417527, 9.139574165769336, 21.308268825762802, 1]

