import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency,boxcox,skew
from scipy.stats import stats
from sklearn.decomposition import PCA,FactorAnalysis,TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,learning_curve,validation_curve
from sklearn.preprocessing import MinMaxScaler,LabelEncoder ,StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score,log_loss,f1_score,roc_auc_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import RFECV
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from catboost import Pool

sample_submission = pd.read_csv('./dataset/sample_submission.csv')
train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')
train_test = train.append(test)

def get_cat_feature(data):
    return [x for x in data.columns if data[x].dtype == 'object' and x not in ['employee_id', 'is_promoted']]

def get_num_feature(data):
    return [x for x in data.columns if data[x].dtype != 'object' and x not in ['employee_id', 'is_promoted']]

def find_and_remove_skew(data,columns):
    skew_feat = data[columns].apply(lambda x : abs(skew(x)))
    skew_feat = skew_feat[skew_feat>0.5].index
    for col in skew_feat:
            data[col],_ = boxcox(data[col]+1)         
    return data

def encode_cat_feature(data,columns,encode_type=1):
    if(encode_type == 1):
        lbl = LabelEncoder()
        for col in columns:
            data[col] = lbl.fit_transform(data[col])
    else:
        for x in columns:
            data[x] = data[x].astype('category')
        
        ndf = pd.get_dummies(data[columns],prefix=columns,columns=columns)
        ndf_columns = ndf.columns.tolist()
        
        for x in ndf_columns:
            data[x] = ndf[x].astype('int8')
        
        data.drop(labels=columns,axis=1,inplace=True)
        
    return data

#TODO check with other combinations
def handle_missing_values(data):
    data.loc[data.education.isnull(),'education'] = 'others'
    data.loc[data['previous_year_rating'].isnull(),'previous_year_rating'] = 3
    data['previous_year_rating'] = data['previous_year_rating'].astype('int8')
    return data
        
def extract_features(data):
    data['reg_awrd_mean'] = data.groupby(['region'])['awards_won?'].transform(lambda x : np.mean(x)*100)
    data['reg_edu_awrd_mean'] = data.groupby(['region','education'])['awards_won?'].transform(lambda x : np.mean(x)*100)    
    data['reg_edu_recr_awrd_mean'] = data.groupby(['region','education','recruitment_channel'])['awards_won?'].transform(lambda x : np.mean(x)*100)        
    data['edu_recr_awrd_mean'] = data.groupby(['education','recruitment_channel'])['awards_won?'].transform(lambda x : np.mean(x)*100)        
    data['total_training_score'] = data.apply(lambda x: x.avg_training_score*x.no_of_trainings,axis=1)
    return data
    
def preprocessing(data):
    data = handle_missing_values(data) 
    data = extract_features(data)
    data = encode_cat_feature(data,['gender'])
    data = encode_cat_feature(data,['department', 'region', 'education', 'recruitment_channel'],2)
    return data


data  = train_test
data = preprocessing(data)
data.to_csv('useful_data.csv',index=None)

categorical_feature =  [x for x in data.columns if (data[x].nunique()==2 or x == 'region') and x not in ['employee_id', 'is_promoted']]
numerical_feature =  [x for x in data.columns if x not in categorical_feature and x not in ['employee_id', 'is_promoted']]

train_x = data[:len(train)]
train_y = train_x.is_promoted.values
dtrain_y = data[:len(train)]['is_promoted']
train_x.drop(['employee_id','is_promoted'],inplace=True,axis=1)
test_x = data[len(train):]
test_x.drop(['employee_id','is_promoted'],inplace=True,axis=1)



#tree based models

class SklearnHelper(object):
    def __init__(self,clf,seed=512,params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        
    def train(self,train_x,train_y):
        self.clf.fit(train_x,train_y)
        
    def predict(self,x):
        return self.clf.predict(x)
    
    def predict_prob(self,x):
        return self.clf.predict_proba(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return (self.clf.fit(x,y).feature_importances_)


class XgbWrapper(object):
    def __init__(self, seed=512, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 400)

    def train(self, xtra, ytra, xte, yte):
        dtrain = xgb.DMatrix(xtra, label=ytra)
        dvalid = xgb.DMatrix(xte, label=yte)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds,
            watchlist, early_stopping_rounds=10)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))
    
    def feature_importances(self,x,y):
        return (self.gbdt.feature_importances_)

class CatWrapper(object):
    def __init__(self,cat_index=None, seed=512,params=None):
        self.param = params
        self.cat_index = cat_index

    def train(self, xtra, ytra, xte, yte):
        dtrain = Pool(xtra,ytra,cat_features=self.cat_index)
        dvalid = Pool(xte,yte,cat_features=self.cat_index)
        self.catb = ctb.CatBoostClassifier(learning_rate=0.25,depth=6,iterations= 500,loss_function='Logloss') 
        self.catb.fit(dtrain,use_best_model=True, eval_set=dvalid)

    def predict(self, x):
        return self.catb.predict_proba(Pool(x,cat_features=self.cat_index))[:,1]

    def feature_importances(self,x,y):
        return (self.catb.feature_importances_)

class LgbWrapper(object):
    def __init__(self,cat_index=None, seed=512, params=None):
        self.param = params
        self.param['seed'] = seed
        self.cat_index = cat_index
        self.nrounds = params.pop('nrounds', 1000)

    def train(self, xtra, ytra, xte, yte):
        ytra = ytra.ravel()
        yte = yte.ravel()
        dtrain = lgb.Dataset(xtra, label=ytra,categorical_feature=self.cat_index)
        dvalid = lgb.Dataset(xte, label=yte)
        watchlist = [dvalid]
        self.gbdt = lgb.train(self.param, dtrain, self.nrounds,watchlist,early_stopping_rounds=60,verbose_eval=10)

    def predict(self, x):
        return self.gbdt.predict(x)
    
    def feature_importances(self,x,y):
        return (self.gbdt.feature_importances_)


def get_prediction(y,predict):
    predictions = []
    for x in predict:
        predictions.append(np.argmax(x))
    
    return f1_score(y,predictions,average='weighted')


def get_oof_prediction(clf,x_train,y_train,x_test,SEED,NFOLDS,clf_type=0):
    skf = StratifiedKFold(y_train,n_folds=NFOLDS,random_state=SEED)
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_ind,test_ind) in enumerate(skf):
        print ('fold numer = '+str(i))
        x_tr = x_train[train_ind]
        y_tr = y_train[train_ind]
        x_ts = x_train[test_ind]
        y_ts = y_train[test_ind]
                       
        if clf_type==0:
            clf.train(x_tr,y_tr)
            oof_train[test_ind] = clf.predict_prob(x_ts)[:,1]
            oof_test_skf[i,:] = clf.predict_prob(x_test)[:,1]
        else:    
            clf.train(x_tr,y_tr,x_ts,y_ts)
            oof_train[test_ind] = clf.predict(x_ts)
            oof_test_skf[i,:] = clf.predict(x_test)
        print("f1 score for fold : ======= ",roc_auc_score(y_ts,oof_train[test_ind]))
    oof_test = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

# Random Forest parameters
rf_params = {
    'n_jobs': 4,
    'n_estimators': 500,
    'criterion': 'gini',
#    'warm_start': True, 
    'max_features': 0.75,
    'min_samples_leaf':65,
    'max_depth': 14,
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': 4,
    'n_estimators':150,
    'criterion': 'entropy',
    'max_features': 0.75,
    'max_depth': 14,
    'min_samples_leaf': 50,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 50,
    'learning_rate' : 0.01
}

#Extreme Gradient Boosting parameters
xgb_params = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'learning_rate':0.15,   
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_leaf': 150,
    'colsample_bytree':0.75,
    'subsample':0.85,
    'reg_lambda':0.01,
    'reg_alpha':0.05,
    'eval_metric':'auc',
    'verbose': 0,
    'nthread':-1,    
}        

#lgbm parameters
lgb_params = {
    'boosting_type':'gbdt',
    'objective':'binary',
    'metric':'auc',
    'learning_rate':0.15,  
    'n_estimators': 1000,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'max_bin': 150,  # Number of bucketed bin for feature values
    'num_leaves': 40,
    'max_depth': 8,
    'reg_alpha': 0.05,
    'reg_lambda': 0.01,
    'min_child_samples':150,
    'seed':52,
    'verbose': 0,
    'nthread':4,
    'early_stopping_round': 100,
}     

from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin,tsec = divmod((datetime.now()-start_time).total_seconds(),60)
        print('\n Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))
        

SEED=512
ntrain = train_x.shape[0]
ntest = test_x.shape[0]
predictors = list(train_x.columns)
cat_index =[train_x[predictors].columns.get_loc(c) for c in categorical_feature ]
ntrain_x = train_x[predictors].values
ntest_x = test_x[predictors].values

start = timer(None)
rfc = SklearnHelper(RandomForestClassifier,seed=SEED,params=rf_params)
rf_oof_train,rf_oof_test = get_oof_prediction(rfc,ntrain_x,train_y,ntest_x,512,10,clf_type=0)
timer(start)

start = timer(None)
etc = SklearnHelper(ExtraTreesClassifier,seed=SEED,params=et_params)
et_oof_train,et_oof_test = get_oof_prediction(etc,ntrain_x,train_y,ntest_x,512,10,clf_type=0)
timer(start)

start = timer(None)
catb = CatWrapper(cat_index=cat_index,seed=SEED)
cat_oof_train,cat_oof_test = get_oof_prediction(catb,ntrain_x,train_y,ntest_x,512,10,clf_type=1)
timer(start)

start = timer(None)
xgbm = XgbWrapper(seed=SEED,params=xgb_params)
xgb_oof_train,xgb_oof_test = get_oof_prediction(xgbm,ntrain_x,train_y,ntest_x,512,10,clf_type=1)
timer(start)

start = timer(None)
lgbm = LgbWrapper(cat_index=cat_index,seed=SEED,params=lgb_params)
lgb_oof_train,lgb_oof_test = get_oof_prediction(lgbm,ntrain_x,train_y,ntest_x,512,10,clf_type=1)
timer(start)




#linear models
data = pd.read_csv('./dataset/useful_data1.csv')
data = find_and_remove_skew(data,numerical_feature)

train_x = data[:len(train)]
train_y = train_x.is_promoted.values
dtrain_y = data[:len(train)]['is_promoted']
train_x.drop(['employee_id','is_promoted'],inplace=True,axis=1)
test_x = data[len(train):]
test_x.drop(['employee_id','is_promoted'],inplace=True,axis=1)

ntrain = train_x.shape[0]
ntest = test_x.shape[0]
SEED = 512
NFOLDS = 10
skf = StratifiedKFold(n_splits=NFOLDS,random_state=SEED)

def get_knn_oof_prediction(x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_ind,test_ind) in enumerate(skf.split(x_train,y_train)):
        model = KNeighborsClassifier(n_neighbors=19)
        y_tr = y_train[train_ind]
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_train[train_ind])
        x_ts = scaler.transform(x_train[test_ind])
        x_test_s = scaler.transform(x_test)
        lda = LinearDiscriminantAnalysis()
        x_tr = lda.fit_transform(x_tr,y_tr)
        x_ts = lda.transform(x_ts)
        x_test_s = lda.transform(x_test_s)           
        model.fit(x_tr,y_tr)
        oof_train[test_ind] = model.predict(x_ts)
        oof_test_skf[i,:] = model.predict(x_test_s)
        print("Test score {} ".format(f1_score(y_train[test_ind],oof_train[test_ind])))
        
    oof_test = stats.mode(oof_test_skf,axis=0)[0]
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)


def get_sgd_oof_prediction(SEED,x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_ind,test_ind) in enumerate(skf.split(x_train,y_train)):
        model = SGDClassifier(max_iter=100,random_state=SEED,loss="squared_hinge",alpha=0.009,penalty='l1')
        y_tr = y_train[train_ind]
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_train[train_ind])
        x_ts = scaler.transform(x_train[test_ind])
        x_test_s = scaler.transform(x_test)
        model.fit(x_tr,y_tr)
        oof_train[test_ind] = model.predict(x_ts)
        oof_test_skf[i,:] = model.predict(x_test_s)
        print("Test score {} ".format(f1_score(y_train[test_ind],oof_train[test_ind])))
        
    oof_test = stats.mode(oof_test_skf,axis=0)[0]
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

def get_log_oof_prediction(SEED,x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_ind,test_ind) in enumerate(skf.split(x_train,y_train)):
        model = LogisticRegression(random_state=SEED,C=0.8252042855888113,penalty='l1',verbose=2)
        y_tr = y_train[train_ind]
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_train[train_ind])
        x_ts = scaler.transform(x_train[test_ind])
        x_test_s = scaler.transform(x_test)
        model.fit(x_tr,y_tr)
        oof_train[test_ind] = model.predict(x_ts)
        oof_test_skf[i,:] = model.predict(x_test_s)
        print("Test score {} ".format(f1_score(y_train[test_ind],oof_train[test_ind])))
        
    oof_test = stats.mode(oof_test_skf,axis=0)[0]
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

def get_qda_oof_prediction(x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_ind,test_ind) in enumerate(skf.split(x_train,y_train)):
        model = QuadraticDiscriminantAnalysis()
        y_tr = y_train[train_ind]
        x_tr = x_train[train_ind]
        x_ts = x_train[test_ind]
        model.fit(x_tr,y_tr)
        oof_train[test_ind] = model.predict(x_ts)
        oof_test_skf[i,:] = model.predict(x_test)
        print("Test score {} ".format(f1_score(y_train[test_ind],oof_train[test_ind])))        
    oof_test = stats.mode(oof_test_skf,axis=0)[0]
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)


predictors = list([x for x in train_x.columns if x not in ['employee_id', 'is_promoted']])
ntrain_x = train_x[predictors].values
ntest_x = test_x[predictors].values

knn_oof_train,knn_oof_test = get_knn_oof_prediction(ntrain_x,train_y,ntest_x)

sgd_oof_train,sgd_oof_test = get_sgd_oof_prediction(SEED,ntrain_x,train_y,ntest_x)

log_oof_train,log_oof_test = get_log_oof_prediction(SEED,ntrain_x,train_y,ntest_x)

qda_oof_train,qda_oof_test = get_qda_oof_prediction(ntrain_x,train_y,ntest_x)


base_predictions_train = pd.DataFrame({
    'Catboost': cat_oof_train.ravel(),
    'LightGbM': lgb_oof_train.ravel(),
    'XGBoost': xgb_oof_train.ravel(),
    'KNN':knn_oof_train.ravel(),
    'SGD':sgd_oof_train.ravel(),
    'QDA':qda_oof_train.ravel(),
    'logistic':log_oof_train.ravel()
    })

base_predictions_test = pd.DataFrame({
    'Catboost': cat_oof_test.ravel(),
    'LightGbM': lgb_oof_test.ravel(),
    'XGBoost': xgb_oof_test.ravel(),
    'KNN':knn_oof_test.ravel(),
    'SGD':sgd_oof_test.ravel(),
    'QDA':qda_oof_test.ravel(),
    'logistic':log_oof_test.ravel()
    })

# level 2
#lr = LogisticRegression(C=0.8)
#lr.fit(base_predictions_train,train_y)
#predict = lr.predict_proba(base_predictions_test)[:,1]
#submission = pd.DataFrame({'employee_id':test['employee_id'],'is_promoted':predict.ravel()})
#submission['is_promoted'] = submission['is_promoted'].apply(lambda x: 1 if x>0.5 else 0)
#submission.to_csv('logistic_regression.csv',index=None)
#

SEED=512
ntrain = base_predictions_train.shape[0]
ntest = base_predictions_test.shape[0]
predictors = list(base_predictions_train.columns)
cat_index =[base_predictions_train[predictors].columns.get_loc(c) for c in ['KNN','SGD','QDA','logistic'] ]
ntrain_x = base_predictions_train[predictors].values
ntest_x = base_predictions_test[predictors].values

start = timer(None)
catb = CatWrapper(cat_index=cat_index,seed=SEED)
cat_oof_train,cat_oof_test = get_oof_prediction(catb,ntrain_x,train_y,ntest_x,512,10,clf_type=1)
timer(start)

start = timer(None)
xgbm = XgbWrapper(seed=SEED,params=xgb_params)
xgb_oof_train,xgb_oof_test = get_oof_prediction(xgbm,ntrain_x,train_y,ntest_x,512,10,clf_type=1)
timer(start)

start = timer(None)
lgbm = LgbWrapper(cat_index=cat_index,seed=SEED,params=lgb_params)
lgb_oof_train,lgb_oof_test = get_oof_prediction(lgbm,ntrain_x,train_y,ntest_x,512,10,clf_type=1)
timer(start)


base_predictions_trainlevel1 = pd.DataFrame({
    'Catboost': cat_oof_traincv.ravel(),
    'LightGbM': lgb_oof_traincv.ravel(),
    'XGBoost': xgb_oof_traincv.ravel(),
    })

base_predictions_testlevel1 = pd.DataFrame({
    'Catboost': cat_oof_testcv.ravel(),
    'LightGbM': lgb_oof_testcv.ravel(),
    'XGBoost': xgb_oof_testcv.ravel(),
    })
    
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(base_predictions_trainlevel1,train_y)
predict = lr.predict_proba(base_predictions_testlevel1)[:,1]
submission = pd.DataFrame({'employee_id':test['employee_id'],'is_promoted':predict.ravel()})
submission['is_promoted'] = submission['is_promoted'].apply(lambda x: 1 if x>0.5 else 0)
submission.to_csv('logistic_regression_lgb_cat_xgb_level1.csv',index=None)




