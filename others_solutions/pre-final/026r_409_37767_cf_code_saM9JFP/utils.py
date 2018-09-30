import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, make_scorer, roc_auc_score, precision_score, recall_score

from scipy.stats import ranksums
from scipy.special import erfinv
from lightgbm import LGBMClassifier

import logging
import time
import gc
import warnings
import yaml
import os

import logging.config

def setup_logging(default_path='logging.yaml',
                  default_level=logging.INFO):
    '''
    Setup logging configuration
    :param default_path: path of the logging configuration yaml file
    :param default_level: default logging level
    :return: No return
    '''
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    return logging

CATEGORICAL_COLUMNS = ['department', 'region', 'education', 'gender', 'recruitment_channel',
                      'no_of_trainings', 'previous_year_rating', 'KPIs_met >80%', 'awards_won?',
                      'length_of_service', 'age']
NUMERICAL_COLUMNS = ['avg_training_score']

def one_hot_encoder(data, categorical_columns = [], nan_as_category = True):
    original_columns = list(data.columns)
    if not categorical_columns:
        categorical_columns = [col for col in data.columns \
                               if not pd.api.types.is_numeric_dtype(data[col].dtype)]
    for c in categorical_columns:
        if nan_as_category:
            data[c].fillna('NaN', inplace = True)
        values = list(data[c].unique())
        for v in values:
            data[str(c) + '_' + str(v)] = (data[c] == v).astype(np.uint8)
    return data

def label_encoder(data, categorical_columns = [], nan_as_category = True):
    original_columns = list(data.columns)
    if not categorical_columns:
        categorical_columns = [col for col in data.columns \
                               if not pd.api.types.is_numeric_dtype(data[col].dtype)]
    for c in categorical_columns:
        if nan_as_category:
            data[c].fillna('NaN', inplace = True)
            value = list(data[c].unique())
            lb_dict = {v:k for k,v in enumerate(value)}
            data[str(c)+'_encoded'] = data[c].map(lb_dict)
    return data

def train_test(file_path = '../input/', nan_as_category = True, lb_encode = True):
    # Read data and merge
    df_train = pd.read_csv(file_path + 'train_LZdllcl.csv')
    df_test = pd.read_csv(file_path + 'test_2umaH9m.csv')
    df = pd.concat([df_train, df_test], axis = 0, ignore_index = True)
    del df_train, df_test
    gc.collect()
    
    # Remove some rows with values not present in test set
    df.drop(df[df['no_of_trainings'] == 10].index, inplace = True)
    df.drop(df[df['length_of_service'] == 37].index, inplace = True)
    
    # Creating two way combinations
    df['region_channel'] = df['region'].astype(str) + '_'+df['recruitment_channel'].astype(str)
    df['education_region']= df['region'].astype(str)+'_'+df['education'].astype(str)
    df['KPI_award'] = df['KPIs_met >80%'].astype(str)+'_'+df['awards_won?'].astype(str)
    df['score_service_ratio'] = df['avg_training_score']/df['length_of_service']

    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['gender', 'awards_won?', 'KPIs_met >80%']:
        df[bin_feature], _ = pd.factorize(df[bin_feature])
        
    cat_columns = CATEGORICAL_COLUMNS + ['KPI_award', 'region_channel', 'education_region']
    
    print(cat_columns)
    # Categorical features with One-Hot encode
    if lb_encode:
        df = label_encoder(df, categorical_columns=cat_columns, nan_as_category = nan_as_category)
    else:
        df = one_hot_encoder(df, categorical_columns=cat_columns, nan_as_category = nan_as_category)
    
    # Some new features
    agg = {'avg_training_score': ['min', 'mean', 'median', 'max', 'std']}
    for col in cat_columns:
        scaler = MinMaxScaler()
        df[str(col)+'_counts'] = df[col].map(df[col].value_counts())
        df[str(col)+'_counts'] = scaler.fit_transform(df[str(col)+'_counts'].values.reshape(-1, 1)).ravel()
        temp = df.groupby(col).agg(agg)
        temp.columns = ['_'.join(c) for c in temp.columns]
        df = df.join(temp, on = col, how = 'left', rsuffix = f'_{col}')
    
    df.drop(cat_columns, axis = 1, inplace = True)
    
    return df

def aggregate(file_path = '../input/', nan_as_category = True, lb_encode = True):
    warnings.simplefilter(action = 'ignore')
    
    print('-' * 20)
    print('train & test (', time.ctime(), ')')
    print('-' * 20)
    df = train_test(file_path, nan_as_category = nan_as_category, lb_encode = lb_encode)
    print('     DF shape:', df.shape)
    
    return df

def corr_feature_with_target(feature, target):
    c0 = feature[target == 0].dropna()
    c1 = feature[target == 1].dropna()
        
    if set(feature.unique()) == set([0, 1]):
        diff = abs(c0.mean(axis = 0) - c1.mean(axis = 0))
    else:
        diff = abs(c0.median(axis = 0) - c1.median(axis = 0))
        
    p = ranksums(c0, c1)[1] if ((len(c0) >= 20) & (len(c1) >= 20)) else 2
        
    return [diff, p]

def woe(X, y):
    tmp = pd.DataFrame()
    tmp["variable"] = X
    tmp["target"] = y
    var_counts = tmp.groupby("variable")["target"].count()
    var_events = tmp.groupby("variable")["target"].sum()
    var_nonevents = var_counts - var_events
    tmp["var_counts"] = tmp.variable.map(var_counts)
    tmp["var_events"] = tmp.variable.map(var_events)
    tmp["var_nonevents"] = tmp.variable.map(var_nonevents)
    events = sum(tmp["target"] == 1)
    nonevents = sum(tmp["target"] == 0)
    tmp["woe"] = np.log(((tmp["var_nonevents"])/nonevents)/((tmp["var_events"])/events))
    tmp["woe"] = tmp["woe"].replace(np.inf, 0).replace(-np.inf, 0)
    tmp["iv"] = (tmp["var_nonevents"]/nonevents - tmp["var_events"]/events) * tmp["woe"]
    iv = tmp.groupby("variable")["iv"].last().sum()
    return tmp["woe"], tmp["iv"], iv

def clean_data(data):
    warnings.simplefilter(action = 'ignore')
    
    # Removing empty features
    nun = data.nunique()
    empty = list(nun[nun <= 1].index)
    
    data.drop(empty, axis = 1, inplace = True)
    print('After removing empty features there are {0:d} features'.format(data.shape[1]))
    
    # Removing features with the same distribution on 0 and 1 classes
    corr = pd.DataFrame(index = ['diff', 'p'])
    ind = data[data['is_promoted'].notnull()].index
    
    for c in data.columns.drop('is_promoted'):
        corr[c] = corr_feature_with_target(data.loc[ind, c], data.loc[ind, 'is_promoted'])

    corr = corr.T
    corr['diff_norm'] = abs(corr['diff'] / data.mean(axis = 0))
    
    to_del_1 = corr[((corr['diff'] == 0) & (corr['p'] > .05))].index
    to_del_2 = corr[((corr['diff_norm'] < .5) & (corr['p'] > .05))].drop(to_del_1).index
    to_del = list(to_del_1) + list(to_del_2)
    if 'employee_id' in to_del:
        to_del.remove('employee_id')
        
    data.drop(to_del, axis = 1, inplace = True)
    print('After removing features with the same distribution on 0 and 1 classes there are {0:d} features'.format(data.shape[1]))
    
    # Removing features with not the same distribution on train and test datasets
    corr_test = pd.DataFrame(index = ['diff', 'p'])
    target = data['is_promoted'].notnull().astype(int)
    
    for c in data.columns.drop('is_promoted'):
        corr_test[c] = corr_feature_with_target(data[c], target)

    corr_test = corr_test.T
    corr_test['diff_norm'] = abs(corr_test['diff'] / data.mean(axis = 0))
    
    bad_features = corr_test[((corr_test['p'] < .05) & (corr_test['diff_norm'] > 1))].index
    bad_features = corr.loc[bad_features][corr['diff_norm'] == 0].index
    
    data.drop(bad_features, axis = 1, inplace = True)
    print('After removing features with not the same distribution on train and test datasets there are {0:d} features'.format(data.shape[1]))
    
    del corr, corr_test
    gc.collect()
    
    ## feature selection based on Information value
    bad_features = []
    for col in data.columns:
        if col not in ['is_promoted', 'employee_id']:
            _, _, iv = woe(data[col], data['is_promoted'])
            if 0.02 < iv > 0.8:
                bad_features.append(col)
            else:
                continue
    data.drop(bad_features, axis = 1, inplace = True)
    print('After removing features which are having information value less than 0.02 and greater than 0.8s {0:d} features'.format(data.shape[1]))
    return data

def cv_scores(estimator, df, num_folds, params, stratified = False, verbose = 100, random_state = 11,
              save_train_prediction = True, train_prediction_file_name = 'train_prediction.csv',
              save_test_prediction = True, test_prediction_file_name = 'test_prediction.csv', iteration = 0,
              logger=None, lgbm_clf = True):
    warnings.simplefilter('ignore')
    
    if hasattr(estimator.get_params(), 'random_seed'):
        params['random_seed'] = np.random.choice(1000, 1)[0]
        
    if hasattr(estimator.get_params(), 'random_state'):
        params['random_state'] = np.random.choice(1000, 1)[0]

    clf = estimator.set_params(**params)

    # Divide in training/validation and test data
    train_df = df[df['is_promoted'].notnull()]
    test_df = df[df['is_promoted'].isnull()]
    print("Starting Classifier. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = random_state)
    else:
        folds = KFold(n_splits = num_folds, shuffle = True, random_state = random_state)
        
    # Create arrays and dataframes to store results
    train_pred = np.zeros(train_df.shape[0])
    train_pred_proba = np.zeros(train_df.shape[0])

    test_pred = np.zeros(train_df.shape[0])
    test_pred_proba = np.zeros(train_df.shape[0])
    
    prediction_proba = np.zeros(test_df.shape[0], dtype= np.float16)
    
    feats = [f for f in train_df.columns if f not in ['index', 'is_promoted', 'employee_id']]
    
    df_feature_importance = pd.DataFrame(index = feats)
    folds_score = []
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['is_promoted'])):
        print('Fold', n_fold, 'started at', time.ctime())
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['is_promoted'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['is_promoted'].iloc[valid_idx]
        cat_feats_index = [i for i,c in enumerate(train_x.columns) if '_encoded' in c]
        
        if lgbm_clf:
            clf.fit(train_x, train_y, 
                eval_set = [(train_x, train_y), (valid_x, valid_y)], 
                categorical_feature =cat_feats_index,
                verbose = verbose, early_stopping_rounds = 200)
        else:
            clf.fit(train_x, train_y, 
                eval_set = [(train_x, train_y), (valid_x, valid_y)], 
                cat_features =cat_feats_index,
                verbose = verbose, early_stopping_rounds = 200)
            
        if lgbm_clf:
            train_pred[train_idx] = clf.predict(train_x, num_iteration = clf.best_iteration_)
            train_pred_proba[train_idx] = clf.predict_proba(train_x, num_iteration = clf.best_iteration_)[:, 1]
            test_pred[valid_idx] = clf.predict(valid_x, num_iteration = clf.best_iteration_)
            test_pred_proba[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)[:, 1]
        else:
            train_pred[train_idx] = clf.predict(train_x)
            train_pred_proba[train_idx] = clf.predict_proba(train_x)[:, 1]
            test_pred[valid_idx] = clf.predict(valid_x)
            test_pred_proba[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        
        prediction_proba += \
                clf.predict_proba(test_df[feats])[:, 1] / folds.n_splits

        df_feature_importance[n_fold] = pd.Series(clf.feature_importances_, index = feats)
        
        print('Fold %2d AUC : %.6f' % (n_fold, roc_auc_score(valid_y, test_pred_proba[valid_idx])))
        print('Fold %2d f1_score : %.6f' % (n_fold, f1_score(valid_y, test_pred[valid_idx])))
        folds_score.append(roc_auc_score(valid_y, test_pred_proba[valid_idx]))
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    roc_auc_train = roc_auc_score(train_df['is_promoted'], train_pred_proba)
    precision_train = precision_score(train_df['is_promoted'], train_pred, average = None)
    recall_train = recall_score(train_df['is_promoted'], train_pred, average = None)
    f1_score_train = f1_score(train_df['is_promoted'], train_pred, average = None)
    
    roc_auc_test = roc_auc_score(train_df['is_promoted'], test_pred_proba)
    precision_test = precision_score(train_df['is_promoted'], test_pred, average = None)
    recall_test = recall_score(train_df['is_promoted'], test_pred, average = None)
    f1_score_test = f1_score(train_df['is_promoted'], test_pred, average = None)
    
    print('Full F1_score %.6f' % f1_score(train_df['is_promoted'], test_pred))
    
    if logger:
        logger.info('*'*50)
        logger.info(params)
        logger.info(folds_score)
        logger.info(iteration)
        logger.info('Full F1_score %.6f' % f1_score(train_df['is_promoted'], test_pred))
        logger.info('*'*50)
    
    df_feature_importance.fillna(0, inplace = True)
    df_feature_importance['mean'] = df_feature_importance.mean(axis = 1)
    
    # Write prediction files
    if save_train_prediction:
        df_prediction = train_df[['employee_id', 'is_promoted']]
        df_prediction['Prediction_prob'] = test_pred_proba
        df_prediction['Prediction'] = test_pred
        df_prediction.to_csv(train_prediction_file_name, index = False)
        del df_prediction
        gc.collect()

    if save_test_prediction:
        df_prediction = test_df[['employee_id']]
        df_prediction['is_promoted_proba'] = prediction_proba
        df_prediction.to_csv(test_prediction_file_name, index = False)
        del df_prediction
        gc.collect()
    
    
    return df_feature_importance, \
           [roc_auc_train, roc_auc_test,
            precision_train[0], precision_test[0], precision_train[1], precision_test[1],
            recall_train[0], recall_test[0], recall_train[1], recall_test[1], f1_score_train[0], f1_score_train[1],
           f1_score_test[0], f1_score_test[1], 0]


def display_folds_importances(feature_importance_df_, n_folds = 5):
    n_columns = 3
    n_rows = (n_folds + 1) // n_columns
    _, axes = plt.subplots(n_rows, n_columns, figsize=(8 * n_columns, 8 * n_rows))
    for i in range(n_folds):
        sns.barplot(x = i, y = 'index', data = feature_importance_df_.reset_index().sort_values(i, ascending = False).head(20), 
                    ax = axes[i // n_columns, i % n_columns])
    sns.barplot(x = 'mean', y = 'index', data = feature_importance_df_.reset_index().sort_values('mean', ascending = False).head(20), 
                    ax = axes[n_rows - 1, n_columns - 1])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    
def df_inputSwapNoise(df, p):
    ### Takes a numpy array and swaps a row of each 
    ### feature with another value from the same column with probability p
	n = df.shape[0]
	idx = list(range(n))
	swap_n = round(n*p)
	for col in df.columns:
		arr = df[col].values
		col_vals = np.random.permutation(arr)
		swap_idx = np.random.choice(idx, size= swap_n)
		arr[swap_idx] = np.random.choice(col_vals, size = swap_n)
		df[col] = arr
	return df

def rank_gauss(x):
    # x is numpy vector
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

