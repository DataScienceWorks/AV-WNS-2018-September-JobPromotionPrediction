

import pandas as pd
from catboost import CatBoostRegressor as cbr
from catboost import CatBoostClassifier as cbc
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import OneHotEncoder 
import random
from sklearn.metrics import f1_score



training_dataset = pd.read_csv( r'train.csv')
test_dataset = pd.read_csv(r'test.csv')
test_dataset['is_promoted'] = [0]*len(test_dataset)
dataset = pd.concat([training_dataset, test_dataset], ignore_index = 1)
'''
dataset = pd.read_csv(r'train.csv')
'''
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Handling Missing values

# predict "education" column by regression

# "education" column
lst = []
foo = ["Bachelor's", "Master's & above","Below Secondary"]
#print(random.choice(foo))
for i in list(X[:, 2]):
    if str(i) == 'nan' : 
        lst.append(random.choice(foo))
    else:    
        lst.append(str(i))
X[:, 2] = lst


# predict "previous_year_rating" column by regression
# "previous_year_rating" column
lst = []
foo1 = [1,2,3,4,5]
#print(random.choice(foo))
for i in list(X[:, 7]):
    if str(i) == 'nan' : 
        lst.append(random.choice(foo1))
    else:    
        lst.append(str(i))
X[:, 7] = lst

'''
reg_dataset = pd.DataFrame(X).dropna()
X_reg_train = reg_dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]].values
y_reg_train = reg_dataset.iloc[:, 7].values
#extract rows with NaN
rows_to_be_pred = pd.DataFrame(X)[pd.DataFrame(X)[7].isnull()]
#test data for regressor-rows for which values need to be predicted.
X_reg_test = rows_to_be_pred.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]].values
regressor = cbr(cat_features=[0, 1, 2, 3, 4])
regressor.fit(X_reg_train, y_reg_train)

y_pred=regressor.predict(X_reg_test)
y_pred2 = [round(i) for i in list(y_pred)]

lst = []
index = 0
for i in list(X[:, 7]):
    if str(i) == 'nan': 
      lst.append(y_pred2[index])
      index += 1
    else: lst.append(i)
X[:, 7] = lst

'''

'''
for i in list(X[:, 7]):
    if str(i) == 'nan' :
        print('nan')
    else:
        print('not')'''
#encoding categorical data to numeric
'''
labelencoder = LE()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

labelencoder1 = LE()
X[:, 1] = labelencoder1.fit_transform(X[:, 1])

labelencoder1 = LE()
X[:, 2] = labelencoder1.fit_transform(X[:, 2])

labelencoder3 = LE()
X[:, 3] = labelencoder3.fit_transform(X[:, 3])

labelencoder4 = LE()
X[:, 4] = labelencoder4.fit_transform(X[:, 4])

onehotencoder = OneHotEncoder(categorical_features = [-12])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder1 = OneHotEncoder(categorical_features = [-11])
X = onehotencoder1.fit_transform(X).toarray()

onehotencoder2 = OneHotEncoder(categorical_features = [-10])
X = onehotencoder2.fit_transform(X).toarray()

onehotencoder3 = OneHotEncoder(categorical_features = [-9])
X = onehotencoder3.fit_transform(X).toarray()

onehotencoder4 = OneHotEncoder(categorical_features = [-8])
X = onehotencoder4.fit_transform(X).toarray()

'''


#splitting data to training and test data
X_train = X[:len(training_dataset), :]
X_test = X[len(training_dataset):, :]
y_train = y[:len(training_dataset)]
'''
from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify=y)
'''

#oversampling
train_df = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
train_df.columns = list(range( len(train_df.columns) )) # list(range(13))


train_df__0 = train_df[train_df[len(train_df.columns)-1] == 0]
train_df__1 = train_df[train_df[len(train_df.columns)-1] == 1]
#oversampled_df = pd.concat([train_df__0 , pd.concat([train_df__1] * 10)], ignore_index=1)
oversampled_df = pd.concat([train_df__0, train_df__1], ignore_index=1)


from sklearn.utils import shuffle
oversampled_df = shuffle(oversampled_df)

X_train = oversampled_df.iloc[:, :-1].values
y_train = oversampled_df.iloc[:, -1].values

'''
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
'''
'''
#handling imbalanced data
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train, y_train = sm.fit_sample(X_train, y_train.ravel())
'''

#Classifier

classifier = cbc( eval_metric='F1', scale_pos_weight = 3.1,cat_features=[0, 1, 2, 3, 4])
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
'''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm, f1_score(y_test, y_pred))
'''


#output
output = test_dataset[['employee_id']]
output['is_promoted'] = list(y_pred)
output.to_csv(r'sample_submission.csv', sep=',', encoding='utf-8', index=False)
