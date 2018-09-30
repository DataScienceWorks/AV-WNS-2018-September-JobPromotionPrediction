



''' #TODO: Paths '''
train_path = r'C:\Users\vishal.gupta33\Desktop\AnalyticsVidhya_14Sep18\train.csv'
test_path = r'C:\Users\vishal.gupta33\Desktop\AnalyticsVidhya_14Sep18\test.csv'
sample_submission_path = r'C:\Users\vishal.gupta33\Desktop\AnalyticsVidhya_14Sep18\sample_submission.csv'





import pandas as pd


dataset_train = pd.read_csv(train_path)
dataset_test = pd.read_csv(test_path)


dataset_test['is_promoted'] = [0]*len(dataset_test)
dataset = pd.concat([dataset_train, dataset_test], ignore_index = 1)


X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values




''' # Manually treating empty cells for "education" column
#  - replacing all empty by the most freq. value -"Bachelor's"
# Also, doing manual encoding such that I can use this column as a numerical column later.'''

l = []
for i in list(X[:, 2]):
    if str(i) == 'nan' or i == "Bachelor's": l.append(1)
    elif i == "Master's & above": l.append(2)
    else:    l.append(0)
X[:, 2] = l




''' Using Regression to predict missing values for "previous_year_rating" column '''
dataset_reg = pd.DataFrame(X).dropna()

X_reg = dataset_reg.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]].values
y_reg = dataset_reg.iloc[:, 7].values


df_to_be_pred = pd.DataFrame(X)[pd.DataFrame(X)[7].isnull()]
df_to_be_pred2 = df_to_be_pred.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]].values


from catboost import CatBoostRegressor
regressor = CatBoostRegressor(cat_features=[0, 1, 3, 4], logging_level = 'Silent')
regressor.fit(X_reg, y_reg)

y_pred__of_df_to_be_pred2 = [round(i) for i in list(regressor.predict(df_to_be_pred2))]


l = []
index = 0
for i in list(X[:, 7]):
    if str(i) == 'nan': 
      l.append(y_pred__of_df_to_be_pred2[index])
      index += 1
    else: l.append(i)
X[:, 7] = l




''' Label Encoding '''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

labelencoder1 = LabelEncoder()
X[:, 1] = labelencoder1.fit_transform(X[:, 1])

labelencoder3 = LabelEncoder()
X[:, 3] = labelencoder3.fit_transform(X[:, 3])

labelencoder4 = LabelEncoder()
X[:, 4] = labelencoder4.fit_transform(X[:, 4])





''' Test-Train Split'''
X_train = X[:len(dataset_train), :]
X_test = X[len(dataset_train):, :]
y_train = y[:len(dataset_train)]




''' Classification '''
from catboost import CatBoostClassifier
classifier = CatBoostClassifier(eval_metric='F1', 
                                cat_features=[0, 1, 3, 4], 
                                scale_pos_weight = 3, 
                                logging_level = 'Silent')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)




''' Writing the output file '''
output_df = dataset_test[['employee_id']]
output_df['is_promoted'] = list(y_pred)
output_df.to_csv(sample_submission_path, sep=',', encoding='utf-8', index=False)
