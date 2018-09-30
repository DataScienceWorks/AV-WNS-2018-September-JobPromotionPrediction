#Importing Library
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.cross_validation as cv
# In[5]:
#Reading Training file
data=pd.read_csv("C:\\Users\\608217107\\desktop\\ML Examples\\AnalytticsVidhya\\LoanPred\\train_LZdllc - Copy - Copy.csv")
# In[6]:
#explore the data to get insight in it
data.info()
# In[7]:
#Data imputation on missing
DataMissing=data.groupby(['department','region','education']).count()
# In[9]:
DataMissing.to_csv('C:\\Users\\608217107\\desktop\\ML Examples\\AnalytticsVidhya\\LoanPred\\Datagroup.csv')
# In[ ]:
'''
Based on Grouping info Below are the Modes of data on grouped columns
HR	region_4	Master's & above
Operations	region_10	Master's & above
Procurement	region_4	Master's & above
R&D	region_13	Master's & above
R&D	region_2	Master's & above
R&D	region_28	Master's & above
R&D	region_4	Master's & above
R&D	region_7	Master's & above
Eduction column:
Fill Master's & Above in the mentioned Depatment and region in education missing 
Apart from above since Bacholors is higest,fill the remaining NanPrevious Year rating column:
Fill Previous Year rating column -1 if KPI>80=1 & Previous Year rating column Nan
else fill -5'''
# In[ ]:
#Read test data
data_test=pd.read_csv("C:\\Users\\608217107\\desktop\\ML Examples\\AnalytticsVidhya\\LoanPred\\test.csv")
X_new=data_test.iloc[:,1:]
# In[10]:
sns.countplot("is_promoted",data=data)
#Data is imbalance
# In[11]:
# now let us check in the number of Percentage
Count_no_promotion = len(data[data["is_promoted"]==0]) # normal transaction are repersented by 0
Count_promotion  = len(data[data["is_promoted"]==1]) # fraud by 1
Percentage_of_no_promotion = Count_no_promotion/(Count_no_promotion+Count_promotion)
print("percentage of normal transacation is",Percentage_of_no_promotion*100)
Percentage_of_promotion=Count_promotion/(Count_no_promotion+Count_promotion)
print("percentage of fraud transacation",Percentage_of_promotion*100)
# In[15]:
#Data Slice
X=data.iloc[:,1:13]
Y=data.iloc[:,13:]
# In[12]:
#ReSampling - Under Sampling
from imblearn.under_sampling import RandomUnderSampler
# In[16]:
rus = RandomUnderSampler(random_state=0)
rus.fit(X, Y)
X_resampled, y_resampled = rus.sample(X,Y)
Y=pd.DataFrame(y_resampled,columns=['is_promoted'])
X=pd.DataFrame(X_resampled,columns=['department','region','education','gender','recruitment_channel','no_of_trainings','age','previous_year_rating','length_of_service','KPIs_met >80%','awards_won?','avg_training_score'])
# In[18]:
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X.department = label_encoder.fit_transform(X.department)
X.region = label_encoder.fit_transform(X.region)
X.education = label_encoder.fit_transform(X.education)
X.gender = label_encoder.fit_transform(X.gender )
X.recruitment_channel = label_encoder.fit_transform(X.recruitment_channel)
# In[ ]:
label_encoder = LabelEncoder()
X_new.department = label_encoder.fit_transform(X_new.department)
X_new.region = label_encoder.fit_transform(X_new.region)
X_new.education = label_encoder.fit_transform(X_new.education)
X_new.gender = label_encoder.fit_transform(X_new.gender )
X_new.recruitment_channel = label_encoder.fit_transform(X_new.recruitment_channel)
# In[ ]:
x_train,x_test,y_train,y_test=cv.train_test_split(X,Y,test_size=0.33,stratify=Y ,random_state=101)
# In[ ]:
param_grid = {"max_depth": [3],
              "scale_pos_weight":[2],
              'learning_rate':[0.375] ,
             'subsample':[1],
            'colsample_bytree':[1],
              "objective":['binary:logistic'], 
              "min_child_weight":[1],
              "n_estimators":[1000] ,
              "reg_alpha" : [0.1],
             "nthread":[4],
              "seed":[27],
             #"eval_metric": ['mlogloss'],
              #"early_stopping_rounds": [10],
             "gamma":[0.21]#[i/10.0 for i in range(0,5)] 
		              }
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="f1", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(x_train, y_train)
# In[ ]:
y_pred=grid_result.predict(x_test)
# In[ ]:
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
# In[ ]:
y_pred=grid_result.predict(X_new)
# In[ ]:
y_pred=pd.DataFrame(y_pred,columns=['is_promoted'])
emp=data_test.employee_id
data_final=pd.concat([emp,y_pred], axis=1)
data_final.to_csv('C:\\Users\\608217107\\desktop\\ML Examples\\AnalytticsVidhya\\LoanPred\\result.csv')