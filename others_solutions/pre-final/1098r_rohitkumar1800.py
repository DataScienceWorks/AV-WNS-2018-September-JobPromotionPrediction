'''
Created on 16-Sep-2018

@author: rohit
'''
# -*- coding: utf-8 -*-
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import csv
import numpy as np
from collections import defaultdict

DATA_PATH = "datasets/"
# MODEL_PATH = DATA_PATH + 'SGD_Model/'
MODEL_PATH = DATA_PATH + 'MultinomialNB_Model/'
MODEL_FILE_EXTENSION = "pkl"
TRAINING_FILE = "train_LZdllcl.csv"
TEST_FILE = "test_2umaH9m.csv"
TARGET_FILE = 'predictions.csv'

def save_to_file(filename, context, folder_path=MODEL_PATH):
    if folder_path.strip():
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if folder_path.strip()[-1] != '/':
            folder_path = folder_path.strip() + "/"
    joblib.dump(context, folder_path + filename + "." +
                MODEL_FILE_EXTENSION)
    return (" file <{}> successfully saved".format(
        folder_path + filename + "." +
        MODEL_FILE_EXTENSION))


def load_file(filename, folder_path=MODEL_PATH):
    if folder_path.strip() and folder_path.strip()[-1] != '/':
        folder_path = folder_path.strip() + "/"
    return joblib.load(folder_path + filename + "." +
                       MODEL_FILE_EXTENSION)


class DataCleaner:
    def __init__(self):
        self.stats = defaultdict(lambda:{"data":defaultdict(int),"max":0, "indexes":{}})
        
    def transform(self, data):
        result_data = []
        for d_t in data:
            d = np.array([self.change_dt(k, d_t[k]) for k in SELECTED_HEADERS])
            result_data.append(d)
        return result_data

    def change_dt(self, key, data, method='transform'):
        if data:
            try:
                data = int(data)
            except:
                pass
        elif method=='transform':
            data = self.stats[key]['max']
        if data not in self.stats[key]['indexes']:
            self.stats[key]['indexes'][data] = len(self.stats[key]['indexes'])
            
        return self.stats[key]['indexes'][data]
    
    def generate_stats(self, data):
        for d_t in data:
            for k in SELECTED_HEADERS:
                d = self.change_dt(k, d_t[k], 'stats')
                self.stats[k]['data'][d]+=1
        for k, v in self.stats.items():
            for k1,_ in sorted(v['data'].items(),key=lambda k:k[1], reverse=True):
                if k1!='':
                    v['max']=k1
                    break
            print "| {} | {} |".format(k, v['max'])
    
    def apply(self, data, optype='test'):
        if optype=="training":
            self.generate_stats(data)
        data = self.transform(data)
        return data
        
        

'''
Feature Selection
'''
SELECTED_HEADERS = [
    #employee_id
    "department",
    "region",
    "education",  
    "gender",
    "recruitment_channel",
    "no_of_trainings",
    "age",
    "previous_year_rating",
    "length_of_service",
    "KPIs_met >80%",
    "awards_won?",
    "avg_training_score"
]

TARGET_FIELD = "is_promoted"
EMPLOYEE_ID_FIELD = "employee_id"
dc = DataCleaner()

def create_sample_data_set(data_file_path, optype='test'):
    data = []
    targets = []
    employee_ids = []
    f = open(data_file_path, "rb")
    fr = csv.DictReader(f)
    for row in fr:
        for key_field in row.keys():
            if key_field==TARGET_FIELD:
                targets.append(row[key_field])
                del row[key_field]
            elif key_field==EMPLOYEE_ID_FIELD:
                employee_ids.append(row[key_field])
                del row[key_field]
            elif key_field not in set(SELECTED_HEADERS):
                del row[key_field]
        data.append(row)
    return_object = {}
    return_object['data'] = dc.apply(data, optype)
    return_object['targets'] = targets
    return_object['employee_ids'] = employee_ids
    print "data length: {}".format(len(data))
    return return_object


def train_model(training_file_path="train_LZdllcl.csv"):
    data_dict = create_sample_data_set(training_file_path,optype='training')
    #clf = SGDClassifier(loss="modified_huber")
    clf = MultinomialNB()
    clf = clf.fit(data_dict['data'], data_dict['targets'])
    print save_to_file("Model", clf)


def test_model(test_file_path="test_2umaH9m.csv", predicted_file_path='predictions.csv'):
    text_clf = load_file("Model")
    test_data = create_sample_data_set(test_file_path)

    predicted = text_clf.predict(test_data['data'])
    f = open(predicted_file_path, 'wb')
    fw = csv.writer(f, [EMPLOYEE_ID_FIELD, TARGET_FIELD])
    fw.writerow([EMPLOYEE_ID_FIELD, TARGET_FIELD])
    for i, prediction in enumerate(predicted):
        employee_id = test_data['employee_ids'][i]
        fw.writerow([employee_id, prediction])
    f.close()
    print "Done"


def test_model_on_manual_data(test_data):
    text_clf = load_file("Model")
    predicted = text_clf.predict(test_data)
    for i, prediction in enumerate(predicted):
        print "Prediction: {},  Original Target: {}".format(prediction, test_data[i]['original_target'])
    print "Done"


if __name__ == "__main__":
    train_model(DATA_PATH+TRAINING_FILE)
    test_model(DATA_PATH+TEST_FILE, MODEL_PATH+TARGET_FILE)
#     test_data = [
#         
#     ]
#     test_model_on_manual_data(test_data)
