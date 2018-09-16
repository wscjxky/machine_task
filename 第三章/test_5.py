# import pandas as pd
import random

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer


def min_feature(x):
    return MinMaxScaler().fit_transform(x)


def nor_feature(x):
    return Normalizer().fit_transform(x)

def standard_feature(x):
    return StandardScaler().fit_transform(x)


def eval(y_true, y_pred):
    acc = accuracy_score(y_true,y_pred)
    ab_error = mean_absolute_error(y_true, y_pred)
    squ_error = mean_squared_error(y_true, y_pred) ** 0.5
    recall = recall_score(y_true, y_pred,average='micro')
    precision = precision_score(y_true, y_pred,average='micro')
    f1 = f1_score(y_true, y_pred,average='micro')
    return acc, recall, precision, f1, ab_error, squ_error
data = np.recfromcsv('data/wine_quality/winequality-white.csv', delimiter=";")
data = np.asarray([list(x) for x in data])


model_list = [LinearRegression(), LinearRegression(), LinearRegression(), LogisticRegression(),
              LinearDiscriminantAnalysis()]
for i, m in enumerate(model_list):
    data_x = data[:, :data.shape[1] - 1]
    data_y = data[:, data.shape[1] - 1]
    if i == 0:
        offset = random.randint(0, data.shape[1] - 2)
        data_x = data_x[:,offset:offset+1]
    if i == 1:
        data_x = data_x[:, 0:random.randint(0, data.shape[1] - 1)]
    if i == 2:
        y_pred = np.exp(cross_val_predict(m, data_x, np.log(data_y), cv=10))
        acc, recall, precision, f1, ab_error, squ_error = eval(y_true=data_y, y_pred=np.round(y_pred))
        # print(acc, recall, precision, f1, ab_error, squ_error)
        continue
    y_pred = cross_val_predict(m, data_x, data_y, cv=10)
    acc, recall, precision, f1, ab_error, squ_error = eval(y_true=data_y, y_pred=np.round(y_pred))
    # print(acc, recall, precision, f1, ab_error, squ_error)
    print(acc, recall, precision, f1, ab_error, squ_error)
