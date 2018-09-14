# import pandas as pd
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

def min_feature(x):
    return MinMaxScaler().fit_transform(x)
def standard_feature(x):
    return StandardScaler().fit_transform(x)
data = np.recfromcsv('data/wine_quality/winequality-white.csv', delimiter=";")
data =np.asarray( [list(x) for x in data])
data_x=data[:,:data.shape[1]-1]
data_y=data[:,data.shape[1]-1]

def eval(y_true,y_pred):
    acc = accuracy_score(y_true, y_pred)
    ab_error = mean_absolute_error(y_true, y_pred)
    squ_error = mean_squared_error(y_true,y_pred) ** 0.5
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc,recall,precision,f1,ab_error,squ_error
model_list=[LinearRegression(),LogisticRegression(),LinearDiscriminantAnalysis()]
for m in model_list:
    y_pred=cross_val_predict(m,data_x,data_y,cv=10)
    try:
        acc=accuracy_score(y_true=data_y,y_pred=y_pred)
        mean_squared_error(data_y,y_pred)
    except ValueError as e:
        print("结果出现小数")
        acc=accuracy_score(y_true=data_y,y_pred=np.around(y_pred))

    print(acc)
