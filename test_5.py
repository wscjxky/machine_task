# import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

data = np.recfromcsv('data/wine_quality/winequality-white.csv', delimiter=";")
data =np.asarray( [list(x) for x in data])
data_x=data[:,:data.shape[1]-1]
data_y=data[:,data.shape[1]-1]

model_list=[LinearRegression(),LogisticRegression(),LinearDiscriminantAnalysis()]
for m in model_list:
    y_pred=cross_val_predict(m,data_x,data_y,cv=10)
    try:
        acc=accuracy_score(y_true=data_y,y_pred=y_pred)
    except ValueError as e:
        print("结果出现小数")
        acc=accuracy_score(y_true=data_y,y_pred=np.around(y_pred))
    print(acc)
