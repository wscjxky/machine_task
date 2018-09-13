import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('data/kaggle_house_price_prediction/kaggle_hourse_price_train.csv')
# 丢弃有缺失值的特征（列）
data.dropna(axis=1, inplace=True)
# 只保留整数的特征
data = data[[col for col in data.dtypes.index if data.dtypes[col] == 'int64']]
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

features = ['YearRemodAdd', 'GarageArea', 'BsmtUnfSF', 'LotArea', 'TotalBsmtSF', 'BsmtFinSF1','1stFlrSF']
model = LinearRegression()
for i in range(4):
    features.pop(random.randint(0, len(features)-1))
    print(features)
    # data[features[0]]=np.square(data[features[0]])
    x = data[features]
    y = data['SalePrice']
    prediction = cross_val_predict(model, x, y, cv=10)
    ab_error = mean_absolute_error(prediction, data['SalePrice'])
    squ_error = mean_squared_error(prediction, data['SalePrice']) ** 0.5
    print(ab_error, squ_error)
    plt.scatter(ab_error,squ_error,10+80*i)
plt.show()