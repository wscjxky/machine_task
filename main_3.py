import random
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import numpy as np
spambase = np.loadtxt('data/spambase/spambase.data', delimiter = ",")
# dota2results = np.loadtxt('data/dota2Dataset/dota2Train.csv', delimiter=',')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

spamx = spambase[:, :57]
spamy = spambase[:, 57]
# dota2x = dota2results[:, 1:]
# dota2y = dota2results[:, 0]
features = ['YearRemodAdd', 'GarageArea', 'BsmtUnfSF', 'LotArea', 'TotalBsmtSF', 'BsmtFinSF1','1stFlrSF']
model = LogisticRegression()
prediction = cross_val_predict(model, spamx, spamy, cv=10)
ac_score=accuracy_score(spamy,prediction)
print(ac_score)
print(spamx.shape)
#
# for i in range(4):
#     features.pop(random.randint(0, len(features)-1))
#     print(features)
#     # data[features[0]]=np.square(data[features[0]])
#     x = spamx[features]
#     y = spamxy['SalePrice']
#     prediction = cross_val_predict(model, x, y, cv=10)
#     ab_error = mean_absolute_error(prediction, data['SalePrice'])
#     squ_error = mean_squared_error(prediction, data['SalePrice']) ** 0.5
#     print(ab_error, squ_error)
#     plt.scatter(ab_error,squ_error,10+80*i)
# plt.show()