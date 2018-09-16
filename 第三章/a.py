import random
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import numpy as np
spambase = np.loadtxt('data/spambase/spambase.data', delimiter = ",")
# dota2results = np.loadtxt('data/dota2Dataset/dota2Train.csv', delimiter=',')
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
dota2results = np.loadtxt('data/dota2Dataset/dota2Train.csv', delimiter=',')

dota2x = dota2results[:, 1:]
dota2y = dota2results[:, 0]
spamx = spambase[:, :57]
spamy = spambase[:, 57]
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(spamx, spamy, test_size=0.1, random_state=0)
spamx = spambase[:, :57]
spamy = spambase[:, 57]
# 对数据的训练集进行标准化
ss = StandardScaler()
# X_train = ss.fit_transform(X_train)  # 先拟合数据在进行标准化
lr = LogisticRegressionCV(multi_class="ovr", fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty="l2",
                          solver="lbfgs", tol=0.01)
re = lr.fit(X_train, Y_train)
#
# # 预测
# X_test = ss.transform(X_test)  # 数据标准化
# Y_predict = lr.predict(X_test)  # 预测

# Dota2数据
dx_train, dx_test, dy_train, dy_test = train_test_split(dota2x, dota2y, test_size=0.1, random_state=0)
ss2 = StandardScaler()
dx_train = ss2.fit_transform(dx_train)
lr2 = LogisticRegressionCV(multi_class="ovr", fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty="l2",
                           solver="lbfgs", tol=0.01)
re2 = lr2.fit(dx_train, dy_train)
dx_test = ss2.transform(dx_test)  # 数据标准化
dy_predict = lr2.predict(dx_test)

Y_predict= re.predict(X_test)
accuracy=accuracy_score(Y_test,Y_predict)
precision = precision_score(Y_test,Y_predict,average='binary')
recall = recall_score(Y_test,Y_predict,average='binary')
f1_score_arg = f1_score(Y_test,Y_predict,average='binary')
print(accuracy)
print(precision)
print(recall)
print(f1_score_arg)
dy_predict= re2.predict(dx_test)
accuracy2=accuracy_score(dy_test,dy_predict)
precision2 = precision_score(dy_test,dy_predict,average='binary')
recall2 = recall_score(dy_test,dy_predict,average='binary')
print(dy_test)
print(dy_predict)
f1_score2 = f1_score(dy_test,dy_predict,average='binary')
print(accuracy2)
print(precision2)
print(recall2)
print(f1_score2)