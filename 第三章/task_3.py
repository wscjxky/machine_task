import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

spambase = np.loadtxt('data/spambase/spambase.data', delimiter=",")
dota2results = np.loadtxt('data/dota2Dataset/dota2Train.csv', delimiter=',')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

# spamx = spambase[:, :57]
# spamy = spambase[:, 57]
plt.scatter
dota2x = dota2results[:, 1:]
dota2y = dota2results[:, 0]
model = LinearDiscriminantAnalysis()
y_pred = cross_val_predict(model, spamx, spamy, cv=10)
print(y_pred)
acc = accuracy_score(y_true=spamy, y_pred=y_pred)
recall = recall_score(y_true=spamy, y_pred=y_pred)
precision = precision_score(y_true=spamy, y_pred=y_pred)
f1 = f1_score(y_true=spamy, y_pred=y_pred)

print(acc, recall, precision, f1)
y_pred = cross_val_predict(model, dota2x, dota2y, cv=10)
print(y_pred)
acc = accuracy_score(y_true=dota2y, y_pred=y_pred)
recall = recall_score(y_true=dota2y, y_pred=y_pred)
precision = precision_score(y_true=dota2y, y_pred=y_pred)
f1 = f1_score(y_true=dota2y, y_pred=y_pred)
print(acc, recall, precision, f1)