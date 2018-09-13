import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 读取数
data = pd.read_csv('data/kaggle_house_price_prediction/kaggle_hourse_price_train.csv')
# 丢弃有缺失值的特征（列）
data.dropna(axis=1, inplace=True)
# 只保留整数的特征
data = data[[col for col in data.dtypes.index if data.dtypes[col] == 'int64']]
features = ['LotArea', 'BsmtUnfSF', 'GarageArea']
target = 'SalePrice'
data = data[features + [target]]
data_shuffled = shuffle(data, random_state=32)  # 这个32不要改变

num_of_samples = data_shuffled.shape[0]
split_line = int(num_of_samples * 0.7)
train_data = data.iloc[:split_line]
test_data = data.iloc[split_line:]


def get_w(x, y):
    '''
    这个函数是计算模型w的值的函数，
    传入的参数分别是x和y，表示数据与标记
    Parameter
    ----------
        x: np.ndarray，pd.Series，传入的特征数据

        y: np.ndarray, pd.Series，对应的标记

    Returns
    ----------
        w: float, 模型w的值
    '''
    # m表示样本的数量

    m = len(x)
    # 求x的均值

    x_mean = np.true_divide(np.sum(x), m)  # YOUR CODE HERE
    # 求w的分子部分

    numerator = np.sum(np.multiply(y, x - x_mean))  # YOUR CODE HERE
    # 求w的分母部分
    denominator = np.sum(np.square(x)) - np.true_divide(np.square(np.sum(x)), m)  # YOUR CODE HERE
    # 求w
    w = np.true_divide(numerator, denominator)  # YOUR CODE HERE
    # 返回w
    return w


def get_b(x, y, w):
    '''
    这个函数是计算模型b的值的函数，
    传入的参数分别是x, y, w，表示数据，标记以及模型的w值
    Parameter
    ----------
        x: np.ndarray，pd.Series，传入的特征数据

        y: np.ndarray, pd.Series，对应的标记

        w: np.ndarray, pd.Series，模型w的值

    Returns
    ----------
        b: float, 模型b的值
    '''
    # 样本个数
    m = len(x)

    # 求b
    b = np.true_divide(np.sum(y - np.multiply(x, w)), m)  # YOUR CODE HERE
    # 返回b
    return b


class myLinearRegression:
    def __init__(self):
        '''
        类的初始化方法，不需要初始化的参数
        这里设置了两个成员变量，用来存储模型w和b的值
        '''
        self.w = None
        self.b = None

    def fit(self, x, y):
        '''
        这里需要编写训练的函数，也就是调用模型的fit方法，传入特征x的数据和标记y的数据
        这个方法就可以求解出w和b
        '''
        self.w = get_w(x, y)
        self.b = get_b(x, y, self.w)

    def predict(self, x):
        '''
        这是预测的函数，传入特征的数据，返回模型预测的结果
        '''
        if self.w == None or self.b == None:
            print("模型还未训练，请先调用fit方法训练")
            return

        return self.w * x + self.b


# 创建一个模型的实例
model1 = LinearRegression()

# 使用训练集对模型进行训练，传入训练集的LotArea和标记SalePrice
model1.fit(train_data['LotArea'], train_data['SalePrice'])

# 对测试集进行预测，并将结果存储在变量prediction中
prediction1 = model1.predict(test_data['LotArea'])


def MAE(y_hat, y):
    # 请你完成MAE的计算过程                                              U
    # YOUR CODE HERE
    m = len(y)
    return np.true_divide(np.fabs(y_hat - y), m)


def RMSE(y_hat, y):
    m = len(y)
    return np.sqrt(np.true_divide(np.square(y_hat - y), m))
plt.figure(figsize = (16, 6))

plt.subplot(121)
plt.plot(train_data['LotArea'].values, train_data['SalePrice'].values, '.', label = 'training data')
plt.plot(train_data['LotArea'].values, model1.predict(train_data['LotArea']), '-', label = 'prediction')
plt.xlabel("LotArea")
plt.ylabel('SalePrice')
plt.title("training set")
plt.legend()

plt.subplot(122)
plt.plot(test_data['LotArea'].values, test_data['SalePrice'].values, '.', label = 'testing data')
plt.plot(test_data['LotArea'].values, prediction1, '-', label = 'prediction')
plt.xlabel("LotArea")
plt.ylabel('SalePrice')
plt.title("testing set")
plt.legend()
plt.show()