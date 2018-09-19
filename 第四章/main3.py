def intermediate_node_num_mistakes(labels_in_node):
    '''
    求树的结点中，样本数少的那个类的样本有多少，比如输入是[1, 1, -1, -1, 1]，返回2

    Parameter
    ----------
    labels_in_node: np.ndarray, pd.Series

    Returns
    ----------
    int：个数

    '''
    # 如果传入的array为空，返回0
    if len(labels_in_node) == 0:
        return 0

    # 统计1的个数
    num_of_one = len(labels_in_node[labels_in_node == 1])  # YOUR CODE HERE

    # 统计-1的个数
    num_of_minus_one = len(labels_in_node[labels_in_node == -1])  # YOUR CODE HERE

    return num_of_one if num_of_minus_one > num_of_one else num_of_minus_one


def gini(labels_in_node):
    '''
    计算一个结点内样本的基尼指数

    Paramters
    ----------
    label_in_data: np.ndarray, 样本的标记，如[-1, -1, 1, 1, 1]

    Returns
    ---------
    gini: float，基尼指数
    '''

    # 统计样本总个数
    num_of_samples = labels_in_node.shape[0]

    if num_of_samples == 0:
        return 0

    # 统计出1的个数
    num_of_positive = len(labels_in_node[labels_in_node == 1])

    # 统计出-1的个数
    num_of_negative = len(labels_in_node[labels_in_node == -1])  # YOUR CODE HERE

    # 统计正例的概率
    prob_positive = num_of_positive / num_of_samples

    # 统计负例的概率
    prob_negative = num_of_negative / num_of_samples  # YOUR CODE HERE

    # 计算基尼值
    gini = 1 - (np.power(prob_positive, 2) + np.power(prob_negative, 2))  # YOUR CODE HERE

    return gini

def compute_gini_indices(data, features, target, annotate=False):
    '''
    计算使用各个特征进行划分时，各特征的基尼指数

    Parameter
    ----------
    data: pd.DataFrame, 带有特征和标记的数据

    features: list(str)，特征名组成的list

    target: str， 特征的名字

    annotate: boolean, default False，是否打印注释

    Returns
    ----------
    gini_indices: dict, key: str, 特征名
                       value: float，基尼指数
    '''

    gini_indices = dict()
    # 对所有的特征进行遍历，使用当前的划分方法对每个特征进行计算
    for feature in features:
        # 左子树保证所有的样本的这个特征取值为0
        left_split_target = data[data[feature] == 0][target]

        # 右子树保证所有的样本的这个特征取值为1
        right_split_target = data[data[feature] == 1][target]

        # 计算左子树的基尼值
        left_gini = gini(left_split_target)

        # 计算左子树的权重
        left_weight = len(left_split_target) / (len(left_split_target) + len(right_split_target))

        # 计算右子树的基尼值
        right_gini = gini(right_split_target)  # YOUR CODE HERE

        # 计算右子树的权重
        right_weight = len(right_split_target) / (len(left_split_target) + len(right_split_target))  # YOUR CODE HERE

        # 计算当前结点的基尼指数
        sum_weight = right_weight + left_weight
        gini_index = ((right_weight / sum_weight) * right_gini) + (
                    (left_weight / sum_weight) * left_gini)  # YOUR CODE HERE

        # 存储
        gini_indices[feature] = gini_index

        if annotate:
            print(" ", feature, gini_index)

    return gini_indices

def best_splitting_feature(data, features, target, criterion='gini', annotate=False):
    '''
    给定划分方法和数据，找到最优的划分特征

    Parameters
    ----------
    data: pd.DataFrame, 带有特征和标记的数据

    features: list(str)，特征名组成的list

    target: str， 特征的名字

    criterion: str, 使用哪种指标，三种选项: 'information_gain', 'gain_ratio', 'gini'

    annotate: boolean, default False，是否打印注释

    Returns
    ----------
    best_feature: str, 最佳的划分特征的名字

    '''
    if criterion == 'information_gain':
        if annotate:
            print('using information gain')

        # 得到当前所有特征的信息增益
        information_gains = compute_information_gains(data, features, target, annotate)

        # information_gains是一个dict类型的对象，我们要找值最大的那个元素的键是谁
        # 根据这些特征和他们的信息增益，找到最佳的划分特征
        best_feature = max(information_gains, information_gains.get)  # YOUR CODE HERE

        return best_feature

    elif criterion == 'gain_ratio':
        if annotate:
            print('using information gain ratio')

        # 得到当前所有特征的信息增益率
        gain_ratios = compute_information_gain_ratios(data, features, target, annotate)

        # 根据这些特征和他们的信息增益率，找到最佳的划分特征
        best_feature = max(gain_ratios, gain_ratios.get)  # YOUR CODE HERE

        return best_feature

    elif criterion == 'gini':
        if annotate:
            print('using gini')

        # 得到当前所有特征的基尼指数
        gini_indices = compute_gini_indices(data, features, target, annotate)

        # 根据这些特征和他们的基尼指数，找到最佳的划分特征
        best_feature = max(gini_indices, key=gini_indices.get)  # YOUR CODE HERE

        return best_feature
    else:
        raise Exception("传入的criterion不合规!", criterion)
def create_leaf(target_values):
    '''
    计算出当前叶子结点的标记是什么，并且将叶子结点信息保存在一个dict中

    Parameter:
    ----------
    target_values: pd.Series, 当前叶子结点内样本的标记

    Returns:
    ----------
    leaf: dict，表示一个叶结点，
            leaf['splitting_features'], None，叶结点不需要划分特征
            leaf['left'], None，叶结点没有左子树
            leaf['right'], None，叶结点没有右子树
            leaf['is_leaf'], True, 是否是叶子结点
            leaf['prediction'], int, 表示该叶子结点的预测值
    '''
    # 创建叶子结点
    leaf = {'splitting_feature': None,
            'left': None,
            'right': None,
            'is_leaf': True}

    # 数结点内-1和+1的个数
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])

    # 叶子结点的标记使用少数服从多数的原则，为样本数多的那类的标记，保存在 leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1

    # 返回叶子结点
    return leaf


def one_hot_encoding(data, features_categorical):
    '''
    Parameter
    ----------
    data: pd.DataFrame

    features_categorical: list(str)
    '''

    # 对所有的离散特征遍历
    for cat in features_categorical:
        # 对这列进行one-hot编码，前缀为这个变量名
        one_encoding = pd.get_dummies(data[cat], prefix=cat)

        # 将生成的one-hot编码与之前的dataframe拼接起来
        data = pd.concat([data, one_encoding], axis=1)

        # 删除掉原始的这列离散特征
        del data[cat]

    return data


def decision_tree_create(data, features, target, criterion='gini', current_depth=0, max_depth=10, annotate=False):


    if criterion not in ['information_gain', 'gain_ratio', 'gini']:
        raise Exception("传入的criterion不合规!", criterion)

    # 复制一份特征，存储起来，每使用一个特征进行划分，我们就删除一个
    remaining_features = features[:]

    # 取出标记值
    target_values = data[target]
    print("-" * 50)
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))

    # 终止条件1
    # 如果当前结点内所有样本同属一类，即这个结点中，各类别样本数最小的那个等于0
    # 使用前面写的intermediate_node_num_mistakes来完成这个判断
    if (intermediate_node_num_mistakes(target_values) == 0):  # YOUR CODE HERE
        print("Stopping condition 1 reached.")
        return create_leaf(target_values)  # 创建叶子结点

    # 终止条件2
    # 如果已经没有剩余的特征可供分割，即remaining_features为空

    if remaining_features == []:
        print("Stopping condition 2 reached.")
        return create_leaf(target_values)  # 创建叶子结点

    # 终止条件3
    # 如果已经到达了我们要求的最大深度，即当前深度达到了最大深度

    if current_depth == max_depth:
        print("Reached maximum depth. Stopping for now.")
        return create_leaf(target_values)  # 创建叶子结点

    # 找到最优划分特征
    # 使用best_splitting_feature这个函数

    splitting_feature = best_splitting_feature(data, features, target)  # YOUR CODE HERE

    # 使用我们找到的最优特征将数据划分成两份
    # 左子树的数据
    left_split = data[data[splitting_feature] == 0]

    # 右子树的数据
    right_split = data[data[splitting_feature] == 0]  # YOUR CODE HERE

    # 现在已经完成划分，我们要从剩余特征中删除掉当前这个特征
    remaining_features.remove(splitting_feature)

    # 打印当前划分使用的特征，打印左子树样本个数，右子树样本个数
    print("Split on feature %s. (%s, %s)" % ( \
        splitting_feature, len(left_split), len(right_split)))

    # 如果使用当前的特征，将所有的样本都划分到一棵子树中，那么就直接将这棵子树变成叶子结点
    # 判断左子树是不是“完美”的
    if len(left_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(left_split[target])

    # 判断右子树是不是“完美”的
    if current_depth == max_depth:
        print("Reached maximum depth. Stopping for now.")
        return create_leaf(target_values)  # 创建叶子结点

    # 递归地创建左子树
    left_tree = decision_tree_create(left_split, remaining_features, target, criterion, current_depth + 1, max_depth,
                                     annotate)

    # 递归地创建右子树

    right_tree = decision_tree_create(right_split, remaining_features, target, criterion, current_depth + 1, max_depth,
                                      annotate)  # YOUR CODE HERE

    # 返回树的非叶子结点
    return {'is_leaf': False,
            'prediction': None,
            'splitting_feature': splitting_feature,
            'left': left_tree,
            'right': right_tree}

# 导入类库

import pandas as pd
import numpy as np
import json

# 导入数据
loans = pd.read_csv('data/lendingclub/lending-club-data.csv', low_memory=False)
# 对数据进行预处理，将safe_loans作为标记
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
del loans['bad_loans']
features = ['grade',  # grade of the loan
            'term',  # the term of the loan
            'home_ownership',  # home_ownership status: own, mortgage or rent
            'emp_length',  # number of years of employment
            ]
target = 'safe_loans'
loans = loans[features + [target]]
from sklearn.utils import shuffle

loans = shuffle(loans, random_state=34)
split_line = int(len(loans) * 0.6)
train_data = loans.iloc[: split_line]
train_data = one_hot_encoding(train_data, features)
one_hot_features = train_data.columns.tolist()
one_hot_features.remove(target)

my_decision_tree = decision_tree_create(train_data, one_hot_features, target, 'gini', max_depth=6, annotate=False)
best_splitting_feature