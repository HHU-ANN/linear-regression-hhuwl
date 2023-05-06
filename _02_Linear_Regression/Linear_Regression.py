# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

#最小二乘实现岭回归
def ridge(data):

    X, y = read_data()

    # 样本数量和特征维度
    N, D = X.shape

    # 超参数
    alpha = 0.05

    A = np.eye(D) * alpha
    w_ridge = np.linalg.inv(X.T * X + alpha * A) @ X.T * y

    return w_ridge * data

#梯度下降法实现Lasso回归    
def lasso(data):  

    X, y = read_data()

    # 超参数
    alpha = 0.01
    epoch = 10000
    lr = 0.001

    # 初始化w
    w = np.zeros(X.shape[1])

    # 梯度下降
    for i in range(epoch):
        # 梯度
        grad = np.matmul(X.T, np.matmul(X, w) - y) / X.shape[0] + alpha * np.sign(w)
        # 防止梯度爆炸
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1:
            grad /= grad_norm
        # 更新梯度
        w -= lr * grad

    return np.matmul(w, data)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y