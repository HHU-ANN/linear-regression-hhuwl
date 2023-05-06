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

    weight = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))

    return weight @ data

#梯度下降法实现Lasso回归    
def lasso(data):  

    #超参数
    lr = 1e-10 #学习率
    epoch = 10000
    alpha = 0.1

    X, y = read_data()

   #初始化
    weight = np.zeros(X.shape[1])

    #梯度下降
    for i in range(epoch):
        gradient = np.dot(X.T, (np.dot(X, weight) - y)) + alpha * np.sign(weight)
        weight -= lr * gradient

    return weight @ data
   

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y