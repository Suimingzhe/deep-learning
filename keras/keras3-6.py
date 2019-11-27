"""
Boston房价预测问题
2019.11.19
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

print(train_data.shape, test_data.shape, train_targets.shape, test_targets.shape)
print(train_data.shape)

#处理数据，再Andrew的ML中也叫做“特征缩放”
mean = train_data.mean(axis = 0) #axis = 0表示每一个特征的平均值，即竖直方向！
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std

#这里注意两个点，1是测试集数据不能做任何处理，也就是测试集的std和mean也要用数据集；2是x-u/标准差，这个标准差减不减u都一样
test_data -= mean
test_data /= std

#以函数的形式构建网络
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape = (train_data.shape[1], ))) #输入的时候是一个一个输入的
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer = 'rmsprop',
                  loss = 'mes',
                  metrics = ['mae']) #mes和mae是回归问题常用的指标
    