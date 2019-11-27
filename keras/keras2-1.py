from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras import optimizers

import os
os.environ['KERAS_BACKEND']='tensorflow' #令keras的后端为tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

"""
print(train_images.shape)
print(len(train_labels))
print(train_labels) #这个是训练的标签 注意：这是一个numpy数组

print(test_images.shape)
print(len(test_labels))
print(test_labels)
"""

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax')) #这里定义了两个完全连接层，其中最后一层是softmax函数，第一层是relu函数！

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy']) #编译步骤，包含三个参数：损失函数，优化器，训练过程中需要监控的指标，本题是准确率！


#下面首先是数据的归一化输入
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255 #这里是/255后取float32型的数据，之前是float64 占内存

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#准备标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#开始训练网络，通过调用网络的fit来完成拟合过程
network.fit(train_images, train_labels, epochs=5, batch_size=128)


#下面看看在测试集上性能如何
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)