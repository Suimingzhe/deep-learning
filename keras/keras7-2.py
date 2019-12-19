"""
使用tensorboard
"""
"""
电影评论分类：二分类问题
2019.11.12
"""


"""
import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import imdb #电影评论的数据集,注：该数据集已经预处理过了
import numpy as np
from keras import losses #compile中的参数
from keras import metrics #compile中的参数
import matplotlib.pyplot as plt
import time
import tensorflow as tf

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)




(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000) #这里的10000是指保留频率最高的前10000个数据，注意这里的数据都是列表！！！列表嵌套列表

#print(train_data[0])
#print(train_labels[0])
#print(max([max(sequence) for sequence in train_data])) #numpy数组解析！！！

#============================================================================================================================

'''
这里先要对数据进行向量化编码！！！
'''
def vectorize_sequences(sequences, dimension=10000): #处理后每一个输入都是10000维的向量！！！！！这里的sequence实际上等于25000
    # 先创建一个0向量
    results = np.zeros((len(sequences), dimension)) #len(sequence)表示测出样本的数量25000，并且向量化一个二维数组！！！！！太重要了吧！！！
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  #这里第一次见到函数的嵌套，第二个索引是列表的形式，妙！
    return results

#向量化训练数据
x_train = vectorize_sequences(train_data)
#向量化测试数据
x_test = vectorize_sequences(test_data)

#下面对标签进行向量化
y_train = np.asarray(train_labels).astype('float32') #采用asarry的区别就是不会占用新的内存
y_test = np.asarray(test_labels).astype('float32') #这里原来的输出形式是列表，要转化为numpy


#============================================================================================================================

'''
下面构建网络
'''

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) #多个样本的情况下，是一个一个输入的，所以每次输入的是1D张量
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) #对于input_size只需要写定义的第一层即可！！！

#一般而言EarlyStopping和ModelCheckpoint一起使用
callbacks_list = [

    keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.1,
        patience = 2
    )
]


#下面进行编译模型选择
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

x_val = x_train[:10000] #交叉验证集 前10000个数据
partial_x_train = x_train[10000:] #真正的训练集 15000个

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#x下面开始训练
tic = time.time()
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 10,
                    verbose = 2,
                    batch_size = 512, #批量样本，说明一次一个进入训练
                    callbacks = callbacks_list,
                    validation_data = (x_val, y_val)) #新加入的，在交叉验证集上验证准确略
toc = time.time()
print("Time: "+ str(1000*(toc - tic)) + "ms")

#=====================================================================================================
#下面画图,history.history是一个字典，包含val_acc(验证精度) acc(训练精度) val_loss(验证损失) loss(训练损失)四个键
history_dict = history.history #先加载字典！

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss'] #提取四个键所对应的值,是一堆数字


epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss') #记下plt.plot的使用形式
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#============================================================================================
plt.clf() #清空图像，窗口打开，可以被重复使用


acc = history.history['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
"""
#============================================================================================
#使用tensorboard

import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import plot_model

max_features = 2000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()

"""
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir = r'E:\python_work\deep-learning\keras\my_log_dir', #日志文件将被写入这个位置
        histogram_freq = 1, #每一轮之后记录激活直方图
        #embeddings_freq = 1 #每一轮之后记录嵌入数据
    )
]

model.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['accuracy'],
)

history = model.fit(
    x_train,
    y_train,
    epochs=1,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks
)
"""

plot_model(model, show_shapes = True, to_file=r'E:\python_work\deep-learning\keras\mdoel2.png')
