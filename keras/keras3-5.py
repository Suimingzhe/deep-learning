"""
多分类问题
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
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
#print(len(train_data), len(train_labels), len(test_data), len(test_labels)) #一共8982个train样本，2246个test样本
#print(train_data[10]) #这些返回类型都是列表！
#print(train_labels[10])

#==================================================================================================================

#进行向量化
#首先是输入数据的向量化，跟之前一样，采用one-hot方法
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences): #i是每一行，而sequence对应每一个值,是列表形式！这里的sequences本身就是样本数*1的形式，而这个1是列表的形式！
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


#然后是标签的向量化！这与3.4不一样，3.4的标签只有0/1，而这里的是多类的one-hot形式，FER就要用这种形式！！！
one_hot_train_labels = to_categorical(train_labels) #转化为8983*46的形式
one_hot_test_labels = to_categorical(test_labels) #转化为2246*46的形式

#print(one_hot_test_labels.shape)
#print(one_hot_test_labels.ndim) #张量为2
#print(one_hot_train_labels.shape)
#print(one_hot_train_labels.ndim) #张量为2

#==================================================================================================================
#下面开始
'''
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

#==================================================================================================================
#下面开始进行compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#==================================================================================================================
#利用交叉验证集进行测试
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#==================================================================================================================
#开始训练
tic = time.time()
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    verbose = 2,
                    validation_data=(x_val, y_val))
toc = time.time()
print("Time: " + str(1000*(toc - tic)) + "ms")

#==================================================================================================================
#开始画图

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label="Validation loss")
plt.legend()
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label="Validation acc")
plt.legend()
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
'''
#==================================================================================================================
#找到合适的epoch，开始重新训练

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,
          one_hot_train_labels,
          epochs=8,
          batch_size=512)


test_loss, test_acc = model.evaluate(x_test, one_hot_test_labels)
print("test_loss:" , test_loss) #这里用加号和逗号的区别就是冒号后面需不需要加空格！！！
print("test_acc:" , test_acc)

#==================================================================================================================
#在测试集上进行预测

predictions = model.predict(x_test)
print(predictions.shape)
print(type(predictions))
print(predictions.dtype)
print(predictions.ndim)
print(predictions[0].shape)
print(predictions[0][0])
print(predictions)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))