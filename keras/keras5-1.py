import numpy as np
from keras.datasets import mnist
from keras import layers
from keras import models
from keras.utils import to_categorical
import time
import matplotlib.pyplot as plt
#==================================================================================================================

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#下面首先是数据的归一化输入,类似于特征缩放
train_images = train_images.reshape((60000, 28, 28, 1 )) #这里明显区别，如果是keras2-1dense层的话，输入格式是（60000，28*28）
train_images = train_images.astype('float32') / 255 #这里是/255后取float32型的数据，之前是float64 占内存,而且/255之后类似特征缩放

test_images = test_images.reshape((10000, 28, 28, 1)) #one-hot编码，多分类问题的常用方法！
test_images = test_images.astype('float32') / 255


#准备标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#==================================================================================================================
#模型搭建！

model = models.Sequential()#这个shape比layers.Dense好理解多了！！！
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))


model.add(layers.Flatten()) #units平铺
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

tic = time.time()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#==================================================================================================================
#利用交叉验证集进行测试
images_val = train_images[:10000]
partial_train_images = train_images[10000:]
labels_val = train_labels[:10000]
partial_train_labels = train_labels[10000:]


history = model.fit(partial_train_images,
                    partial_train_labels,
                    epochs = 10,
                    verbose = 2,
                    batch_size = 128,
                    validation_data = (images_val, labels_val)) #卷积一次输入的64张图片，不是很多

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