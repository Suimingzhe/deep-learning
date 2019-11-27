"""
电影评论分类：二分类问题
2019.11.12
"""

from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import imdb #电影评论的数据集,注：该数据集已经预处理过了
import numpy as np
from keras import losses #compile中的参数
from keras import metrics #compile中的参数
import matplotlib.pyplot as plt
import time




(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000) #这里的10000是指保留频率最高的前10000个数据，注意这里的数据都是列表！！！列表嵌套列表

#print(train_data[0])
#print(train_labels[0])
#print(max([max(sequence) for sequence in train_data])) #numpy数组解析！！！

#============================================================================================================================

#这段有意思，可以加强一下小知识！！！
word_index = imdb.get_word_index() #word_index是一个将单词映射为整数的字典(good - 5)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #创建一个新的字典并交换键值对！即整数映射为单词
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]]) #get函数是取字典中的值,?表示如果没有对应的话用？来表示
#print(decoded_review)

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
                    epochs = 20,
                    verbose = 2,
                    batch_size = 512, #批量样本，说明一次一个进入训练
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

#=======================================================================================
#下面开始重新训练网络，在最佳位置停止
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train,
          y_train,
          epochs = 4,
          verbose = 2,
          batch_size = 512) #这里不加入交叉验证集了，所以从程序来看交叉验证集的目的就是为了帮助我找到训练的最佳epoch!
                            #而且最重要的是当找到epoch之后，训练的是所有的train包括交叉验证集！！！！！

#x下面开始正式的test
test_loss, test_acc = model.evaluate(x_test,y_test) #这里test时候没有batch_size了！！！！
print("test_loss:" , test_loss) #这里用加号和逗号的区别就是冒号后面需不需要加空格！！！
print("test_acc:" , test_acc)

prediction = model.predict(x_test)
print(prediction) #用于test的25000个结果



