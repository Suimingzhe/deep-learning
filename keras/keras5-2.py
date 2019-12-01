"""
猫狗大战程序，由于数据集较小，故使用了数据增强！
第一部分:未使用数据增强
第二部分：使用了数据增强

图像分类问题的三种策略：1.从头开始训练一个小型模型 2.使用预训练的网络做特征提取 3.对与训练的网络进行微调

过拟合的处理方式：1.dropout层 2.数据增强

深度学习的就是梯度下降。每次的参数更新有两种方式。
第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，
计算量开销大，计算速度慢，这称为Batch gradient descent，批梯度下降。
另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，stochastic gradient descent。
现在一般采用的是一种折中手段，mini-batch gradient decent，小批的梯度下降，
这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。
另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。
基本上现在的梯度下降都是基于mini-batch的，所以经常出现的batch_size，就是指这个。

有个小技巧，函数中的batch_size是对该函数的输入而言的！

训练时候的epoch中显示的进度条后面的loss和accuracy都是针对训练集本身的！即一次梯度下降后的情况

save用于画图之前，fit之后！
-2019.11.28
"""

#==================================================================================================================
import os #可以获取路径，更多的用途以后在了解
import shutil #对文件进行操作
import numpy as np
from keras.datasets import mnist
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical
import time
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image #图像预处理工具箱

#==================================================================================================================
#对图片进行分类！首先建立文件夹！ 注意os.mkdir只能运行一遍！
original_dataset_dir = 'E:\python_work\data\kaggle_original_data' #windows使用反斜杠/，linux系统中使用斜杠\，要注意！

#首先建一个总的文件夹cats_and_dogs_small
base_dir = 'E:\python_work\data\cats_and_dogs_small' #创建一个新的路径
#os.mkdir(base_dir) #对这个新的路径创建了一个文件夹

#在总的文件夹里分别建立train，validation，test文件夹
train_dir = os.path.join(base_dir, 'train') #os.path.join是路径拼接！！！
#os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)

#猫和狗的训练目录
train_cats_dir = os.path.join(train_dir, 'cats')
#os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
#os.mkdir(train_dogs_dir)

#猫和狗的验证目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
#os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
#os.mkdir(validation_dogs_dir)

#猫和狗的测试目录
test_cats_dir = os.path.join(test_dir, 'cats')
#os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
#os.mkdir(test_dogs_dir)

#==================================================================================================================
#下面对上面建立的文件夹开始添加图片

# Copy first 1000 cat images to train_cats_dir
#fnames = ['cat.'+ str(i) +'.jpg' for i in range(1000)] #列表解析，单引号里面的格式是因为图片的格式就是这样的
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)] #显然第二种比第一种表达方式好很多！！！！！
#print(fnames)['cat.0.jpg', 'cat.1.jpg', 'cat.2.jpg', 'cat.3.jpg', 'cat.4.jpg', 'cat.5.jpg', 'cat.6.jpg',...'cat.999.jpg']
'''
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname) #os.mkdir才是创建，这里只是添加路径而已！
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst) #这一段是将图片的名字对应的内容直接copy！src-> dst

# Copy next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
'''

#检查文件夹的情况
print('total training cat images:', len(os.listdir(train_cats_dir))) #os.listdir遍历所有内容的名字，构成一个list
#print(os.listdir(train_cats_dir)) #['cat.0.jpg', 'cat.1.jpg', 'cat.10.jpg', 'cat.100.jpg',...'cat.999.jpg'],注意listdir的排列顺序！！！
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

#==================================================================================================================
#下面开始构建网络
"""

model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(150, 150, 3))) #RGB图像输入，channel=3
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid')) #sigmoid函数概率预测模型只需要一个units！

model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=1e-4), #这里与之前的optimizer='rmsprop'的区别就是这种方式是自己定义的，可以传入参数！
              loss='binary_crossentropy', #sigmoid类型的概率输出采用binary_crossentropy模型
              metrics=['accuracy']) #基于keras版本的问题，这里是全程accuracy不能写acc，否则后面画图画不出来
"""

#==================================================================================================================
#下面开始数据预处理，是比较完整的处理过程，好好记下来

"""
1.读取图像文件
2.将JPEG文件解码为RGB像素网格
3.将这些像素网格转换位浮点数张量
4.特征缩放，像素缩放到[0,1]之间，NN喜欢处理较小的像素点！


#采用ImageDataGenerator Python中，这种一边循环一边计算的机制，称为生成器：generator
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255) #将所有图像乘以1/255缩放

#文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
#ttain_dir是train的文件路径
train_generator = train_datagen.flow_from_directory(
        directory=train_dir, #该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用
        target_size=(150, 150), #图像将被resize成该尺寸
        color_mode="rgb", #表示这些图片是否会被转换为单通道或三通道的图片.默认rgb所以该语句可以省略
        batch_size=20, #批量，且是随机抽取生成！！！！！！
        class_mode='binary') #class_model表示返回的标签数组的形式，binary返回1D的二值标签，这里用改标签也是因为使用了binary_crossentropy损失！

validation_generator = validation_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=(150, 150),
        batch_size=20, #批量，且是随机抽取生成！！！！！！
        class_mode='binary')

#下面是随机的，每次结果都不一样
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    #print(data_batch)
    #print(labels_batch)
    break

"""

# ==================================================================================================================
# 下面开始训练
#在生成器上训练和之前的直接fit效果相同，这里一次20个一起输入，然后整体进行梯度下降
"""
history = model.fit_generator(train_generator,
                              steps_per_epoch=100, #一共运行100次梯度下降，这个数值算出来的，一共2000个训练样本，2000/20=100，也就是需要100次才能把所有的数据处理完
                              epochs=30,
                              verbose = 2,
                              validation_data=validation_generator,
                              validation_steps=50) #这个值也是算出来的，总共1000个validation，1000/20=50，抽取50个批次


#一次20个，由batch_size决定，step_per_epoch是算出来的，总的训练样本除以batch_size，这个数字也代表了计算loss，accuracy，梯度下降参数更新的次数
#然后epoch是进行总的循环次数，所以总的梯度下降更新参数的次数应该是100*30=3000次！

#其次就是一次多少个输入的问题，这里batch_size=20，但实际上是一张一张图片输入，对一次batch_size而言，所有参数都不变，这20个算完之后，总的计算loss和accuracy
#以及更新参数，即1次梯度下降，然后下一组20个同样的操作！！！



#将整个模型保存下来,以后直接载入模型与训练数据即可开始训练,不用再定义网络和编译模型.(这种方法已经保存了模型的结构和权重,以及损失函数和优化器
model.save('cats_and_dogs_small_1.h5')

# ==================================================================================================================
# 画图,这里是两个曲线一起画，比之前更简化！

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend() #添加plot中的label

plt.figure() #与MATLAB中figure(1),即创建了一个新的画图窗口

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
#从结果可以看到严重的过拟合！！！！
"""
# ==================================================================================================================
# 使用数据增强 data augmentation，使用的是ImageDataGenerator，之前的特征缩放/255也是用的这个

"""
#这个生成器都是根据参数随机生成的！
datagen = ImageDataGenerator(rotation_range=40, #角度值0-180
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2, #随机错切变换的角度
                             zoom_range=0.2, #随机缩放的角度
                             horizontal_flip=True, #随机将一半的图像水平翻转
                             fill_mode='nearest') #用于填充新创建像素的方法

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)] #这个fnames是完整的每一个图片的路径！
print(fnames) #['E:\\python_work\\data\\cats_and_dogs_small\\train\\cats\\cat.0.jpg', 'E:\\python_work\\data\\cats_and_dogs_small\\train\\cats\\cat.1.jpg',...]
img_path = fnames[3] #由于listdir特殊的排列顺序，实际上fnames[3]是100.jpg
print(img_path)

#img是原始图像第100张然后resize了！x是resize后处理成4d张量！
img = image.load_img(path=img_path, target_size=(150, 150)) #图片resize过程, 前者是路径，如果有红线前面加上r，为了与关键字区别！
x = image.img_to_array(img) #输出numpy数组，RGB图像自动生成x*x*3模式

x = x.reshape((1,) + x.shape) #将形状改为（1,150,150,3）

i = 0
for batch in datagen.flow(x, batch_size=10): #flow:采集数据和标签数组，生成批量增强数据。输入x一定是ndim=4的矩阵或元组！batch_size是对x处理，而x只有一个
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break #生成器都需要手动打破其无限循环！！

plt.show() #这里生成的四张图片都不一样，都是随机的！
"""

# ==================================================================================================================
# 进一步处理过拟合，加入dropout！

model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(150, 150, 3))) #RGB图像输入，channel=3
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Flatten())
model.add(layers.Dropout(rate=0.5)) #dropout在flatten之后加！
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid')) #sigmoid函数概率预测模型只需要一个units！

#model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=1e-4), #这里与之前的optimizer='rmsprop'的区别就是这种方式是自己定义的，可以传入参数！
              loss='binary_crossentropy', #sigmoid类型的概率输出采用binary_crossentropy模型
              metrics=['accuracy']) #基于keras版本的问题，这里是全程accuracy不能写acc，否则后面画图画不出来

# ==================================================================================================================
# 最后的训练！

#还是预处理过程，这里和之前相比，讲数据预处理（rescale=1./255）和数据增强进行了合并！
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

#注意，验证数据数据不能增强！！！这个名称跟书上不一样，应该是书上不严谨！训练过程与test无关！
validation_datagen = ImageDataGenerator(rescale=1./255)

#train_datagen.flow_from_directory主要干的事情：设置标签类型，resize，一次生成多少个
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150), #resize过程在.flow_from_directory中进行
                                                    batch_size=32, #这里有问题！！！表示每个批量的样本数
                                                    class_mode='binary')
for inputs_batch, labels_batch in train_generator:
    print(inputs_batch.shape)
    print(labels_batch.ndim)

    break


validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(150, 150),
                                                              batch_size=32, #这里有问题！！！
                                                              class_mode='binary')

# ==================================================================================================================
# 开始训练！
"""
tic = time.time()

#这里没有用到上述的image，所以张量就不用转换！
#注意这里的steps_per_epoch * batch_size = 100 *32 = 3200 > 2000个训练样本

history = model.fit_generator(train_generator,
                              steps_per_epoch=100, #一共运行100次梯度下降，这个数值算出来的，一共2000个训练样本，2000/20=100，也就是需要100次才能把所有的数据处理完
                              epochs=100, #每一个epochs代表循环完了一次所有的train
                              verbose = 2,
                              validation_data=validation_generator,
                              #这里为什么验证集也要分batch？不可以一起输入么？每一个epoch停顿的原因就是因为要算validation上的损失和精度！
                              validation_steps=50) #这个值也是算出来的，总共1000个validation，1000/20=50，抽取50个批次

toc = time.time()
print("Time: " + str(1000*(toc - tic)) + "ms")

#保存结果
model.save('cats_and_dogs_small_2.h5')


# ==================================================================================================================
# 画图！

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend() #添加plot中的label

plt.figure() #与MATLAB中figure(1),即创建了一个新的画图窗口

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
"""

