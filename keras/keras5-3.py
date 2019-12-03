"""
使用VGG16来进行预训练网络，from keras.applications import VGG16

训练网络的方法： 1.特征提取 2.微调模型
1. 特征提取就是去除之前训练好的网络的卷积基（convolution+maxpooling）,完全连接层不包含空间信息
(1)不使用数据增强的快速特征提取，即调用conv_base的输出作为新的Dense层的输入，但这样进无法进行数据增强
优点：最后加入了两个层，所以epoch速度很快，但是原始的VGG16来处理训练图片很慢
(2)进行扩展，即原来的conv_base加上Dense作为整个大模型，但计算代价比第一种高,速度慢，优点就是可以使用数据增强，端到端
2.微调模型，与特征提取互为补充，就是解冻VGG16的顶部模块（卷积层） + 自己设计的完全连接层，一起更新参数
1) Add your custom network on top of an already trained base network.
2) Freeze the base network.
3) Train the part you added.
4) Unfreeze some layers in the base network.
5) Jointly train both these layers and the part you added.
这里为什么解冻VGG顶部的几个卷积层，而下面的不解冻，因为下面的部分刚刚开始，提取到的更多是可复用特征，即得到的回报少，而上面的卷积层是更专业化的特征
而且不能解冻太多也是因为参数太多而过拟合！

"""
import os
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

#=======================================================================================================================
#导入VGG16

from keras.applications import VGG16

conv_base = VGG16(weights = 'imagenet', #指定模型初始化的权重检查点
                  include_top = False, #表示是否包含全连接层，VGG16有三个全连接层，最后一个是softmax，这三个都不包括
                  input_shape = (150, 150, 3)) #最好输入，如果不写，网络也可以处理任意形状的输入

conv_base.summary()
#=======================================================================================================================
#不使用数据增强快速提取特征,并且加入Dence和dropout

base_dir = 'E:\python_work\data\cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
"""
#定义了一个特征提取的函数，其实就是利用VGG16来进行训练！但是这个训练只是预测结果，因为权重保存下来了！
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512)) #np.zeros里面两个括号！shape可省略,ndim=4
    labels = np.zeros(shape=(sample_count)) #ndim=1
    generator = datagen.flow_from_directory(directory, #参数也是路径，这里实际是train_dir
                                            target_size=(150, 150), #处理为150*150 size, 这里仅针对height和width
                                            batch_size=batch_size, #一次提取20张图片
                                            class_mode='binary') #sigmoid函数采用binary分类,这里会输出Found.....in to 2 classes!，怎么分类的不知道...就是0和1是如何定义的，不知道...
    i = 0
    for inputs_batch, labels_batch in generator: #这里inputs_butch就是(20，150，150，3)的4D张量，out_butch就是20的标签，ndim=1
        features_batch = conv_base.predict(inputs_batch) #提取到的特征
        features[i * batch_size : (i + 1) * batch_size] = features_batch #将定义的0向量替换[0:20][20:40]...
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch #标签不变！只是特征变了！
        i += 1
        print(i)
        if i * batch_size >= sample_count: #>2000训练样本跳出
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000) #注意!这里的train_features就是上述处理的（2000*4*4*512）的形式，而train_labels还是（2000，）的原始标签不变！！
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

#平铺

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

#加入完全连接层和dropout

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512)) #这里要注意，跟样本无关，就是这样，或者input_shape = （4*4*512）也行！
model.add(layers.Dropout(0.5)) #dropout用于最后一层之前！
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy', #重要的是这个！对应sigmoid
              metrics=['accuracy'])

#=======================================================================================================================
#训练,这里没有数据增强，也就是只有fit的时候显示的就是20,40,60.../2000,而且也不需要validation_steps！
# 使用fit_generator时候才会是1,2,3.../step_per_epoch，这里的1,2,3代表的是随机增强里面的batch_size！

history = model.fit(train_features,
                    train_labels,
                    epochs=30,
                    batch_size=20, #这里不同于5-1需要step，因为之前用到的batch_size=20 已经完全处理为4*4*512了，并且连接在了一起！
                    validation_data=(validation_features, validation_labels))

#=======================================================================================================================
#画图

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
"""
#=======================================================================================================================
#使用数据增强
#首先搭建网络，因为是利用VGTG16的权重，所以一定要冻结！
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


#冻结权重的方法：属性.trainable=False，这里trainable只的是可以训练的主权重矩阵和偏置向量，所以conv和dense层一层2个，maxpooling层不算
print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))
#这里一定是先冻结！而后再编译！！
conv_base.trainable = False
print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))



train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100, #这里虽然是100*20=2000，但是每一个epoch都是随机数据增强，所以输入都不一样，无数据增强的fit的话就都是一样的了
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)

#=======================================================================================================================
#画图
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#结果表明，利用VGG16 + 自己定义的dense层和softmax，并且使用数据增强（第二种方式）结果最好，验证精度达到96%


#下面进行预训练第二种方法，微调 fine-tuning,记住这是两部fit，第一步就是上面的方法二，第二部是解冻之后的fit如下所示！
conv_base.trainable = True
print(len(model.trainable_weights)) #显示30，一共13conv + 2个dense 共30个可训练权重矩阵和偏差

#这一段写的好，不用if if if这样按名字写
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

print(len(model.trainable_weights)) #显示10 顶部3个conv + 2个dense 共10

#解冻过后开始第二阶段的fit

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)

#=======================================================================================================================
#画图
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


#=======================================================================================================================
#画图刚好看，曲线更平滑的方法!!!-让每个损失和精度都替换为指数移动平均值！ EMA处理！
#acc = points,是一个列表的值

def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points: #空为false
      previous = smoothed_points[-1] #将要加入到列表末位置之前 的最后一个元素提取出来
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point) #只是起到加入第一个值这样的作用
  return smoothed_points #返回的列表相当于[10, 10*0.8+11*0.2, ...] 原来的是[10,11,...],简单来说就是距离变近了


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#=======================================================================================================================
#在测试集上最终确认，注意这里，测试集也是生成器，而且20*50 = 1000张测试图片！
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)