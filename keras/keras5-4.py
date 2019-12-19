"""
比较重点 卷积可视化
1.可视化卷积神经网络的中间输出（中间激活）
2.可视化卷积神经网络的过滤器
3.可视化图像中类激活的热力图

h5文件的属性，model = load_model('xxx.h5')，则model.summary/layers/
"""
import keras
import numpy as np
from keras_preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import models
import cv2
import tensorflow as tf


old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
#加载之前训练好的模型，这里加载的是5.2节使用数据增强后的模型

model = load_model('cats_and_dogs_small_2.h5')
model.summary()

#=======================================================================================================================
#1.可视化中间激活

#预处理单张图片
img_path = r'E:\python_work\data\cats_and_dogs_small\test\cats\cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150)) #加载路径和resize
#print(img) #<PIL.Image.Image image mode=RGB size=150x150 at 0x2BBC682E198> 要转化成array之后才行
img_tensor = image.img_to_array(img) #将图片处理为tensor形式
print(img_tensor.shape)  #150*150*3格式
#img_tensor = np.expand_dims(img_tensor, axis=0) #表示在0位置插入ndim
#print(img_tensor.shape)
img_tensor = img_tensor.reshape(1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
print(img_tensor.shape)
img_tensor /= 255.
plt.imshow(img_tensor[0]) #四维向量用三维形式显示
plt.show()

"""
#采用opencv处理图片
img_path = r'E:\python_work\data\cats_and_dogs_small\test\cats\cat.1700.jpg'
img_src = cv2.imread(img_path)
img_src = cv2.resize(img_src, (150, 150))

b,g,r = cv2.split(img_src)
img2 = cv2.merge([r,g,b])
img2 = img2.reshape(1, img2.shape[0], img2.shape[1], img2.shape[2])
print(img2.shape)
#img2 = img2 * (1. / 255)
plt.imshow(img2[0])
plt.show()
"""

layer_outputs = [layer.output for layer in model.layers[:8]] #提取前8层的输出
#for layer_output in layer_outputs:
    #print(layer_output, type(layer_output)) #从这可以看到实际上网络中都是4D张量！
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) #创建一个模型，给定一个输入，可以返回输出

activations = activation_model.predict(img_tensor)
#print(len(activations)) #单输入多输出，输出长度为8，即8层网络结构,列表中的每一个元素是卷积或池化之后的结果！

first_layer_activation = activations[0]
print(first_layer_activation.shape) #输出为第一层卷积后的输出，注意是4D张量

#矩阵画图，第一个卷积层之后的可视化，只查看第4个通道
plt.matshow(first_layer_activation[0, : , : , 3], cmap = 'viridis') #要看3D张量的结果，就只需要让第一个值为0
#plt.show()

#下面在8个特征图中每一个中提取并绘制每一个通道，然后所有的结果叠加到一个大的图像张量中，按通道排列
layer_name = []
for layer in model.layers[:8]:
    layer_name.append(layer.name) #['conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2', 'conv2d_3', 'max_pooling2d_3', 'conv2d_4', 'max_pooling2d_4']


import keras

# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16


for layer_name, layer_activation in zip(layer_names, activations): #zip打包元组，打包后的长度为最小列表的长度


    n_features = layer_activation.shape[-1] #对应filter的数目，也就是通道数
    size = layer_activation.shape[1] #对应输出的height或width

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()
