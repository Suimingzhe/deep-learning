from keras import models
from keras import layers
from keras import optimizers


model = models.Sequential() #定义模型的两种方法之一：Sequrntial方法！ 先创建一个空的sequential
model.add(layers.Dense(32, activation= 'relu', input_shape=(784,))) #只接受第一个维度大小为784的2D张量（numpu矩阵）
model.add(layers.Dense(10, activation='softnax')) #第二层可以需要input_shape的值，自动根据上一层的输出来确定

model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = 'mse',
              metrics = ['accuracy']) #编译配置过程

model.fit(input_tensor, target_tensor, batch_size = 128, epochs = 10)

"""
采用第二种模型定义方式API
from keras import models
from keras import layers

input_tensor = layers.Input(shape = (784,))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs = input_tensor, outputs = output_tensor)
"""




