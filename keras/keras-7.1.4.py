from keras import layers, Input
from keras.models import Model
import numpy as np
import tensorflow as tf

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

#inception模块

"""
x = Input(shape = (28, 28, 192,))
branch_a = layers.Conv2D(filters = 64, kernel_size = (1, 1), strides = 1, padding = 'same', activation = 'relu')(x)

branch_b = layers.Conv2D(96, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', padding = 'same', strides=1)(branch_b)

branch_c = layers.Conv2D(16, 1, activation='relu')(x)
branch_c = layers.Conv2D(32, 3, padding = 'same', strides=1, activation='relu')(branch_c)

branch_d = layers.MaxPooling2D(pool_size = (3, 3), strides = 1, padding = 'same')(x)
branch_d = layers.Conv2D(32, 1, activation='relu')(branch_d)

output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1) #按最后一个轴进行拼接
model = Model(x, output)
model.summary()
"""


#residual模块

#resnet模块（10层之后加入有效果）
x = Input(shape = (39, 39, 10))
y = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = 1, padding = 'same', activation = 'relu')(x)
y = layers.Conv2D(128, 3, padding = 'same', activation = 'relu')(y)
y = layers.Conv2D(128, 3, padding = 'same', activation = 'relu')(y)

residual = layers.Conv2D(filters = 128, kernel_size = (1, 1), strides = 1, padding = 'same', activation = 'relu')(x)

y = layers.add([y, residual])

model = Model(x, y)
model.summary()

