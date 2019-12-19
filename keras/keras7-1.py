from keras import Input, layers
import tensorflow as tf
from keras.models import Sequential, Model #Model为转换为一个模型
import numpy as np

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

"""
inout_tensor = Input(shape = (32, )) #一个张量

dense = layers.Dense(units = 32, activation = 'relu') #一个层是一个函数
output_tensor = dense(inout_tensor) #可以在一个张量上调用一个层，它会返回一个张量
"""

"""
#之前的采用keras Sequential搭建模型

seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

seq_model.summary()
"""

#采用keras API函数式来搭建模型

input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
model = Model(input_tensor, output_tensor) #这句话很重要
model.summary()

"""



model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128
)

score, acc = model.evaluate(x_train, y_train)
print(score, acc)

"""