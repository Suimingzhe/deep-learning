import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.datasets import mnist
import keras
import numpy as np
import matplotlib.pyplot as plt


a = np.array(1013145465)
print(a)
print(a.dtype)

"""
a = np.array([[[1,2,3],
               [1,3,5]],
              [[8,6,4],
             [9,6,4]],
             [[4,3,6],
              [9,3,4]],
             [[3,5,4],
              [6,4,5]]])

print(a)
print(a.shape)
print(a.ndim)
"""


