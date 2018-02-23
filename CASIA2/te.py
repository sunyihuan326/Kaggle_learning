# coding:utf-8
'''
Created on 2018/2/3.

@author: chk01
'''
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy
image = Image.open("1.jpg")
op = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
op_stand = tf.image.per_image_standardization(op)
with tf.Session() as sess:
    a, b = sess.run([op, op_stand])
    print(a)
    print(b)
