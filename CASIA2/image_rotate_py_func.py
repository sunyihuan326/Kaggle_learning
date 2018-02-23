# coding:utf-8
'''
Created on 2018/2/1.

@author: chk01
'''
import tensorflow as tf
import numpy as np
import scipy.misc


def random_rotate_image(image_file, num):
    with tf.Graph().as_default():
        tf.set_random_seed(666)
        file_contents = tf.read_file(image_file)
        image = tf.image.decode_image(file_contents, channels=3)
        image_rotate_en_list = []

        def random_rotate_image_func(image):
            # 旋转角度范围
            angle = np.random.uniform(low=-30.0, high=30.0)
            return scipy.misc.imrotate(image, angle, 'bicubic')

        for i in range(num):
            image_rotate = tf.py_func(random_rotate_image_func, [image], tf.uint8)
            image_rotate_en_list.append(tf.image.encode_png(image_rotate))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())
            results = sess.run(image_rotate_en_list)
            for idx, re in enumerate(results):
                with open('data/' + str(idx) + '.png', 'wb') as f:
                    f.write(re)


random_rotate_image('my_data/a/1.jpg', 20)
