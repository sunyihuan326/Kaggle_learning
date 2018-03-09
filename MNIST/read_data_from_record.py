# coding:utf-8 
'''
created on 2018/2/26

@author:Dxq
'''
import tensorflow as tf
import matplotlib.pyplot as plt

record_file = 'C:/Users/chk01/Desktop/mnist_data/test_0shifted_mnist.tfrecords'
file_queue = tf.train.string_input_producer([record_file], shuffle=True, capacity=2000)

reader = tf.TFRecordReader()
key, _value = reader.read(file_queue)

img_feature = tf.parse_single_example(
    _value, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature(shape=[], dtype=tf.string)
    }
)
image = tf.decode_raw(img_feature['image_raw'], tf.uint8)
image = tf.reshape(image, [28, 28])
label = tf.cast(img_feature['label'], tf.int32)

batch_x, batch_y = tf.train.batch([image, label], batch_size=128, num_threads=30, capacity=2000)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(100000):
        _x, _y = sess.run([batch_x, batch_y])

        print(_x[0].shape)
        plt.imshow(_x[0])
        plt.show()
        print(type(_x))
        print(_y)
    coord.request_stop()
    coord.join(threads)
# with tf.device('/gpu:0')
