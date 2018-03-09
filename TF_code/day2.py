# coding:utf-8
'''
Created on 2018/2/6.

@author: chk01
'''
import tensorflow as tf
import numpy as np

# with tf.Graph().as_default() as g:
#     x = tf.constant(1, dtype=tf.int32)
#     y = tf.Variable([1.0, 1.0], dtype=tf.float32)
#     init = tf.global_variables_initializer()
#     with tf.Session(graph=g) as sess:
#         sess.run(init)
#         print(sess.run(x))
#     pass
# # tf_record_file的生成
# # 1、需要writer
# writer = tf.python_io.TFRecordWriter(path='path_for_record_to_save2')
# # # 数据
# _label = 1  # 真实标签
# _raw_data = np.array([[1, 2], [2, 3]])  # 相对应的数据
# #
# label = tf.train.Feature(int64_list=tf.train.Int64List(value=[_label]))
# raw_data = tf.train.Feature(bytes_list=tf.train.BytesList(value=[_raw_data.tobytes()]))
# #
# features = tf.train.Features(feature={'label': label, 'raw_data': raw_data})
# example = tf.train.Example(features=features)
# record = example.SerializeToString()  # 序列化为str
# # # 写入以及关闭
# for i in range(100):
#     writer.write(record=record)  # record 要求为str
# writer.close()
tf_record_file = 'F:/dataSets/CASIA/HWDB1/train.tfrecord'
capacity = 2000
# capacity 32 must be bigger than min_after_dequeue 100.
file_queue = tf.train.string_input_producer([tf_record_file], shuffle=True, capacity=capacity,num_epochs=1)
# print(tf.GraphKeys.QUEUE_RUNNERS)  # queue_runners
# print(tf.get_default_graph().get_collection(name='queue_runners'))

reader = tf.TFRecordReader()
key, _value = reader.read(file_queue)
# print(key, value)

img_features = tf.parse_single_example(
    _value,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })
image = tf.decode_raw(img_features['image_raw'], tf.float32)
image = tf.reshape(image, [96, 96])
label = tf.cast(img_features['label'], tf.int32)

x_batch, y_batch = tf.train.shuffle_batch([image, label], batch_size=32, num_threads=2, capacity=250,
                                          min_after_dequeue=200)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(threads)
    for i in range(200):
        print(i)
        # print(sess.run(y_batch))
    coord.request_stop()
    coord.join(threads)
    # if i ==99:
    #     coord.request_stop()
    # label = tf.cast(img_features['label'], tf.int32)

    # print(x_batch)
    # coord = tf.train
