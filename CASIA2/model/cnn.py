# coding:utf-8
'''
Created on 2017/12/19

@author: sunyihuan
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy.io as scio


def one_hot(y, classes):
    # m, _ = y.reshape(-1, 1).shape
    return np.eye(classes)[y]


def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.div(tf.nn.l2_loss(features - centers_batch), int(len_features))
    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features
    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)
    return loss, centers_update_op


def model(lr=0.01, epoches=200, minibatch_size=64, drop_prob=.2):
    X = tf.placeholder(tf.float32, shape=[None, 96, 96])
    XX = tf.reshape(X, shape=[-1, 96, 96, 1])
    Y = tf.placeholder(tf.int32, shape=[None, ])
    YY = tf.one_hot(Y, 3755, on_value=1, off_value=None, axis=1)

    dp = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    reg1 = tf.contrib.layers.l2_regularizer(scale=0.1)
    conv1 = tf.layers.conv2d(XX, 32, 5, padding='same', activation=tf.nn.relu, kernel_regularizer=reg1)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same')

    conv2 = tf.layers.conv2d(conv1, 64, 3, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same')

    # conv3 = tf.layers.conv2d(conv2, 128, 3, padding='same', activation=tf.nn.relu)
    # conv3 = tf.layers.average_pooling2d(conv3, 2, 2, padding='same')

    # convZ = tf.layers.flatten(pool3)
    convZ = tf.contrib.layers.flatten(conv2)

    fc1 = tf.layers.dense(convZ, 256, activation=tf.nn.relu)
    # fc1 = tf.layers.batch_normalization(fc1)
    # fc1 = tf.layers.dropout(fc1, rate=dp, training=True)
    #
    fc2 = tf.layers.dense(fc1, 512, activation=tf.nn.relu)
    # fc2 = tf.layers.batch_normalization(fc2)
    # fc2 = tf.layers.dropout(fc2, rate=dp, training=True)

    # fc3 = tf.layers.dense(fc2, 2, activation=None, name='fc3')

    # fc3_out = tf.nn.relu(fc3)
    # fc3 = tf.layers.batch_normalization(fc3)
    # fc3 = tf.layers.dropout(fc3, rate=dp, training=True)
    ZL = tf.layers.dense(fc2, 3755, activation=None)

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ZL, labels=Y))

    learning_rate = tf.train.exponential_decay(lr,
                                               global_step=global_step,
                                               decay_steps=10, decay_rate=0.9)
    learning_rate = tf.maximum(learning_rate, .001)

    # with tf.variable_scope('loss_scope'):
    #     centerloss, centers_update_op = get_center_loss(fc2, Y, 0.5, 10)
    #     # self.loss = tf.losses.softmax_cross_entropy(onehot_labels=util.makeonehot(self.y, self.CLASSNUM), logits=self.score)
    #     # lambda则0.1-0.0001之间不等
    loss = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=ZL)
    # with tf.control_dependencies([]):
    # train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    predict_op = tf.argmax(ZL, 1, name='predict')

    correct_prediction = tf.equal(predict_op, tf.argmax(YY, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    add_global = global_step.assign_add(1)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for epoch in range(epoches):
            image_batch_now, label_batch_now = sess.run([img, label])
            __, _loss, _ = sess.run([add_global, loss, train_op],
                                    feed_dict={X: image_batch_now, Y: label_batch_now, dp: drop_prob})
            print(_loss)

        coord.request_stop()
        coord.join(threads)
    print(1)
    #     if epoch % 5 == 0:
    #         train_accuracy = accuracy.eval({X: trX[:2000], Y: trY[:2000], dp: 0.0})
    #         test_accuracy = accuracy.eval({X: teX[:2000], Y: teY[:2000], dp: 0.0})
    #         print("Cost after epoch %i: %f tr-acc: %f te-acc: %f" % (epoch, _loss, train_accuracy, test_accuracy))
    # train_accuracy = accuracy.eval({X: trX[:2000], Y: trY[:2000], dp: 0.0})
    # test_accuracy = accuracy.eval({X: teX[:2000], Y: teY[:2000], dp: 0.0})
    #
    # # 修改网络倒数层为2，然后输出特征
    # # _fc3 = fc3.eval({X: teX[:2000], Y: teY[:2000], dp: 0.0})
    # # plt.scatter(_fc3[:, 0], _fc3[:, 1], c=teY[:2000])
    # # plt.show()
    # print("Train Accuracy:", train_accuracy)
    # print("Test Accuracy:", test_accuracy)
    # saver.save(sess, "save/model.ckpt")


def read(path):
    batch_size = 64
    image_size = 96

    filename_queue = tf.train.string_input_producer([path], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.float32)

    min_after_dequeue = 10000
    image = tf.reshape(image, [image_size, image_size])
    label = tf.cast(img_features['label'], tf.int32)
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=3,
                                                      capacity=capacity,
                                                      # allow_smaller_final_batch=True,
                                                      min_after_dequeue=min_after_dequeue
                                                      )
    return image_batch, label_batch


tfrecord_file = 'F:/dataSets/CASIA/HWDB1/train.tfrecord'
img, label = read(tfrecord_file)
model()
