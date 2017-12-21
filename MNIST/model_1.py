# coding:utf-8
'''
Created on 2017/12/19

@author: sunyihuan
'''
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

train_dir = 'C:/Users/syh03/Desktop/Kaggle/MNIST/data/train.csv'
test_dir = 'C:/Users/syh03/Desktop/Kaggle/MNIST/data/test.csv'

# read_data
data = pd.read_csv(train_dir)
X_data = np.array(data.iloc[:, 1:].values, dtype=np.float32)
Y_data = np.array(data.iloc[:, 0].values, dtype=np.int32)

pre_data = pd.read_csv(test_dir)
preX = np.array(pre_data.values, dtype=np.float32)
print(preX.shape)


# show data
# X_data = np.reshape(X_data, (-1, 28, 28))
def show_data(X, Y):
    for i in range(1, 10):
        plt.subplot(330 + i)
        plt.imshow(X[i], cmap=plt.get_cmap('gray'))
        plt.title(Y[i])
    plt.show()


def one_hot(y, classes):
    m, _ = y.reshape(-1, 1).shape
    return np.eye(classes)[y].reshape(m, -1)


Y_data = one_hot(Y_data, 10)
trX, teX, trY, teY = train_test_split(X_data, Y_data, test_size=.2, shuffle=True)
print(trX.shape)
print(teX.shape)


def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def model(trX, trY, teX, teY, lr=0.5, epoches=200, minibatch_size=64, keep_prob=.8):
    X = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    XX = tf.reshape(X, shape=[-1, 28, 28, 1])
    Y = tf.placeholder(tf.int32, shape=[None, 10])
    kp = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    conv1 = tf.layers.conv2d(XX, 32, 5, strides=(2, 2), padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same')

    conv2 = tf.layers.conv2d(pool1, 64, 5, strides=(2, 2), padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same')

    conv3 = tf.layers.conv2d(pool2, 128, 5, strides=(2, 2), padding='same', activation=tf.nn.relu)
    pool3 = tf.layers.average_pooling2d(conv3, 2, 2, padding='same')

    convZ = tf.layers.flatten(pool3)

    fc1 = tf.layers.dense(convZ, 256, activation=tf.nn.relu)
    fc1 = tf.layers.batch_normalization(fc1)
    fc1 = tf.layers.dropout(fc1, rate=kp, training=True)

    fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu)
    fc2 = tf.layers.batch_normalization(fc2)
    fc2 = tf.layers.dropout(fc2, rate=kp, training=True)

    ZL = tf.layers.dense(fc2, 10, activation=tf.nn.softmax)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ZL, labels=Y))

    learning_rate = tf.train.exponential_decay(lr,
                                               global_step=global_step,
                                               decay_steps=10, decay_rate=0.9)
    learning_rate = tf.maximum(learning_rate, .0001)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    predict_op = tf.argmax(ZL, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    add_global = global_step.assign_add(1)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoches):
            minibatches = random_mini_batches(trX, trY, minibatch_size)

            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                __, _loss, _ = sess.run([add_global, loss, train_op],
                                        feed_dict={X: minibatch_X, Y: minibatch_Y, kp: keep_prob})
            if epoch % 5 == 0:
                train_accuracy = accuracy.eval({X: trX, Y: trY, kp: 1.0})
                test_accuracy = accuracy.eval({X: teX, Y: teY, kp: 1.0})
                print("Cost after epoch %i: %f tr-acc: %f te-acc: %f" % (epoch, _loss, train_accuracy, test_accuracy))
        train_accuracy = accuracy.eval({X: trX, Y: trY, kp: 1.0})
        test_accuracy = accuracy.eval({X: teX, Y: teY, kp: 1.0})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        prediction = predict_op.eval({X: preX, kp: 1.0})
        result = pd.read_csv('C:/Users/syh03/Desktop/Kaggle/MNIST/data/sample_submission.csv')
        result['Label'] = prediction
        result.to_csv('C:/Users/syh03/Desktop/Kaggle/MNIST/data/result.csv')


def data_check(data):
    res = list(np.argmax(data, 1))
    num = len(res)
    classes = data.shape[1]
    for i in range(classes):
        print(str(i) + '的比例', round(100.0 * res.count(i) / num, 2), '%')
    print('<------------------分割线---------------------->')


data_check(trY)
data_check(teY)
model(trX, trY, teX, teY, epoches=1000)
