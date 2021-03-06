# coding:utf-8
'''
Created on 2017/12/21.

@author: chk01
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

a = np.array([[1, 2], [3, 4]])
sum0 = np.sum(a, axis=0)
sum1 = np.sum(a, axis=1)

print(sum0, sum1)
assert 1 == 0

print(plt.colors())


def myfind(x, y):
    return [a for a in range(len(y)) if y[a] == x]


a = [1, 2, 3, 4, 4, 3, 2, 1]
print(myfind(1, a))
x = np.random.rand(50, 30) - .5

# basic
f1 = plt.figure(1)

plt.subplot(211)
plt.scatter(x[:, 1], x[:, 0])

# with label
plt.subplot(212)
label = list(np.ones(20)) + list(2 * np.ones(15)) + list(3 * np.ones(15))
label = np.array(label)
print(label)
print(label.shape)
# cmap=plt.cm.Spectral
plt.scatter(x[:, 1], x[:, 0], s=15.0 * label, c=label)
# plt.show()

# with legend
f2 = plt.figure(2)
idx_1 = range(20)
p1 = plt.scatter(x[idx_1, 1], x[idx_1, 0], c='m', label=str(1), s=30)
idx_2 = range(20, 35)
p2 = plt.scatter(x[idx_2, 1], x[idx_2, 0], marker='+', c='c', label='2', s=50)
idx_3 = range(35, 50)
p3 = plt.scatter(x[idx_3, 1], x[idx_3, 0], marker='o', c='r', label='3', s=15)
# plt.legend(loc='upper right')
plt.show()
assert 1 == 0
ckpt = tf.train.get_checkpoint_state('save/')
print(ckpt.model_checkpoint_path)
saver = tf.train.import_meta_graph("save/model.ckpt.meta")
print(saver)
# value
a = tf.train.NewCheckpointReader('save/model.ckpt.index')
b = tf.get_default_graph().get_operations()[0].name
print(tf.get_default_graph().get_tensor_by_name("Placeholder:0"))
print(b)

tf.contrib.layers.l2_regularizer()
tf.train.AdamOptimizer()
tf.nn.softmax_cross_entropy_with_logits()
tf.nn.sigmoid_cross_entropy_with_logits()
tf.gather()


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


# 损失函数定义
with tf.variable_scope('loss_scope'):
    self.centerloss, self.centers_update_op = get_center_loss(self.features, self.y, 0.5, self.CLASSNUM)
    # self.loss = tf.losses.softmax_cross_entropy(onehot_labels=util.makeonehot(self.y, self.CLASSNUM), logits=self.score)
    self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.score) + 0.05 * self.centerloss
    # tf.summary.scalar('loss',self.loss)
    # 优化器
with tf.control_dependencies([self.centers_update_op]):
    self.train_op = tf.train.MomentumOptimizer(0.001, 0.9).minimize(self.loss)


    # .meta文件保存了当前图结构
    #
    # .index文件保存了当前参数名
    #
    # .data文件保存了当前参数值
