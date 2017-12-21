
# coding:utf-8
'''
Created on 2017/12/21.

@author: chk01
'''
import tensorflow as tf


ckpt = tf.train.get_checkpoint_state('save/')
print(ckpt.model_checkpoint_path)
saver = tf.train.import_meta_graph("save/model.ckpt.meta")
print(saver)
# value
a = tf.train.NewCheckpointReader('save/model.ckpt.index')
b = tf.get_default_graph().get_operations()[0].name
print(tf.get_default_graph().get_tensor_by_name("Placeholder:0"))
print(b)


# .meta文件保存了当前图结构
#
# .index文件保存了当前参数名
#
# .data文件保存了当前参数值