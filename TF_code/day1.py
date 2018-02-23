# coding:utf-8
'''
Created on 2018/2/5.

@author: chk01
'''
import tensorflow as tf

g = tf.Graph()
f = tf.Graph()
with g.as_default():
    c = tf.constant(dtype=tf.int32, value=1, name='Const')
    d = tf.multiply(1, 2)
    print(d)
    print(g.as_graph_def())
    g.add_to_collection('a', 1)
    print(g.collections)
    # print(g.finalized)
    # print(g.seed)
    print(g.get_operations())
    print(f.get_all_collection_keys())
    g.finalize()
    print(g.finalized)
    print(c.graph)
    print(c.consumers())
    print(c.op)
    print('c.op', c.op.get_attr('dtype'))
    print(tf.global_variables())
    assert c.graph is g

with f.as_default():
    d = tf.constant(dtype=tf.int32, value=1)
    assert d.graph is f
