# coding:utf-8
'''
Created on 2018/2/7.

@author: chk01
'''
import tensorflow as tf
from CASIA2.model.config import cfg

# Supervisor的使用
# a = tf.Variable(1)
# b = tf.Variable(2)
# c = tf.add(a, b)
# update = tf.assign(a, c)
# tf.summary.scalar("a", a)
# init_op = tf.initialize_all_variables()
merged_summary_op = tf.summary.merge_all()


# sv = tf.train.Supervisor(logdir="/home/keith/tmp/", init_op=init_op)  # logdir用来保存checkpoint和summary
# saver = sv.saver  # 创建saver
# with sv.managed_session() as sess:  # 会自动去logdir中去找checkpoint，如果没有的话，自动执行初始化
#     for i in range(1000):
#         update_ = sess.run(update)
#         if i % 10 == 0:
#             merged_summary = sess.run(merged_summary_op)
#             sv.summary_computed(sess, merged_summary, global_step=i)
#         if i % 100 == 0:
#             saver.save(sess, logdir="/home/keith/tmp/", global_step=i)
# 总结
#
# 从上面代码可以看出，Supervisor帮助我们处理一些事情
# （1）自动去checkpoint加载数据或初始化数据
# （2）自身有一个Saver，可以用来保存checkpoint
# （3）有一个summary_computed用来保存Summary
# 所以，我们就不需要：
# （1）手动初始化或从checkpoint中加载数据
# （2）不需要创建Saver，使用sv内部的就可以
# （3）不需要创建summary writer

def main(_):
    print(cfg.logdir)
    a = tf.Variable(1, dtype=tf.int32, trainable=True)

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        sess.run(a)
    return 0


if __name__ == '__main__':
    tf.app.run()
