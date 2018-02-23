# coding:utf-8
'''
Created on 2018/1/31.

@author: chk01
'''
import tensorflow as tf


def pre_process(images):
    if FLAGS.random_flip_up_down:
        images = tf.image.random_flip_up_down(images)
    if FLAGS.random_flip_left_right:
        images = tf.image.random_flip_left_right(images)
    if FLAGS.random_brightness:
        images = tf.image.random_brightness(images, max_delta=0.3)
    if FLAGS.random_contrast:
        images = tf.image.random_contrast(images, 0.8, 1.2)
    new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
    images = tf.image.resize_images(images, new_size)
    return images


def batch_data(file_labels, sess, batch_size=128):
    image_list = [file_label[0] for file_label in file_labels]
    label_list = [int(file_label[1]) for file_label in file_labels]
    print('tag2 {0}'.format(len(image_list)))
    images_tensor = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(label_list, dtype=tf.int64)
    input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor])

    labels = input_queue[1]
    images_content = tf.read_file(input_queue[0])
    # images = tf.image.decode_png(images_content, channels=1)
    images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
    # images = images / 256
    images = pre_process(images)
    # print images.get_shape()
    # one hot
    labels = tf.one_hot(labels, 3755)
    image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                      min_after_dequeue=10000)
    # print 'image_batch', image_batch.get_shape()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return image_batch, label_batch, coord, threads


def show_data(self, samples_per_class=10, fig_path='data_fig'):
    number_image = len(self.images_path) // len(self.classes)
    samples_per_class = min(samples_per_class, number_image)
    file_queue = tf.train.string_input_producer(self.images_path, shuffle=False)
    image_reader = tf.WholeFileReader()
    key, image = image_reader.read(file_queue)
    image_decode = tf.image.decode_jpeg(image)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        plt.figure()
        for y, cls in enumerate(self.classes):
            idxs = np.flatnonzero(np.array(self.labels) == cls)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            print(idxs)
            for i, idx in enumerate(idxs):
                image_i = sess.run(image_decode)
                plt_idx = i * len(self.classes) + y + 1
                print(plt_idx)
                print(cls)
                plt.subplot(samples_per_class, len(self.classes), plt_idx)
                plt.imshow(image_i)
                if i == 0:
                    plt.title(cls)
                plt.axis('equal')
        coord.request_stop()
        coord.join(threads)

        plt.savefig(fig_path)
    return fig_path
