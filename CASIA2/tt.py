# coding:utf-8
'''
Created on 2018/1/30.

@author: chk01
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import os.path
import warnings
import scipy.misc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")


def save_filename(root_path, output_filename):
    """
    Input:
        root_path: the root path of dataset
        output_filename: the file_name of saving file, it save under root path default.
    Output:
        None
    Describe:
        get the filename and label of data under the root_path and save it in a file
    """
    folders = os.listdir(root_path)
    with open(os.path.join(root_path, output_filename), "w") as fopen:
        for folder in folders:
            path_folder = os.path.join(root_path, folder)
            for root, sub_folder, files in os.walk(path_folder):
                for file in files:
                    string_write = root_path + "/" + folder + "/" + file + " " + folder + "\n"
                    fopen.write(string_write)


root_path = "data"
output_file = 'train.txt'
save_filename(root_path, output_file)


def get_image_path(save_filename):
    image_path = []
    labels = []
    with open(save_filename, 'r') as fopen:
        for line in fopen:
            image_path.append(line.split(' ')[0])
            labels.append(line.split(' ')[1])
    return image_path, labels


save_file = "data/train.txt"


# image_path, labels = get_image_path(save_file)
# print("path is:\n {};\n\n labels is:\n {}".format(image_path, labels))


def show_picture(image_path):
    number_image = len(image_path)
    file_queue = tf.train.string_input_producer(image_path, shuffle=False)
    image_reader = tf.WholeFileReader()
    key, image = image_reader.read(file_queue)
    image_decode = tf.image.decode_jpeg(image)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        plt.figure()
        for i in range(number_image):
            image_i = sess.run(image_decode)
            plt.subplot(3, 2, i + 1)
            plt.imshow(image_i)
            plt.savefig('1.jpg')
        plt.show()
        coord.request_stop()
        coord.join(threads)

        # 函数中将图像画出，可以看到分别是三张谈笑风声的长者和我爱旅游的儿子。


# show_picture(image_path)


def resize_picutre(image_path):
    number_image = len(image_path)
    file_queue = tf.train.string_input_producer(image_path, shuffle=False)
    image_reader = tf.WholeFileReader()
    key, image = image_reader.read(file_queue)
    image_decode = tf.image.decode_jpeg(image)

    # change  image into gray image
    image_gray = (tf.image.rgb_to_grayscale(image_decode))

    # resize origin_image
    image_resize = tf.image.resize_images(image_gray, [100, 100], method=0)
    image_squeeze = tf.squeeze(image_resize)
    print(image_squeeze)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        plt.figure()
        for i in range(number_image):
            image_i = sess.run(image_squeeze)
            if i < 6:
                plt.subplot(2, 2, i + 1)
                plt.imshow(image_i, cmap='gray')

            new_path = image_path[i].replace("data", "my_data")
            if not os.path.exists(new_path[:new_path.rfind('/')]):
                os.makedirs(new_path[:new_path.rfind('/')])
            scipy.misc.imsave(new_path, image_i)
        plt.show()
        coord.request_stop()
        coord.join(threads)

        # 调整了尺寸之后，儿子还是那么帅。


# resize_picutre(image_path)
# root_path = "my_data"
# output_file = 'train_resize.txt'
# save_filename(root_path, output_file)

# save_file = "my_data/train_resize.txt"
# resize_image_path, resize_labels = get_image_path(save_file)
# print("path is:\n {};\n\n labels is:\n {}".format(resize_image_path, resize_labels))




def rotate_image(image_path, num):
    def random_rotate_image_func(image):
        # 旋转角度范围
        angle = np.random.uniform(low=-1.0, high=1.0)
        return scipy.misc.imrotate(image, angle, 'bicubic')

    number_image = len(image_path)
    print(number_image)
    file_queue = tf.train.string_input_producer(image_path, shuffle=False)
    image_reader = tf.WholeFileReader()
    key, image = image_reader.read(file_queue)
    image_decode = tf.image.decode_jpeg(image, channels=3)

    image_rotate = tf.py_func(random_rotate_image_func, [image_decode], tf.uint8)
    #     # some bug in tensorflow  1.2.1, so you need to upgrade tensorflow version
    #     print('image_decode', image_decode)
    #     image_rotate = tf.py_func(random_rotate_image_func, [image_decode], tf.uint8)
    #     print(image_rotate)
    #
    # image_rotate2 = tf.image.decode_jpeg(image_rotate)
    #
    #     # assert 1 == 0
    # image_resize = tf.image.resize_images(image_rotate, [100, 100], method=0)
    # image_squeeze = tf.squeeze(image_resize)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    #
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        plt.figure()
        for i in range(number_image):
            image_i_list = sess.run(image_rotate)
            new_path = "res/" + image_path[i][3:]
            scipy.misc.imsave(new_path, image_i_list)
            scipy.misc.imsave('{}.jpg'.format(i), image_i_list)
            # for idx, re in enumerate(image_i_list):
            #     with open('my_data/' + str(idx) + '.png', 'wb') as f:
            #         f.write(re)
        coord.request_stop()
        coord.join(threads)

        # 得到的十章图片分别是长者以不同方向谈笑风声。


rotate_num = 2
rotate_image(['my_data/a/1.jpg', 'my_data/a/2.jpg', 'my_data/b/1.jpg', 'my_data/b/2.jpg'], rotate_num)


def brighten_image(image_path, num):
    number_image = len(image_path)
    file_queue = tf.train.string_input_producer(image_path, shuffle=False)
    image_reader = tf.WholeFileReader()
    key, image = image_reader.read(file_queue)
    image_decode = tf.image.decode_jpeg(image, channels=1)

    image_list = []
    for i in range(num):
        # something wrong with tensorflow  1.0.1
        image_bright = tf.image.random_brightness(image_decode, max_delta=0.3)
        image_resize = tf.image.resize_images(image_bright, [100, 100], method=0)
        image_squeeze = tf.squeeze(image_resize)
        image_list.append(image_squeeze)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        plt.figure()
        for i in range(number_image):
            image_i_list = sess.run(image_list)
            for j in range(len(image_i_list)):
                if i == 0:
                    plt.subplot(2, int(num // 2), j + 1)
                    plt.imshow(image_i_list[j], cmap='gray')
                scipy.misc.imsave(image_path[i][:-4] + "_brighten" + str(j) +
                                  image_path[i][-4:], image_i_list[j])
        plt.show()
        coord.request_stop()
        coord.join(threads)


brighten_num = 10


# brighten_image(resize_image_path, brighten_num)


def clip_image(image_path, num, image_size):
    number_image = len(image_path)
    file_queue = tf.train.string_input_producer(image_path, shuffle=False)
    image_reader = tf.WholeFileReader()
    key, image = image_reader.read(file_queue)
    image_decode = tf.image.decode_jpeg(image, channels=1)

    clip = tf.random_uniform([num], 10, 20)

    image_list = []
    for i in range(num):
        clip_value = tf.cast(clip[i], "int32")
        cropped_image = tf.image.crop_to_bounding_box(image_decode, tf.cast(clip[i], "int32"),
                                                      tf.cast(clip[i], "int32"), image_size - clip_value,
                                                      image_size - clip_value)
        image_resize = tf.image.resize_images(cropped_image, [100, 100], method=0)
        image_squeeze = tf.squeeze(image_resize)
        image_list.append(image_squeeze)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        plt.figure()
        for i in range(number_image):
            image_i_list = sess.run(image_list)
            for j in range(len(image_i_list)):
                if i == 0:
                    plt.subplot(2, int(num // 2), j + 1)
                    plt.imshow(image_i_list[j], cmap='gray')
                    # scipy.misc.imsave(image_path[i][:-4] + "_clip" + str(j) +
                    #                   image_path[i][-4:], image_i_list[j])
        plt.show()
        coord.request_stop()
        coord.join(threads)


clip_num = 1
# clip_image(resize_image_path, clip_num, 100)

root_path = "data"
output_file = 'train_transform.txt'


# save_filename(root_path, output_file)
# image_path_transform, labels_transform = get_image_path(root_path + "/" + output_file)


# print(len(image_path_transform))


def calcluate_mean(image_path):
    number_image = len(image_path)
    file_queue = tf.train.string_input_producer(image_path, shuffle=False)
    image_reader = tf.WholeFileReader()
    key, image = image_reader.read(file_queue)
    image_decode = tf.image.decode_jpeg(image)
    image_decode = tf.cast(image_decode, "float32") / 256

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        image_sum = sess.run(image_decode)
        for i in range(number_image - 1):
            image_sum += sess.run(image_decode)
        image_sum /= number_image
        coord.request_stop()
        coord.join(threads)
    return image_sum


image_means = calcluate_mean(['my_data/a/1.jpg', 'my_data/a/2.jpg', 'my_data/b/1.jpg', 'my_data/b/2.jpg'])


# print(image_means.shape)
# print(labels_transform[0:10])
# sample = np.arange(len(labels_transform))
# np.random.shuffle(sample)
# image_path = [image_path_transform[i] for i in sample]
# labels = [labels_transform[i] for i in sample]
# print(labels[0:10])
#
# b = list(set(labels))
# label_dict = dict(zip(b, range(len(b))))
# labels = list(map(lambda x: label_dict.get(x), labels))
# print(label_dict)
# print(labels)

# import pickle
#
# with open("data/label_dict.txt", 'wb') as fopen:
#     pickle.dump(label_dict, fopen)
#
# with open("data/label_dict.txt", 'rb') as fopen:
#     load_label_dict = pickle.load(fopen)
#     print(load_label_dict)


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def save_tfrecord(image_path, save_path, image_means):
#     number_image = len(image_path)
#
#     file_queue = tf.train.string_input_producer(image_path, shuffle=False)
#     image_reader = tf.WholeFileReader()
#     key, image = image_reader.read(file_queue)
#     image_decode = tf.image.decode_jpeg(image)
#     image_decode = tf.cast(image_decode, "float32") / 256
#
#     with tf.Session() as sess:
#         writer = tf.python_io.TFRecordWriter(save_path)
#
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#         for i in range(number_image):
#             image_i = sess.run(image_decode)
#             image_i = image_i - image_means
#             label = int(labels[i])
#             image_raw = image_i.tostring()
#             example = tf.train.Example(features=tf.train.Features(feature={
#                 'label': int64_feature(label),
#                 'image_raw': bytes_feature(image_raw)}))
#             writer.write(example.SerializeToString())
#         writer.close()
#         coord.request_stop()
#         # coord.join(threads)


save_path = "my_data/tfrecord/test.tfrecords"


# save_tfrecord(['my_data/b/1.jpg', 'my_data/a/1.jpg', 'my_data/b/2.jpg', 'my_data/a/2.jpg'], save_path, image_means)


def read_data(tfrecords_file, batch_size, image_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.float32)

    min_after_dequeue = 1000
    image = tf.reshape(image, [image_size, image_size])
    label = tf.cast(img_features['label'], tf.int32)
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=32,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch

# tfrecords_file = "my_data/tfrecord/test.tfrecords"
# batch_size = 10
# image_size = 100
# image_batch_j, label_batch_j = read_data(tfrecords_file, batch_size, image_size)
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     for j in range(100):
#         image_batch_now, label_batch_now = sess.run([image_batch_j, label_batch_j])
#         for i in range(len(image_batch_now)):
#             if j == 0:
#                 plt.figure()
#                 plt.imshow(image_batch_now[i], cmap='gray')
#                 plt.show()
#     coord.request_stop()
#     coord.join(threads)
