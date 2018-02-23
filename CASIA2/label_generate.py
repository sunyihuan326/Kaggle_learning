# coding:utf-8
'''
Created on 2018/2/3.

@author: chk01
'''

import pickle

with open('F:/dataSets/CASIA/HWDB1/char_dict', 'rb') as fp:
    chinese2class = pickle.load(fp, encoding='ASCII')
    class2chinese = {}
    for key, item in chinese2class.items():
        num = len(str(item))
        class2chinese.update({'0' * (5 - num) + str(item): key})
with open('class2label', 'rb') as fp2:
    class2label = pickle.load(fp2, encoding='ASCII')
    label2class = {}
    label2chinese = {}
    for key, item in class2label.items():
        label2class.update({item: key})
        label2chinese.update({item: class2chinese[key]})


def num2Chinese():
    print(chinese2class)
    print(class2chinese)
    print(class2label)
    print(label2class)
    print(label2chinese)

    f = open('label2class', 'wb')
    pickle.dump(label2class, f)
    f.close()


# a=scio.loadmat('chinese2classes')['data']
# print(a)
num2Chinese()
