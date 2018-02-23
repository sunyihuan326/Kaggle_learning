# coding:utf-8
'''
Created on 2018/1/30.

@author: chk01
'''
import struct
import os
from PIL import Image

count = 0
path = "F:/dataSets/CASIA/HWDB1_tst_gnt"
file_list = os.listdir(path)
for file in file_list:
    f = open(path + '/' + file, 'rb')
    while f.read(1) != "":
        f.seek(-1, 1)
        global count
        count += 1
        try:
            length_bytes = struct.unpack('<I', f.read(4))[0]
            # print(length_bytes)

            tag_code = f.read(2).decode(encoding="gbk")
            print('tag_code', tag_code)
            width = struct.unpack('<H', f.read(2))[0]
            print(width)
            height = struct.unpack('<H', f.read(2))[0]
            print(height)

            im = Image.new('RGB', (width, height))
            img_array = im.load()
            # print img_array[0,7]
            for x in range(0, height):
                for y in range(0, width):
                    pixel = struct.unpack('<B', f.read(1))[0]
                    img_array[y, x] = (pixel, pixel, pixel)

            fdir = path + '/' + tag_code
            if not os.path.exists(fdir):
                os.makedirs(fdir)

            filename = fdir + '/' + str(count) + '.png'
            im.save(filename)
        except Exception as e:
            print(e)
    f.close()
print(count)
