# @File : imagenet_process.py.py 
# -*- coding: utf-8 -*-
# @Time   : 2023/7/1 11:46 下午 
# @Author : Shijie Zhang
# @Software: PyCharm
import pickle
import os
import numpy as np
import cv2

def get_img_np(data_folder, idx, img_size):
    #data_folder = '/Users/jcyzhang/Downloads/Imagenet32_train_npz'
    #idx = 6
    data_file = os.path.join(data_folder, 'train_data_batch_')
    #img_size=64

    #d = np.load(data_file + str(idx))
    d = np.load(data_file + str(idx)+'.npz')
    x = d['data']
    y = d['labels']
    mean_image = d['mean']
    #print('mean_image', mean_image.shape)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    #x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
    print('x', x.shape)

    return x, y


def save_img(path, base, length):
    root = path #'/Users/jcyzhang/Downloads/Imagenet64'
    for i in range(length):
        name = base+i
        img = x[i].transpose(1,2,0)
        cate = str(y[i])
        path = os.path.join(root, cate)
        os.makedirs(path, exist_ok = True)
        cv2.imwrite(os.path.join(root, cate, '{}.jpg'.format(name)), img)


data_folder = '/home/add_disk1/zhangshijie/contrast_learning/imagenet32/Imagenet32_train_npz/'
base = 0
for idx in range(1,11):
    x, y = get_img_np(data_folder, idx, 32)
    save_img('/home/add_disk1/zhangshijie/contrast_learning/moco/Imagenet32', base, len(x))
    base += len(x)
