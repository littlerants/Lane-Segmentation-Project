# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:10:09 2020

@author: Administrator
"""
import numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle
'''
label_list = []
image_list = []

image_dir = 'E:\\数据集\\车道线数据集\\Imagedata\\Image_data\\'
label_dir = 'E:\\数据集\\车道线数据集\\Imagedata\\Gray_Label\\'
#print(os.listdir(image_dir))
i = 1
for s1 in os.listdir(image_dir):
    image_sub_dir1 = os.path.join(image_dir, s1)
    label_sub_dir1 = os.path.join(label_dir, s1.replace('ColorImage', 'Label'))
    #print(image_sub_dir1, label_sub_dir1)


    for s2 in os.listdir(image_sub_dir1):
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        label_sub_dir2 = os.path.join(label_sub_dir1, 'Label')
        #print(image_sub_dir2, label_sub_dir2)

        for s3 in os.listdir(image_sub_dir2):
            image_sub_dir3 = os.path.join(image_sub_dir2, s3)
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)
            #print(image_sub_dir3, '--------', label_sub_dir3)

            for s4 in os.listdir(image_sub_dir3):
                image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                label_sub_dir4 = os.path.join(label_sub_dir3, s4)
                #print(image_sub_dir4, '--------', label_sub_dir4)

                for s5 in os.listdir(image_sub_dir4):
                    image_sub_dir5 = os.path.join(image_sub_dir4, s5)
                    label_sub_dir5 = os.path.join(label_sub_dir4, s5.replace('.jpg', '_bin.png'))
                    #print(image_sub_dir5, '-----', label_sub_dir5)

                    if not os.path.exists(image_sub_dir5):
                        print('hhh', image_sub_dir5)
                    if not os.path.exists(label_sub_dir5):
                        print('hhh', label_sub_dir5)
                    i = i + 1
                    #print(i)
                    image_list.append(image_sub_dir5)
                    label_list.append(label_sub_dir5)

print(len(image_list))
print(len(label_list))


save = pd.DataFrame({'image_list': image_list, 'label_list': label_list})
save_shuffle = shuffle(save)
print(save_shuffle)

#print(save_shuffle)
save_shuffle.to_csv('..\\data_list\\train.csv', index=False)
'''
data = pd.read_csv('../data_list/full.csv')
#print(pd)

import sklearn as sk
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.95, random_state=1000)

train_real, val = train_test_split(train, test_size=0.03, random_state=100)



print(len(train_real))
print(len(val))
print(len(test))

train_real.to_csv('..//data_list//train_1.csv', index=False)
val.to_csv('..//data_list//val_1.csv', index=False)
test.to_csv('..//data_list//test_1.csv', index=False)
print('成功运行.......')








