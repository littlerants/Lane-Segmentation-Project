# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 20:10:30 2019

@author: Administrator
"""

# 训练模型参数配置

params = dict()
params['num_class'] = 8
params['epochs'] = 8
params['batch_size'] = 32
params['learning_rate'] = 1e-3
params['pretrained'] = True

# 文件名配置
params['train_csv'] = 'train.csv'
params['val_csv'] = 'val.csv'
params['test_csv'] = 'test.csv'

# 路径配置
params['root_dir'] = './data_list/'
params['model_save_path'] = './ModelSaveDir/'

# 网络参数配置--deeplabv3plus
params['resnet18'] = [2, 2, 2, 2]
params['resnet34'] = [3, 4, 6, 3]
params['resnet50'] = [3, 4, 6, 3]
params['resnet101'] = [3, 4, 23, 3]
params['resnet152'] = [3, 8, 36, 3]

params['16os'] = [2, 2, 1]
params['8os'] = [2, 1, 1]

params['atrous_rate_list'] = [1, 6, 12, 18]
params['ASPP_OUTDIM'] = 256

