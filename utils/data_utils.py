#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-6 下午3:12
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from conf.configure import Configure


def load_dataset(base_data_dir, op_scope):
    train_path = Configure.processed_train_path.format(base_data_dir, op_scope)
    if not os.path.exists(train_path):
        train = pd.read_csv(Configure.original_train_path)

        shuffled_index = np.arange(0, train.shape[0], 1)
        np.random.shuffle(shuffled_index)
        random_indexs = shuffled_index[:int(train.shape[0] * 0.1)]
        train = train.iloc[random_indexs, :]
    else:
        train = pd.read_hdf(train_path, 'data')

    test_path = Configure.processed_test_path.format(base_data_dir, op_scope)
    if not os.path.exists(test_path):
        test = pd.read_csv(Configure.original_test_path)

        shuffled_index = np.arange(0, test.shape[0], 1)
        np.random.shuffle(shuffled_index)
        random_indexs = shuffled_index[:int(test.shape[0] * 0.0001)]
        test = test.iloc[random_indexs, :]
    else:
        test = pd.read_hdf(test_path, 'data')

    return train, test


def save_dataset(base_data_dir, train, test, op_scope):

    if train is not None:
        train_path = Configure.processed_train_path.format(base_data_dir, op_scope)
        train.to_hdf(train_path, 'data', append=True)

    if test is not None:
        test_path = Configure.processed_test_path.format(base_data_dir, op_scope)
        test.to_hdf(test_path, 'data', append=True)
