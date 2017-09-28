#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-28 下午10:10
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from conf.configure import Configure
from utils import data_utils

from optparse import OptionParser


def main(base_data_dir):
    op_scope = 1
    if os.path.exists(Configure.processed_train_path.format(base_data_dir, op_scope + 1)):
        return

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(base_data_dir, op_scope)
    print("train: {}, test: {}".format(train.shape, test.shape))

    print("train: {}, test: {}".format(train.shape, test.shape))
    print("---> save datasets")
    data_utils.save_dataset(base_data_dir, train, test, op_scope + 1)


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "-d", "--base_data_dir",
        dest="base_data_dir",
        default="stop_words_and_stem_words",
        help="""base dataset dir: 
                    stop_words_and_stem_words, 
                    stop_words_and_no_stem_words, 
                    no_stop_words_and_stem_words, 
                    no_stop_words_and_no_stem_words"""
    )

    options, _ = parser.parse_args()
    print("========== perform data cleaning and basic statistic feature engineering ==========")
    main(options.base_data_dir)
