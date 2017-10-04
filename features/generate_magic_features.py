#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-4 下午5:04
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import numpy as np
import pandas as pd
from utils import data_utils
from conf.configure import Configure
from optparse import OptionParser


def generate_common_word_count(df):
    df['q1_q2_common_word_count'] = df.apply(lambda row: len(
        set(str(row['question1'])).intersection(set(str(row['question2'])))
    ), axis=1)
    df['cq1_cq2_common_word_count'] = df.apply(lambda row: len(
        set(str(row['cleaned_question1'])).intersection(set(str(row['cleaned_question2'])))
    ), axis=1)


def main(base_data_dir):
    op_scope = 4
    if os.path.exists(Configure.processed_train_path.format(base_data_dir, op_scope + 1)):
        return

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(base_data_dir, op_scope)
    print("train: {}, test: {}".format(train.shape, test.shape))

    print('---> generate common word count')
    generate_common_word_count(train)
    generate_common_word_count(test)

    print("train: {}, test: {}".format(train.shape, test.shape))
    print("---> save datasets")
    data_utils.save_dataset(base_data_dir, train, test, op_scope + 1)

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "-d", "--base_data_dir",
        dest="base_data_dir",
        default="perform_stem_words",
        help="""base dataset dir: 
                    perform_stem_words, 
                    perform_no_stem_words,
                    full_data_perform_stem_words,
                    full_data_perform_no_stem_words"""
    )

    options, _ = parser.parse_args()
    print("========== generate some magic features ==========")
    main(options.base_data_dir)
