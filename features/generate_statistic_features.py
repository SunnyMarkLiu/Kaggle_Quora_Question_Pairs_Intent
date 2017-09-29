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

from utils import data_utils

from optparse import OptionParser


def generate_contain_word_features(df, word):
    """
    是否包含特定的单词, 如 not
    """
    df[word + "_count_cq1"] = df['cleaned_question1'].apply(lambda x: str(x).strip().split().count(word))
    df[word + "_count_cq2"] = df['cleaned_question2'].apply(lambda x: str(x).strip().split().count(word))

    df[word + "_cq1_cq2_both_gt_0"] = df.apply(lambda raw: ((raw[word + "_count_cq1"] > 0) & (raw[word + "_count_cq2"] > 0)), axis=1)
    df[word + "_cq1_cq2_one_gt_0"] = df.apply(lambda raw: ((raw[word + "_count_cq1"] > 0) | (raw[word + "_count_cq2"] > 0)), axis=1)
    df[word + "_cq1_cq2_diff"] = df.apply(lambda raw: (((raw[word + "_count_cq1"] > 0) & (raw[word + "_count_cq2"] <= 0)) |
                                                      ((raw[word + "_count_cq1"] <= 0) & (raw[word + "_count_cq2"] > 0))), axis=1)

def main(base_data_dir):
    op_scope = 1
    # if os.path.exists(Configure.processed_train_path.format(base_data_dir, op_scope + 1)):
    #     return

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(base_data_dir, op_scope)
    print("train: {}, test: {}".format(train.shape, test.shape))

    print('---> generate contain_word count features')
    generate_contain_word_features(train, 'not')
    generate_contain_word_features(test, 'not')
    generate_contain_word_features(train, 'best')
    generate_contain_word_features(test, 'best')

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
                    perform_no_stem_words"""
    )

    options, _ = parser.parse_args()
    print("========== generate some statistic features ==========")
    main(options.base_data_dir)
