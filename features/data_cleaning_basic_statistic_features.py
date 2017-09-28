#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-28 上午11:38
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from utils import data_utils


def main():
    op_scope = 0
    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(op_scope)
    print("train: {}, test: {}".format(train.shape, test.shape))
    print("---> generate basic statistic features")
    train['num_of_chars_q1'] = train['question1'].apply(lambda x: len(str(x)))
    train['num_of_chars_q2'] = train['question2'].apply(lambda x: len(str(x)))
    test['num_of_chars_q1'] = test['question1'].apply(lambda x: len(str(x)))
    test['num_of_chars_q2'] = test['question2'].apply(lambda x: len(str(x)))

    train['num_of_words_q1'] = train['question1'].apply(lambda x: len(str(x).split()))
    train['num_of_words_q2'] = train['question2'].apply(lambda x: len(str(x).split()))
    test['num_of_words_q1'] = test['question1'].apply(lambda x: len(str(x).split()))
    test['num_of_words_q2'] = test['question2'].apply(lambda x: len(str(x).split()))

    print("train: {}, test: {}".format(train.shape, test.shape))
    print("---> save datasets")
    data_utils.save_dataset(train, test, op_scope + 1)


if __name__ == "__main__":
    print("========== perform data cleaning and basic statistic feature engineering ==========")
    main()
