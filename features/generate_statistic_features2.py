#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-3 上午11:44
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from utils import data_utils
from conf.configure import Configure
from optparse import OptionParser


def generate_question_introducer_word_features(df):
    """
    疑问句引导词, how, which, what...
    """
    df['q1_contain_how'] = df['question1'].map(lambda x: int('how' in str(x).lower()))
    df['q2_contain_how'] = df['question2'].map(lambda x: int('how' in str(x).lower()))
    df['q1_contain_which'] = df['question1'].map(lambda x: int('which' in str(x).lower()))
    df['q2_contain_which'] = df['question2'].map(lambda x: int('which' in str(x).lower()))
    df['q1_contain_what'] = df['question1'].map(lambda x: int('what' in str(x).lower()))
    df['q2_contain_what'] = df['question2'].map(lambda x: int('what' in str(x).lower()))
    df['q1_contain_why'] = df['question1'].map(lambda x: int('why' in str(x).lower()))
    df['q2_contain_why'] = df['question2'].map(lambda x: int('why' in str(x).lower()))
    df['q1_contain_when'] = df['question1'].map(lambda x: int('when' in str(x).lower()))
    df['q2_contain_when'] = df['question2'].map(lambda x: int('when' in str(x).lower()))
    df['q1_contain_who'] = df['question1'].map(lambda x: int('who' in str(x).lower()))
    df['q2_contain_who'] = df['question2'].map(lambda x: int('who' in str(x).lower()))
    df['q1_contain_where'] = df['question1'].map(lambda x: int('where' in str(x).lower()))
    df['q2_contain_where'] = df['question2'].map(lambda x: int('where' in str(x).lower()))

    df['both_contain_how'] = df.apply(lambda raw: raw['q1_contain_how'] * raw['q2_contain_how'], axis=1)
    df['both_contain_which'] = df.apply(lambda raw: raw['q1_contain_which'] * raw['q2_contain_which'], axis=1)
    df['both_contain_what'] = df.apply(lambda raw: raw['q1_contain_what'] * raw['q2_contain_what'], axis=1)
    df['both_contain_why'] = df.apply(lambda raw: raw['q1_contain_why'] * raw['q2_contain_why'], axis=1)
    df['both_contain_when'] = df.apply(lambda raw: raw['q1_contain_when'] * raw['q2_contain_when'], axis=1)
    df['both_contain_who'] = df.apply(lambda raw: raw['q1_contain_who'] * raw['q2_contain_who'], axis=1)
    df['both_contain_where'] = df.apply(lambda raw: raw['q1_contain_where'] * raw['q1_contain_where'], axis=1)

    return df

def generate_question_symbol_count(df):
    """
    ? 的数量, 代表一个 question 中包含的问题数目
    """
    df['q1_?_nums'] = df['question1'].map(lambda x: str(x).count('?'))
    df['q2_?_nums'] = df['question2'].map(lambda x: str(x).count('?'))
    df['same_?_num'] = df.apply(lambda raw: int(raw['q1_?_nums'] == raw['q2_?_nums']), axis=1)


def main(base_data_dir):
    op_scope = 2
    # if os.path.exists(Configure.processed_train_path.format(base_data_dir, op_scope + 1)):
    #     return

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(base_data_dir, op_scope)
    print("train: {}, test: {}".format(train.shape, test.shape))

    train = generate_question_introducer_word_features(train)
    test = generate_question_introducer_word_features(test)

    generate_question_symbol_count(train)
    generate_question_symbol_count(test)

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
    print("========== generate some statistic features 2 ==========")
    main(options.base_data_dir)
