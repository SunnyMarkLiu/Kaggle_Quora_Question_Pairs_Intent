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

import numpy as np
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


def generate_symbol_count(df):
    """
    ? 的数量, 代表一个 question 中包含的问题数目
    """
    df['q1_?_nums'] = df['question1'].map(lambda x: str(x).count('?'))
    df['q2_?_nums'] = df['question2'].map(lambda x: str(x).count('?'))
    df['same_?_num'] = df.apply(lambda raw: int(raw['q1_?_nums'] == raw['q2_?_nums']), axis=1)

    df['q1_math_count'] = df['question1'].map(lambda x: str(x).count('math'))
    df['q2_math_count'] = df['question2'].map(lambda x: str(x).count('math'))

def generate_char_count(df):
    """
    字符出现的次数
    """
    s = 'abcdefghijklmnopqrstuvwxyz'
    def calc_char_count(row):
        q1 = str(row['question1']).strip().lower()
        q2 = str(row['question2']).strip().lower()
        fs1 = [0] * 26
        fs2 = [0] * 26
        for index in range(len(q1)):
            c = q1[index]
            if 0 <= s.find(c):
                fs1[s.find(c)] += 1
        for index in range(len(q2)):
            c = q2[index]
            if 0 <= s.find(c):
                fs2[s.find(c)] += 1
        return fs1, fs2, list(abs(np.array(fs1) - np.array(fs2)))

    df['char_counts'] = df.apply(lambda row: calc_char_count(row), axis=1)
    df['q1_char_count'] = df['char_counts'].map(lambda x: x[0])
    df['q2_char_count'] = df['char_counts'].map(lambda x: x[1])
    df['char_count_diff'] = df['char_counts'].map(lambda x: x[2])

    df['max_q1_char_count'] = df['q1_char_count'].map(lambda x: np.max(x))
    df['min_q1_char_count'] = df['q1_char_count'].map(lambda x: np.min(x))
    df['mean_q1_char_count'] = df['q1_char_count'].map(lambda x: np.mean(x))

    df['max_q2_char_count'] = df['q2_char_count'].map(lambda x: np.max(x))
    df['min_q2_char_count'] = df['q2_char_count'].map(lambda x: np.min(x))
    df['mean_q2_char_count'] = df['q2_char_count'].map(lambda x: np.mean(x))

    df['max_char_count_diff'] = df['char_count_diff'].map(lambda x: np.max(x))
    df['min_char_count_diff'] = df['char_count_diff'].map(lambda x: np.min(x))
    df['mean_char_count_diff'] = df['char_count_diff'].map(lambda x: np.mean(x))

    for i in range(26):
        df['q1_char_{}_count'.format(s[i])] = df['q1_char_count'].map(lambda x: x[i])
        df['q2_char_{}_count'.format(s[i])] = df['q2_char_count'].map(lambda x: x[i])
        df['char_{}_count'.format(s[i])] = df['char_count_diff'].map(lambda x: x[i])

    df.drop(['char_counts', 'q1_char_count', 'q2_char_count', 'char_count_diff'], axis=1, inplace=True)

def main(base_data_dir):
    op_scope = 2
    # if os.path.exists(Configure.processed_train_path.format(base_data_dir, op_scope + 1)):
    #     return

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(base_data_dir, op_scope)
    print("train: {}, test: {}".format(train.shape, test.shape))

    print('---> generate question introducer word features')
    train = generate_question_introducer_word_features(train)
    test = generate_question_introducer_word_features(test)

    print('---> generate symbol features')
    generate_symbol_count(train)
    generate_symbol_count(test)

    print('---> generate char count features')
    generate_char_count(train)
    generate_char_count(test)

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
