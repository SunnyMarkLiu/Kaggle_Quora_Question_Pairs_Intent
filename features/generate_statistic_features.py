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

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import data_utils

from optparse import OptionParser


def generate_contain_word_features(df, word):
    """
    是否包含特定的单词, 如 not
    """
    df[word + "_count_cq1"] = df['cleaned_question1'].apply(lambda x: str(x).strip().split().count(word))
    df[word + "_count_cq2"] = df['cleaned_question2'].apply(lambda x: str(x).strip().split().count(word))

    df[word + "_cq1_cq2_both_gt_0"] = df.apply(lambda raw: int((raw[word + "_count_cq1"] > 0) & (raw[word + "_count_cq2"] > 0)), axis=1)
    df[word + "_cq1_cq2_one_gt_0"] = df.apply(lambda raw: int((raw[word + "_count_cq1"] > 0) | (raw[word + "_count_cq2"] > 0)), axis=1)
    df[word + "_cq1_cq2_diff"] = df.apply(lambda raw: int(((raw[word + "_count_cq1"] > 0) & (raw[word + "_count_cq2"] <= 0)) |
                                                      ((raw[word + "_count_cq1"] <= 0) & (raw[word + "_count_cq2"] > 0))), axis=1)


def generate_matchshared_words_features(row, question):
    """
    两个句子中都存在的词汇总数
    """
    q1words = {}
    q2words = {}
    for word in str(row[question+"1"]).lower().split():
        q1words[word] = q1words.get(word, 0) + 1
    for word in str(row[question+"2"]).lower().split():
        q2words[word] = q2words.get(word, 0) + 1
    n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
    n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
    n_tol = sum(q1words.values()) + sum(q2words.values())
    if 1e-6 > n_tol:
        return 0.
    else:
        return 1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol


def train_tfidf_vectorizer(train, test, question):
    tfidf = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_txt = pd.Series(
                train[question+'1'].tolist() + train[question+'2'].tolist() +
                test[question+'1'].tolist() + test[question+'2'].tolist()).astype(str)
    tfidf.fit(tfidf_txt)
    return tfidf


def generate_tfidf_features(tfidf_vectorizer, df, question):
    """
    TFIDF相关统计特征
    """
    def calc_tfidf(raw):
        cq1_tfidf = tfidf_vectorizer.transform([str(raw[question+'1'])]).data
        cq2_tfidf = tfidf_vectorizer.transform([str(raw[question+'2'])]).data
        cq1_tfidf = [0.] if cq1_tfidf.shape[0] == 0 else cq1_tfidf
        cq2_tfidf = [0.] if cq2_tfidf.shape[0] == 0 else cq2_tfidf

        return cq1_tfidf, cq2_tfidf

    df['tfidf'] = df.apply(lambda raw: calc_tfidf(raw), axis=1)

    df['sum_'+question+'1_tfidf'] = df['tfidf'].map(lambda raw: np.sum(raw[0]))
    df['sum_'+question+'2_tfidf'] = df['tfidf'].map(lambda raw: np.sum(raw[1]))
    df['mean_'+question+'1_tfidf'] = df['tfidf'].map(lambda raw: np.mean(raw[0]))
    df['mean_'+question+'2_tfidf'] = df['tfidf'].map(lambda raw: np.mean(raw[1]))
    df['var_'+question+'1_tfidf'] = df['tfidf'].map(lambda raw: np.var(raw[0]))
    df['var_'+question+'2_tfidf'] = df['tfidf'].map(lambda raw: np.var(raw[1]))
    df['std_'+question+'1_tfidf'] = df['tfidf'].map(lambda raw: np.std(raw[0]))
    df['std_'+question+'2_tfidf'] = df['tfidf'].map(lambda raw: np.std(raw[1]))
    df['len_'+question+'1_tfidf'] = df['tfidf'].map(lambda raw: len(raw[0]))
    df['len_'+question+'2_tfidf'] = df['tfidf'].map(lambda raw: len(raw[1]))

    df.drop(['tfidf'], inplace=True, axis=1)
    return df


def main(base_data_dir):
    op_scope = 1
    # if os.path.exists(Configure.processed_train_path.format(base_data_dir, op_scope + 1)):
    #     return

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(base_data_dir, op_scope)
    print("train: {}, test: {}".format(train.shape, test.shape))

    print('---> generate contain word count features')
    generate_contain_word_features(train, 'not')
    generate_contain_word_features(test, 'not')
    generate_contain_word_features(train, 'best')
    generate_contain_word_features(test, 'best')

    print('---> generate match shared words features')
    train['q_match_shared_words'] = train.apply(lambda x: generate_matchshared_words_features(x, "question"), axis=1)
    test['q_match_shared_words'] = test.apply(lambda x: generate_matchshared_words_features(x, "question"), axis=1)
    train['cq_match_shared_words'] = train.apply(lambda x: generate_matchshared_words_features(x, "cleaned_question"), axis=1)
    test['cq_match_shared_words'] = test.apply(lambda x: generate_matchshared_words_features(x, "cleaned_question"), axis=1)

    print('---> calc cleaned_question tfidf object')
    tfidf_vectorizer = train_tfidf_vectorizer(train, test, question='cleaned_question')
    print('---> generate cleaned_question tfidf features')
    train = generate_tfidf_features(tfidf_vectorizer, train, question='cleaned_question')
    test = generate_tfidf_features(tfidf_vectorizer, test, question='cleaned_question')

    print('---> calc question tfidf object')
    tfidf_vectorizer = train_tfidf_vectorizer(train, test, question='question')
    print('---> generate question tfidf features')
    train = generate_tfidf_features(tfidf_vectorizer, train, question='question')
    test = generate_tfidf_features(tfidf_vectorizer, test, question='question')

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
