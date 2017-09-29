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
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
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

    df['max_' + question + '1_tfidf'] = df['tfidf'].map(lambda raw: np.max(raw[0]))
    df['max_' + question + '2_tfidf'] = df['tfidf'].map(lambda raw: np.max(raw[1]))
    df['min_' + question + '1_tfidf'] = df['tfidf'].map(lambda raw: np.min(raw[0]))
    df['min_' + question + '2_tfidf'] = df['tfidf'].map(lambda raw: np.min(raw[1]))
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


def train_hash_vectorizer(train, test, question):
    hash_vectorizer = HashingVectorizer(ngram_range=(1, 1))
    tfidf_txt = pd.Series(
                train[question+'1'].tolist() + train[question+'2'].tolist() +
                test[question+'1'].tolist() + test[question+'2'].tolist()).astype(str)
    hash_vectorizer.fit(tfidf_txt)
    return hash_vectorizer


def generate_hash_features(hash_vectorizer, df, question):
    """
    hash 相关统计特征
    """
    def calc_hash(raw):
        cq1_hash = hash_vectorizer.transform([str(raw[question+'1'])]).data
        cq2_hash = hash_vectorizer.transform([str(raw[question+'2'])]).data
        cq1_hash = [0.] if cq1_hash.shape[0] == 0 else cq1_hash
        cq2_hash = [0.] if cq2_hash.shape[0] == 0 else cq2_hash

        return cq1_hash, cq2_hash

    df['hash'] = df.apply(lambda raw: calc_hash(raw), axis=1)

    df['max_' + question + '1_hash'] = df['hash'].map(lambda raw: np.max(raw[0]))
    df['max_' + question + '2_hash'] = df['hash'].map(lambda raw: np.max(raw[1]))
    df['min_' + question + '1_hash'] = df['hash'].map(lambda raw: np.min(raw[0]))
    df['min_' + question + '2_hash'] = df['hash'].map(lambda raw: np.min(raw[1]))
    df['sum_'+question+'1_hash'] = df['hash'].map(lambda raw: np.sum(raw[0]))
    df['sum_'+question+'2_hash'] = df['hash'].map(lambda raw: np.sum(raw[1]))
    df['mean_'+question+'1_hash'] = df['hash'].map(lambda raw: np.mean(raw[0]))
    df['mean_'+question+'2_hash'] = df['hash'].map(lambda raw: np.mean(raw[1]))
    df['var_'+question+'1_hash'] = df['hash'].map(lambda raw: np.var(raw[0]))
    df['var_'+question+'2_hash'] = df['hash'].map(lambda raw: np.var(raw[1]))
    df['std_'+question+'1_hash'] = df['hash'].map(lambda raw: np.std(raw[0]))
    df['std_'+question+'2_hash'] = df['hash'].map(lambda raw: np.std(raw[1]))
    df['len_'+question+'1_hash'] = df['hash'].map(lambda raw: len(raw[0]))
    df['len_'+question+'2_hash'] = df['hash'].map(lambda raw: len(raw[1]))

    df.drop(['hash'], inplace=True, axis=1)
    return df


def generate_question_occur_count(train, test):
    """
    问题在数据集中出现的次数
    """
    q_num = {}
    for index, raw in train.iterrows():
        q1 = str(raw['cleaned_question1']).strip()
        q2 = str(raw['cleaned_question2']).strip()
        q_num[q1] = q_num.get(q1, 0) + 1
        if q1 != q2:
            q_num[q2] = q_num.get(q2, 0) + 1

    for index, raw in test.iterrows():
        q1 = str(raw['cleaned_question1']).strip()
        q2 = str(raw['cleaned_question2']).strip()
        q_num[q1] = q_num.get(q1, 0) + 1
        if q1 != q2:
            q_num[q2] = q_num.get(q2, 0) + 1

    train['cleaned_question1_occur_count'] = train['cleaned_question1'].map(q_num)
    train['cleaned_question2_occur_count'] = train['cleaned_question2'].map(q_num)
    test['cleaned_question1_occur_count'] = test['cleaned_question1'].map(q_num)
    test['cleaned_question2_occur_count'] = test['cleaned_question2'].map(q_num)

    return train, test


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

    print('---> calc question hash object')
    hash_vectorizer = train_hash_vectorizer(train, test, question='question')
    print('---> generate question hash features')
    train = generate_hash_features(hash_vectorizer, train, question='question')
    test = generate_hash_features(hash_vectorizer, test, question='question')

    print('---> calc cleaned_question hash object')
    hash_vectorizer = train_hash_vectorizer(train, test, question='cleaned_question')
    print('---> generate cleaned_question hash features')
    train = generate_hash_features(hash_vectorizer, train, question='cleaned_question')
    test = generate_hash_features(hash_vectorizer, test, question='cleaned_question')

    print('---> generate question occurs count')
    train, test = generate_question_occur_count(train, test)

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
    print("========== generate some statistic features ==========")
    main(options.base_data_dir)
