#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-3 下午5:29
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
from utils import data_utils
from optparse import OptionParser

from utils.distance_utils import DistanceUtil

def generate_jaccard_similarity_distance(df):
    df['cq_jaccard_dist'] = df[['cleaned_question1', 'cleaned_question2']].apply(
        lambda row: DistanceUtil.jaccard_similarity_distance(df['cleaned_question1'], df['cleaned_question2']),
        axis=1
    )

def generate_count_based_cos_distance(df):
    df['q_count_cos_dis'] = df.apply(
        lambda row: DistanceUtil.countbased_cos_distance(str(row['question1']), str(row['question2'])),
        axis=1
    )

def generate_levenshtein_distance(df):
    df['q_levenshtein_dis'] = df.apply(
        lambda row: DistanceUtil.levenshtein_distance(str(row['question1']), str(row['question2'])),
        axis=1
    )

def generate_fuzzy_matching_ratio(df, question='question'):
    df['q_fuzzy_matching_ratio'] = df.apply(
        lambda row: DistanceUtil.fuzzy_matching_ratio(str(row[question+'1']), str(row[question+'2']),
                                                      ratio_func='ratio'),
        axis=1
    )
    df['q_fuzzy_matching_partial_ratio'] = df.apply(
        lambda row: DistanceUtil.fuzzy_matching_ratio(str(row[question+'1']), str(row[question+'2']),
                                                      ratio_func='partial_ratio'),
        axis=1
    )
    df['q_fuzzy_matching_token_sort_ratio'] = df.apply(
        lambda row: DistanceUtil.fuzzy_matching_ratio(str(row[question+'1']), str(row[question+'2']),
                                                      ratio_func='token_sort_ratio'),
        axis=1
    )
    df['q_fuzzy_matching_token_set_ratio'] = df.apply(
        lambda row: DistanceUtil.fuzzy_matching_ratio(str(row[question+'1']), str(row[question+'2']),
                                                      ratio_func='token_set_ratio'),
        axis=1
    )

def generate_char_ngram_distance(df, ngrams, question='question'):
    """
    基于 n-gram 的 char 各种距离
    :param df: 
    :param ngrams: list
    :param question: 
    :return: 
    """
    def calc_char_ngram_distance(row, n, q='question'):
        def get_char_ngrams(doc, n):
            return [doc[i:i + n] for i in range(len(doc) - n + 1)]

        q1_ngrams_chars = get_char_ngrams(str(row[q+'1']), n)
        q2_ngrams_chars = get_char_ngrams(str(row[q+'2']), n)

        cut_len = min((len(q1_ngrams_chars), len(q2_ngrams_chars)))

        cos_distances = []
        levenshtein_distances = []
        for i in range(cut_len):
            cos_distance = DistanceUtil.countbased_cos_distance(list(q1_ngrams_chars[i]), list(q2_ngrams_chars[i]))
            levenshtein_distance = DistanceUtil.levenshtein_distance(q1_ngrams_chars[i], q2_ngrams_chars[i])
            cos_distances.append(cos_distance)
            levenshtein_distances.append(levenshtein_distance)

        cos_distances = [0.0] if len(cos_distances) == 0 else cos_distances
        levenshtein_distances = [0.0] if len(levenshtein_distances) == 0 else levenshtein_distances

        return np.mean(cos_distances), np.mean(levenshtein_distances)

    for ngram in ngrams:
        df['char_ngram_distance'] = df.apply(lambda row: calc_char_ngram_distance(row, ngram, q=question), axis=1)
        df['{}_ngram_{}_cos_distance'.format(question, ngram)] = df['char_ngram_distance'].map(lambda x: x[0])
        df['{}_ngram_{}_levenshtein_distance'.format(question, ngram)] = df['char_ngram_distance'].map(lambda x: x[1])
    df.drop(['char_ngram_distance'], axis=1, inplace=True)

    return df


def generate_word_ngram_distance(df, ngrams, question='question'):
    """
    基于 n-gram 的 word 各种距离
    :param df: 
    :param ngrams: list
    :param question: 
    :return: 
    """
    def calc_word_ngram_distance(row, n, q='question'):
        def get_words_ngrams(doc, n):
            doc = doc.split()
            return [doc[i:i + n] for i in range(len(doc) - n + 1)]

        q1_ngrams_words = get_words_ngrams(str(row[q + '1']), n)
        q2_ngrams_words = get_words_ngrams(str(row[q + '2']), n)

        cut_len = min((len(q1_ngrams_words), len(q2_ngrams_words)))

        cos_distances = []
        levenshtein_distances = []
        for i in range(cut_len):
            cos_distance = DistanceUtil.countbased_cos_distance(q1_ngrams_words[i], q2_ngrams_words[i])
            levenshtein_distance = DistanceUtil.levenshtein_distance(q1_ngrams_words[i], q2_ngrams_words[i])
            cos_distances.append(cos_distance)
            levenshtein_distances.append(levenshtein_distance)

        cos_distances = [0.0] if len(cos_distances) == 0 else cos_distances
        levenshtein_distances = [0.0] if len(levenshtein_distances) == 0 else levenshtein_distances

        return np.mean(cos_distances), np.mean(levenshtein_distances)

    for ngram in ngrams:
        df['word_ngram_distance'] = df.apply(lambda row: calc_word_ngram_distance(row, ngram, q=question), axis=1)
        df['{}_ngram_{}_word_cos_distance'.format(question, ngram)] = df['word_ngram_distance'].map(lambda x: x[0])
        df['{}_ngram_{}_word_levenshtein_distance'.format(question, ngram)] = df['word_ngram_distance'].map(lambda x: x[1])
    df.drop(['word_ngram_distance'], axis=1, inplace=True)

    return df


def main(base_data_dir):
    op_scope = 3
    # if os.path.exists(Configure.processed_train_path.format(base_data_dir, op_scope + 1)):
    #     return

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(base_data_dir, op_scope)
    print("train: {}, test: {}".format(train.shape, test.shape))

    # print('---> generate jaccard similarity distance')
    # generate_jaccard_similarity_distance(train)
    # generate_jaccard_similarity_distance(test)

    print('---> generate count-based cos-distance, levenshtein_distance')
    generate_count_based_cos_distance(train)
    generate_count_based_cos_distance(test)
    generate_levenshtein_distance(train)
    generate_levenshtein_distance(test)

    print('---> generate fuzzy matching ratio')
    generate_fuzzy_matching_ratio(train)
    generate_fuzzy_matching_ratio(test)
    generate_fuzzy_matching_ratio(train, question='cleaned_question')
    generate_fuzzy_matching_ratio(test, question='cleaned_question')

    print('---> generate char ngram distance')
    train = generate_char_ngram_distance(train, ngrams=[10],question='question')
    test = generate_char_ngram_distance(test, ngrams=[10], question='question')

    print('---> generate word ngram distance')
    train = generate_word_ngram_distance(train, ngrams=[4],question='question')
    test = generate_word_ngram_distance(test, ngrams=[4], question='question')

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
    print("========== generate some distance features ==========")
    main(options.base_data_dir)