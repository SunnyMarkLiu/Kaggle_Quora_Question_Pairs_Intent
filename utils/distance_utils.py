#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-3 下午6:40
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from fuzzywuzzy import fuzz


class DistanceUtil(object):

    @staticmethod
    def jaccard_similarity_distance(set_a, set_b):
        """
        Jaccard Similarity Distance
        """
        intersection = set(set_a).intersection(set(set_b))
        union = set(set_a).union(set(set_b))
        if len(union) == 0:
            return 0.0
        return 1.0 * len(intersection) / len(union)

    @staticmethod
    def countbased_cos_distance(tokenization1, tokenization2):
        """
        基于计数的 cos 距离
        """
        from collections import Counter
        from scipy import spatial

        def build_vector(iterable1, iterable2):
            counter1 = Counter(iterable1)
            counter2 = Counter(iterable2)
            all_items = set(counter1.keys()).union(set(counter2.keys()))
            vector1 = [counter1[k] for k in all_items]
            vector2 = [counter2[k] for k in all_items]
            vector1 = [1e-6] if len(vector1) == 0 else vector1
            vector2 = [1e-6] if len(vector2) == 0 else vector2
            return vector1, vector2

        v1, v2 = build_vector(tokenization1, tokenization2)
        dist = 1 - spatial.distance.cosine(v1, v2)
        return dist

    @staticmethod
    def levenshtein_distance(s1, s2):
        """
        莱文斯坦距离，又称Levenshtein距离/ edit distance，是编辑距离的一种。指两个字串之間，由一个转成另一个所需的最少编辑操作次数。
        允许的编辑操作包括将一个字符替换成另一个字符，插入一个字符，刪除一个字符。
        """
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    @staticmethod
    def fuzzy_matching_ratio(str1, str2, ratio_func='partial_ratio'):
        """
        字符串模糊匹配
        :param str1: 字符串
        :param str2: 字符串
        :param ratio_func: ratio, partial_ratio, token_sort_ratio, token_set_ratio
        :return: 
        """
        if ratio_func == 'ratio':
            # Normalize to [0 - 1] range.
            return fuzz.ratio(str1, str2) / 100.0
        if ratio_func == 'partial_ratio':
            return fuzz.partial_ratio(str1, str2) / 100.0
        if ratio_func == 'token_sort_ratio':
            return fuzz.token_sort_ratio(str1, str2) / 100.0
        if ratio_func == 'token_set_ratio':
            return fuzz.token_set_ratio(str1, str2) / 100.0
