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

from time import time
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
from utils import data_utils
from utils.text.preprocessor import TextPreProcessor

english_stopwords = set(stopwords.words('english'))
word_tokenize = WordPunctTokenizer().tokenize
preprocessor = TextPreProcessor()
stop_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these',
              'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while',
              'during', 'to', 'What', 'Which', 'Is', 'If', 'While', 'This']


def get_unigram_words(que):
    """
    获取单一有效词汇
    """
    return [word for word in word_tokenize(que.lower()) if word not in english_stopwords]


def generate_unigram_words_features(df):
    df['unigrams_ques1'] = df['question1'].apply(lambda x: get_unigram_words(str(x)))
    df['unigrams_ques2'] = df['question2'].apply(lambda x: get_unigram_words(str(x)))

    def get_common_unigrams(row):
        """ 获取两个问题中包含相同词汇的数目 """
        return len(set(row["unigrams_ques1"]).intersection(set(row["unigrams_ques2"])))

    def get_common_unigram_ratio(row):
        """ 获取两个问题中包含相同词汇的比例 """
        return float(row["unigrams_common_count"]) / \
               max(len(set(row["unigrams_ques1"]).union(set(row["unigrams_ques2"]))), 1)

    df["unigrams_common_count"] = df.apply(lambda row: get_common_unigrams(row), axis=1)
    df["unigrams_common_ratio"] = df.apply(lambda row: get_common_unigram_ratio(row), axis=1)

    df.drop(['unigrams_ques1', 'unigrams_ques2'], inplace=True, axis=1)
    return df


def clean_text(text, remove_stop_words=True, stem_words=False):
    """
    Clean the text, with the option to remove stop_words and to stem words
    """
    # 首先清洗缩略词
    text = re.sub(r"ain't", " is not ", text.lower())
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"shan't", "shall not", text)
    text = re.sub(r"sha'n't", "shall not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"how'd", "how did", text)
    text = re.sub(r"how'd'y", "how do you", text)
    text = re.sub(r"where'd", "where did", text)
    text = re.sub(r"'m", " am ", text)
    text = re.sub(r"'d", " would had ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'cause", "because", text)
    text = re.sub(r"ma'am", "madam", text)
    text = re.sub(r"o'clock", "of the clock", text)
    text = re.sub(r"y'all", "you all", text)
    # 去除超链接
    text = re.sub(r"(\S*)https?://\S*", lambda m: m.group(1), text)

    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"iii", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    text = re.sub(r"(\d+)kgs\s*", lambda m: m.group(1) + ' kilogram ', text)  # e.g. 4kgs => 4 kg
    text = re.sub(r"(\d+)kg\s*", lambda m: m.group(1) + ' kilogram ', text)  # e.g. 4kg => 4 kg
    text = re.sub(r"(\d+)k\s*", lambda m: m.group(1) + '000 ', text)  # e.g. 4k => 4000
    text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
    text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

    text = re.sub(r"1st", " first ", text)
    text = re.sub(r"2nd", " second ", text)
    text = re.sub(r"3rd", " third ", text)
    text = re.sub(r"4th", " fourth ", text)
    text = re.sub(r"5th", " fifth ", text)
    text = re.sub(r"6th", " sixth ", text)
    text = re.sub(r"7th", " seventh ", text)
    text = re.sub(r"8th", " eighth ", text)
    text = re.sub(r"9th", " ninth ", text)
    text = re.sub(r"10th", " tenth ", text)

    text = re.sub(r"0", " zero ", text)
    text = re.sub(r"1", " one ", text)
    text = re.sub(r"2", " two ", text)
    text = re.sub(r"3", " three ", text)
    text = re.sub(r"4", " four ", text)
    text = re.sub(r"5", " five ", text)
    text = re.sub(r"6", " six ", text)
    text = re.sub(r"7", " seven ", text)
    text = re.sub(r"8", " eight ", text)
    text = re.sub(r"9", " nine ", text)

    text = re.sub(r"&amp", " and ", text)
    text = re.sub(r"&quot", ' " ', text)
    text = re.sub(r"&lt", " less than ", text)
    text = re.sub(r"&gt", " greater than ", text)
    text = re.sub(r"&nbsp", " ", text)

    # 去除标点符合
    text = ''.join([c for c in text if c not in punctuation])
    # 去除空白字符
    text = re.sub(r"\s+", " ", text)

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return text


def generate_cleaned_unigram_words_features(df):
    df['cleaned_unigrams_ques1'] = df['cleaned_question1'].apply(lambda x: get_unigram_words(str(x)))
    df['cleaned_unigrams_ques2'] = df['cleaned_question2'].apply(lambda x: get_unigram_words(str(x)))

    def get_common_unigrams(row):
        """ 获取两个问题中包含相同词汇的数目 """
        return len(set(row["cleaned_unigrams_ques1"]).intersection(set(row["cleaned_unigrams_ques2"])))

    def get_common_unigram_ratio(row):
        """ 获取两个问题中包含相同词汇的比例 """
        return float(row["cleaned_unigrams_common_count"]) / \
               max(len(set(row["cleaned_unigrams_ques1"]).union(set(row["cleaned_unigrams_ques2"]))), 1)

    df["cleaned_unigrams_common_count"] = df.apply(lambda row: get_common_unigrams(row), axis=1)
    df["cleaned_unigrams_common_ratio"] = df.apply(lambda row: get_common_unigram_ratio(row), axis=1)

    df.drop(['cleaned_unigrams_ques1', 'cleaned_unigrams_ques2'], inplace=True, axis=1)
    return df


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

    print('---> generate unigram_words features before cleaned')
    train = generate_unigram_words_features(train)
    test = generate_unigram_words_features(test)

    print('---> clean text')
    start = time()
    print('clean train question1')
    train['cleaned_question1'] = train['question1'].apply(lambda x: clean_text(str(x)))
    print('clean train question2')
    train['cleaned_question2'] = train['question2'].apply(lambda x: clean_text(str(x)))
    print('clean test question1')
    test['cleaned_question1'] = test['question1'].apply(lambda x: clean_text(str(x)))
    print('clean test question2')
    test['cleaned_question2'] = test['question2'].apply(lambda x: clean_text(str(x)))
    stop = time()
    print("text cleaned, cost {}s".format(stop, str(stop - start)))

    print('---> generate unigram_words features after cleaned')
    train = generate_cleaned_unigram_words_features(train)
    test = generate_cleaned_unigram_words_features(test)

    print("train: {}, test: {}".format(train.shape, test.shape))
    print("---> save datasets")
    data_utils.save_dataset(train, test, op_scope + 1)


if __name__ == "__main__":
    print("========== perform data cleaning and basic statistic feature engineering ==========")
    main()
