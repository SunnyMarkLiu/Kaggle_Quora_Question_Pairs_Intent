#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-10-5 上午11:28
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import cPickle
import numpy as np
from utils import data_utils
from optparse import OptionParser
from conf.configure import Configure


def generate_word_vector_map():
    """
    build index mapping words in the embeddings set
    to their embedding vector
    """
    embeddings_index = {}
    embeddings_index_path = '/d_2t/lq/kaggle/Kaggle_Quora_Question_Pairs_Intent/embeddings_index.pkl'
    if os.path.exists(embeddings_index_path):
        with open(embeddings_index_path, "rb") as f:
            embeddings_index = cPickle.load(f)
        return embeddings_index

    f = open(Configure.pretrained_wordvectors)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    with open(embeddings_index_path, "wb") as f:
        cPickle.dump(embeddings_index, f, -1)

    return embeddings_index


def generate_wordvectors_features(df):

    def get_wordvector(word):
        embedding_vector = embeddings_index.get(word)
        embedding_vector = embedding_vector if embedding_vector else [0] * 300
        return embedding_vector

    df['cleaned_question1_vc'] = df['cleaned_question1'].map(lambda x: [get_wordvector(word) for word in str(x).split()])
    df['cleaned_question2_vc'] = df['cleaned_question2'].map(lambda x: [get_wordvector(word) for word in str(x).split()])

    return df

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
print("========== generate word vector features ==========")
base_data_dir = options.base_data_dir

op_scope = 5
if os.path.exists(Configure.processed_train_path.format(base_data_dir, op_scope + 1)):
    exit()

print("---> load datasets from scope {}".format(op_scope))
train, test = data_utils.load_dataset(base_data_dir, op_scope)
print("train: {}, test: {}".format(train.shape, test.shape))

print('---> generate word vector mapping')
embeddings_index = generate_word_vector_map()

print("train: {}, test: {}".format(train.shape, test.shape))
print("---> save datasets")
data_utils.save_dataset(base_data_dir, train, test, op_scope + 1)

