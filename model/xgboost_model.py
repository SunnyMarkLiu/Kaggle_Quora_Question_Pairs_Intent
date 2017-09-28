#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-28 下午12:50
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import time

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import xgboost as xgb

from utils import data_utils, feature_util


def main():
    # final operate dataset
    files = os.listdir('../input')
    op_scope = 0
    for f in files:
        if 'operate' in f:
            op = int(f.split('_')[1])
            if op > op_scope:
                op_scope = op

    print("---> load datasets from scope {}".format(op_scope))
    train, test = data_utils.load_dataset(op_scope)

    id_train = train['id']
    id_test = test['test_id']

    train.drop(['id', 'qid1', 'qid2', 'question1', 'question2'], axis=1, inplace=True)
    test.drop(['test_id', 'question1', 'question2'], axis=1, inplace=True)

    shuffled_index = np.arange(0, train.shape[0], 1)
    np.random.shuffle(shuffled_index)

    # random_indexs = shuffled_index[:int(train.shape[0] * 0.70)]
    random_indexs = shuffled_index
    # random_indexs = np.arange(0, train.shape[0], 2)
    train = train.iloc[random_indexs, :]

    y_train = train['is_duplicate']
    del train['is_duplicate']
    print("train: {}, test: {}".format(train.shape, test.shape))
    print('---> feature check before modeling')
    feature_util.feature_check_before_modeling(train, test, train.columns)

    print("---> start cv training")
    X_train = train
    X_test = test
    df_columns = train.columns.values
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)

    xgb_params = {
        'eta': 0.01,
        'subsample': 0.9,
        'max_depth': 8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'updater': 'grow_gpu',
        'gpu_id': 1,
        'nthread': -1,
        'silent': 1
    }

    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)

    cv_result = xgb.cv(dict(xgb_params),
                       dtrain,
                       num_boost_round=400,
                       early_stopping_rounds=100,
                       verbose_eval=20,
                       show_stdv=False,
                       )

    best_num_boost_rounds = len(cv_result)
    print('best_num_boost_rounds = {}'.format(best_num_boost_rounds))
    # train model
    print('---> training on total training data')
    model = xgb.train(dict(xgb_params), dtrain,
                      num_boost_round=best_num_boost_rounds)

    print('---> predict submit')
    y_pred = model.predict(dtest)
    df_sub = pd.DataFrame({'test_id': id_test, 'is_duplicate': y_pred})
    submission_path = '../result/{}_submission_{}.csv.gz'.format('xgboost',
                                                                 time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                               time.localtime(time.time())))
    df_sub.to_csv(submission_path, index=False, compression='gzip')
    print('---> submit to kaggle')
    kg_password = raw_input("kaggle password: ")
    kg_comment = raw_input("submit comment: ")
    cmd = "kg submit {} -u sunnymarkliu -p '{}' -c quora-question-pairs -m '{}'".format(submission_path,
                                                                                        kg_password,
                                                                                        kg_comment)
    os.system(cmd)


if __name__ == "__main__":
    print("========== apply xgboost model ==========")
    main()
