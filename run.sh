#!/usr/bin/env bash

# base dataset direction:
# -> stop_words_and_stem_words,
# -> stop_words_and_no_stem_words,
# -> no_stop_words_and_stem_words,
# -> no_stop_words_and_no_stem_words

base_data_dir='stop_words_and_stem_words'
echo "==> base_data_dir:" ${base_data_dir}

cd features
sh run.sh ${base_data_dir}
cd ../model/

python xgboost_model.py -d ${base_data_dir}
