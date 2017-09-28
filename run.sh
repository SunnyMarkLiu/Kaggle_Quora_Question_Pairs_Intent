#!/usr/bin/env bash
cd features
sh run.sh
cd ../model/

# base dataset direction:
# -> stop_words_and_stem_words,
# -> stop_words_and_no_stem_words,
# -> no_stop_words_and_stem_words,
# -> no_stop_words_and_no_stem_words

base_data_dir='stop_words_and_stem_words'
python xgboost_model.py -d ${base_data_dir}
