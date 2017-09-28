#!/usr/bin/env bash

# base dataset direction:
# -> stop_words_and_stem_words,
# -> stop_words_and_no_stem_words,
# -> no_stop_words_and_stem_words,
# -> no_stop_words_and_no_stem_words

base_data_dir=$1
echo ${base_data_dir}
python data_cleaning_basic_statistic_features.py -d ${base_data_dir}
