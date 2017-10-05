#!/usr/bin/env bash

# base dataset direction:
# -> perform_stem_words,
# -> perform_no_stem_words

base_data_dir=$1
python preprocess_cleaning.py -d ${base_data_dir}
python generate_statistic_features.py -d ${base_data_dir}
python generate_statistic_features2.py -d ${base_data_dir}
python generate_distance_features.py -d ${base_data_dir}
python generate_magic_features.py -d ${base_data_dir}
python generate_wordvector_distances.py -d ${base_data_dir}
