#!/usr/bin/env bash

# base dataset direction:
# -> perform_stem_words,
# -> perform_no_stem_words

base_data_dir='perform_no_stem_words'
echo "==> base_data_dir:" ${base_data_dir}

cd features
sh run.sh ${base_data_dir}
cd ../model/

python xgboost_model.py -d ${base_data_dir}
