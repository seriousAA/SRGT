#!/usr/bin/env bash

data_root=$1
output_dir=$2

for percent in 1 2 5 10; do
      for fold in 1 2 3 4 5 6 7 8 9 10; do
          python $(dirname "$0")/sample_pkl_class_wise.py ${data_root}/patch_annfile.pkl ${percent} ${fold} ${output_dir}
          cp ${data_root}/ori_annfile.pkl ${output_dir}/${fold}_${percent}/labeled/ori_annfile.pkl
          cp ${data_root}/split_config.json ${output_dir}/${fold}_${percent}/labeled/split_config.json
          cp ${data_root}/ori_annfile.pkl ${output_dir}/${fold}_${percent}/unlabeled/ori_annfile.pkl
          cp ${data_root}/split_config.json ${output_dir}/${fold}_${percent}/unlabeled/split_config.json
      done
done
