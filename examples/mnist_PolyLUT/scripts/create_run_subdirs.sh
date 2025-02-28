#!/bin/bash

CONFIG_DIR=./ensemble_configs/averaging/deg1_same_output_scale_post_trans

for i in {1..3}; do 
new_dir=${CONFIG_DIR}_configs/config${i}
mkdir -p $new_dir
find -wholename "${CONFIG_DIR}/*.yml" -print | head -n2 | xargs mv -t $new_dir
done