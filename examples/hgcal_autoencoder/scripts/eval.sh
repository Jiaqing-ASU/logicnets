#!/bin/bash
#  Copyright (C) 2023, Advanced Micro Devices, Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


CUDA_DEVICE=$1

DATA_DIR=./data/hgcal22data_signal_driven_ttbar_v11/nElinks_5/
DATA_FILE=./data/hgcal22data_signal_driven_ttbar_v11/hgcal22data_signal_driven_ttbar_v11_nELinks5.npy 
SAVE_DIR=./test
EXP_NAME=small_lr0.01_warm_restart_freq50_wd0.01_batch_size512
CONFIG=./grid_search_configs/avg_emd_small_configs/config1/small_lr0.01_warm_restart_freq50_wd0.01_batch_size512.yml
CKPT=./sml_avg_emd_hp_grid_search/small_lr0.01_warm_restart_freq50_wd0.01_batch_size512/best_loss.pth


# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 train.py \
#     --data_dir $DATA_DIR \
#     --data_file $DATA_FILE \
#     --save_dir $SAVE_DIR \
#     --experiment_name $EXP_NAME \
#     --evaluate \
#     --checkpoint $CKPT \
#     --gpu \
#     --hparams_config $CONFIG

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 train_cpu.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG