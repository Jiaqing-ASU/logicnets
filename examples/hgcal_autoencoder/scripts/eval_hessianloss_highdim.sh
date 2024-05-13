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

EXP_NAME=small_pareto_model_best_loss
CONFIG=./pareto_configs/small_pareto_model/small_pareto_model.yml
CKPT=./pareto_configs/small_pareto_model/small_pareto_model_best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_pareto_model_best_loss
CONFIG=./pareto_configs/medium_pareto_model/medium_pareto_model.yml
CKPT=./pareto_configs/medium_pareto_model/medium_pareto_model_best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=large_pareto_model_best_loss
CONFIG=./pareto_configs/large_pareto_model/large_pareto_model.yml
CKPT=./pareto_configs/large_pareto_model/large_pareto_model_best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=weak1_pareto_model_best_loss
CONFIG=./pareto_configs/weak1_pareto_model/weak1_pareto_model.yml
CKPT=./pareto_configs/weak1_pareto_model/weak1_pareto_model_best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=weak2_pareto_model_best_loss
CONFIG=./pareto_configs/weak2_pareto_model/weak2_pareto_model.yml
CKPT=./pareto_configs/weak2_pareto_model/weak2_pareto_model_best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=small_averaging_fixed_mask_averaging_small_fixed_mask_ensemble_size2_seed1
CONFIG=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size2_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size2_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=small_averaging_fixed_mask_averaging_small_fixed_mask_ensemble_size2_seed2
CONFIG=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size2_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size2_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=small_averaging_fixed_mask_averaging_small_fixed_mask_ensemble_size4_seed1
CONFIG=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size4_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size4_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=small_averaging_fixed_mask_averaging_small_fixed_mask_ensemble_size4_seed2
CONFIG=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size4_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size4_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=small_averaging_fixed_mask_averaging_small_fixed_mask_ensemble_size4_seed963241121
CONFIG=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size4_seed963241121/hparams.yml
CKPT=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size4_seed963241121/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=small_averaging_fixed_mask_averaging_small_fixed_mask_ensemble_size8_seed1
CONFIG=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size8_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size8_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=small_averaging_fixed_mask_averaging_small_fixed_mask_ensemble_size8_seed2
CONFIG=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size8_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size8_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=small_averaging_fixed_mask_averaging_small_fixed_mask_ensemble_size16_seed1
CONFIG=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size16_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size16_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=small_averaging_fixed_mask_averaging_small_fixed_mask_ensemble_size16_seed2
CONFIG=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size16_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size16_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=small_averaging_fixed_mask_averaging_small_fixed_mask_ensemble_size16_seed963241121
CONFIG=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size16_seed963241121/hparams.yml
CKPT=./averaging_fixed_mask_configs/small_averaging_fixed_mask/averaging_small_fixed_mask_ensemble_size16_seed963241121/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size2_seed1
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size2_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size2_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size2_seed2
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size2_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size2_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size4_seed1
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size4_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size4_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size4_seed2
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size4_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size4_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size4_seed524926359
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size4_seed524926359/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size4_seed524926359/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size8_seed1
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size8_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size8_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size8_seed2
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size8_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size8_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size16_seed1
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size16_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size16_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size16_seed2
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size16_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size16_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size16_seed524926359
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size16_seed524926359/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size16_seed524926359/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size32_seed1
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size32_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size32_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size32_seed2
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size32_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size32_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=medium_averaging_fixed_mask_averaging_medium_fixed_mask_ensemble_size32_seed524926359
CONFIG=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size32_seed524926359/hparams.yml
CKPT=./averaging_fixed_mask_configs/medium_averaging_fixed_mask/averaging_medium_fixed_mask_ensemble_size32_seed524926359/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=large_averaging_fixed_mask_averaging_large_fixed_mask_ensemble_size2_seed1
CONFIG=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size2_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size2_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=large_averaging_fixed_mask_averaging_large_fixed_mask_ensemble_size2_seed2
CONFIG=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size2_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size2_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=large_averaging_fixed_mask_averaging_large_fixed_mask_ensemble_size4_seed1
CONFIG=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size4_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size4_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=large_averaging_fixed_mask_averaging_large_fixed_mask_ensemble_size4_seed2
CONFIG=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size4_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size4_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=large_averaging_fixed_mask_averaging_large_fixed_mask_ensemble_size8_seed1
CONFIG=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size8_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size8_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=large_averaging_fixed_mask_averaging_large_fixed_mask_ensemble_size8_seed2
CONFIG=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size8_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size8_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=large_averaging_fixed_mask_averaging_large_fixed_mask_ensemble_size16_seed1
CONFIG=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size16_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size16_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=large_averaging_fixed_mask_averaging_large_fixed_mask_ensemble_size16_seed2
CONFIG=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size16_seed2/hparams.yml
CKPT=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size16_seed2/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=large_averaging_fixed_mask_averaging_large_fixed_mask_ensemble_size16_seed432384445
CONFIG=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size16_seed432384445/hparams.yml
CKPT=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size16_seed432384445/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG

EXP_NAME=large_averaging_fixed_mask_averaging_large_fixed_mask_ensemble_size32_seed1
CONFIG=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size32_seed1/hparams.yml
CKPT=./averaging_fixed_mask_configs/large_averaging_fixed_mask/averaging_large_fixed_mask_ensemble_size32_seed1/best_loss.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 loss_cpu_highdim.py \
    --data_dir $DATA_DIR \
    --data_file $DATA_FILE \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --checkpoint $CKPT \
    --gpu \
    --hparams_config $CONFIG