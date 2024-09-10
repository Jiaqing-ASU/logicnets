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

#adaboost
MODEL_NAME=adaboost_large_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/adaboost/adaboost_large_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/adaboost/adaboost_large_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=adaboost_medium_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/adaboost/adaboost_medium_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/adaboost/adaboost_medium_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=adaboost_small_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/adaboost/adaboost_small_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/adaboost/adaboost_small_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


#averaging
MODEL_NAME=averaging_large_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_large_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_large_ensemble_size32/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=averaging_large_ensemble_size16
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_large_ensemble_size16/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_large_ensemble_size16/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=averaging_large_ensemble_size8
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_large_ensemble_size8/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_large_ensemble_size8/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=averaging_large_ensemble_size4
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_large_ensemble_size4/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_large_ensemble_size4/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=averaging_large_ensemble_size2
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_large_ensemble_size2/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_large_ensemble_size2/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=averaging_medium_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_medium_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_medium_ensemble_size32/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=averaging_medium_ensemble_size16
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_medium_ensemble_size16/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_medium_ensemble_size16/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=averaging_medium_ensemble_size8
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_medium_ensemble_size8/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_medium_ensemble_size8/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=averaging_medium_ensemble_size4
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_medium_ensemble_size4/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_medium_ensemble_size4/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=averaging_medium_ensemble_size2
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_medium_ensemble_size2/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_medium_ensemble_size2/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=averaging_small_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_small_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_small_ensemble_size32/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=averaging_small_ensemble_size16
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_small_ensemble_size16/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_small_ensemble_size16/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=averaging_small_ensemble_size8
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_small_ensemble_size8/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_small_ensemble_size8/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=averaging_small_ensemble_size4
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_small_ensemble_size4/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_small_ensemble_size4/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=averaging_small_ensemble_size2
CONFIG=./model_ckpts/jet_substructure/averaging/averaging_small_ensemble_size2/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/averaging_small_ensemble_size2/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


# bagging
MODEL_NAME=bagging_large_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/bagging/bagging_large_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/bagging/bagging_large_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=bagging_medium_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/bagging/bagging_medium_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/bagging/bagging_medium_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=bagging_small_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/bagging/bagging_small_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/bagging/bagging_small_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME
