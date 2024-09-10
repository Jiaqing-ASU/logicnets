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
MODEL_NAME=adaboost_large_independent_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/adaboost_independent/adaboost_large_independent_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/adaboost_independent/adaboost_large_independent_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=adaboost_medium_independent_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/adaboost_independent/adaboost_medium_independent_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/adaboost_independent/adaboost_medium_independent_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=adaboost_small_independent_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/adaboost_independent/adaboost_small_independent_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/adaboost_independent/adaboost_small_independent_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


#bagging
MODEL_NAME=bagging_large_independent_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/bagging_independent/bagging_large_independent_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/bagging_independent/bagging_large_independent_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=bagging_medium_independent_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/bagging_independent/bagging_medium_independent_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/bagging_independent/bagging_medium_independent_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=bagging_small_independent_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/bagging_independent/bagging_small_independent_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/bagging_independent/bagging_small_independent_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME
