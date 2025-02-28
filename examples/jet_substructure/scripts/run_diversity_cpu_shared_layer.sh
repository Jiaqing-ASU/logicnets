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


#averaging
MODEL_NAME=large_shared_input_output_layers_ensemble_size2
CONFIG=./model_ckpts/jet_substructure/averaging/large_shared_input_output_layers_ensemble_size2/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/large_shared_input_output_layers_ensemble_size2/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=large_shared_input_output_layers_ensemble_size4
CONFIG=./model_ckpts/jet_substructure/averaging/large_shared_input_output_layers_ensemble_size4/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/large_shared_input_output_layers_ensemble_size4/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=large_shared_input_output_layers_ensemble_size8
CONFIG=./model_ckpts/jet_substructure/averaging/large_shared_input_output_layers_ensemble_size8/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/large_shared_input_output_layers_ensemble_size8/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=large_shared_input_output_layers_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/averaging/large_shared_input_output_layers_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/large_shared_input_output_layers_ensemble_size32/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=medium_shared_input_output_layers_ensemble_size2
CONFIG=./model_ckpts/jet_substructure/averaging/medium_shared_input_output_layers_ensemble_size2/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/medium_shared_input_output_layers_ensemble_size2/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=medium_shared_input_output_layers_ensemble_size4
CONFIG=./model_ckpts/jet_substructure/averaging/medium_shared_input_output_layers_ensemble_size4/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/medium_shared_input_output_layers_ensemble_size4/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=medium_shared_input_output_layers_ensemble_size8
CONFIG=./model_ckpts/jet_substructure/averaging/medium_shared_input_output_layers_ensemble_size8/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/medium_shared_input_output_layers_ensemble_size8/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=medium_shared_input_output_layers_ensemble_size16
CONFIG=./model_ckpts/jet_substructure/averaging/medium_shared_input_output_layers_ensemble_size16/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/medium_shared_input_output_layers_ensemble_size16/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=medium_shared_input_output_layers_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/averaging/medium_shared_input_output_layers_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/medium_shared_input_output_layers_ensemble_size32/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=small_shared_input_output_layers_ensemble_size2
CONFIG=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size2/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size2/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=small_shared_input_output_layers_ensemble_size4
CONFIG=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size4/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size4/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=small_shared_input_output_layers_ensemble_size8
CONFIG=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size8/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size8/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=small_shared_input_output_layers_ensemble_size16
CONFIG=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size16/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size16/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=small_shared_input_output_layers_ensemble_size32
CONFIG=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size32/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size32/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

MODEL_NAME=small_shared_input_output_layers_ensemble_size64
CONFIG=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size64/hparams.yml
CKPT=./model_ckpts/jet_substructure/averaging/small_shared_input_output_layers_ensemble_size64/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME

# MODEL_NAME=small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size2
# CONFIG=./model_ckpts/jet_substructure/averaging/small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size2/hparams.yml
# CKPT=./model_ckpts/jet_substructure/averaging/small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size2/best_accuracy.pth

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
#     --evaluate \
#     --checkpoint $CKPT \
#     --config $CONFIG \
#     --model_name $MODEL_NAME

# MODEL_NAME=small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size4
# CONFIG=./model_ckpts/jet_substructure/averaging/small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size4/hparams.yml
# CKPT=./model_ckpts/jet_substructure/averaging/small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size4/best_accuracy.pth

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
#     --evaluate \
#     --checkpoint $CKPT \
#     --config $CONFIG \
#     --model_name $MODEL_NAME

# MODEL_NAME=small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size8
# CONFIG=./model_ckpts/jet_substructure/averaging/small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size8/hparams.yml
# CKPT=./model_ckpts/jet_substructure/averaging/small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size8/best_accuracy.pth

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
#     --evaluate \
#     --checkpoint $CKPT \
#     --config $CONFIG \
#     --model_name $MODEL_NAME

# MODEL_NAME=small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size16
# CONFIG=./model_ckpts/jet_substructure/averaging/small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size16/hparams.yml
# CKPT=./model_ckpts/jet_substructure/averaging/small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size16/best_accuracy.pth

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
#     --evaluate \
#     --checkpoint $CKPT \
#     --config $CONFIG \
#     --model_name $MODEL_NAME

# MODEL_NAME=small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size32
# CONFIG=./model_ckpts/jet_substructure/averaging/small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size32/hparams.yml
# CKPT=./model_ckpts/jet_substructure/averaging/small_shared_sparse_input_layer_shared_input_bitwidth6_ensemble_size32/best_accuracy.pth

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
#     --evaluate \
#     --checkpoint $CKPT \
#     --config $CONFIG \
#     --model_name $MODEL_NAME
