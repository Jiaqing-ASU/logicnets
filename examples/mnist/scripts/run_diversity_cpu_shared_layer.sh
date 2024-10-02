#!/bin/bash

CUDA_DEVICE=$1

CONFIG=./model_ckpts/mnist/averaging/averaging_xs_ensemble_size2/hparams.yml
CKPT=./model_ckpts/mnist/averaging/averaging_xs_ensemble_size2/best_accuracy.pth
SAVE_DIR=./diversity
EXP_NAME=averaging_xs_ensemble_size2

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --config $CONFIG \
    --checkpoint $CKPT

CONFIG=./model_ckpts/mnist/averaging/averaging_xs_ensemble_size4/hparams.yml
CKPT=./model_ckpts/mnist/averaging/averaging_xs_ensemble_size4/best_accuracy.pth
SAVE_DIR=./diversity
EXP_NAME=averaging_xs_ensemble_size4

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --config $CONFIG \
    --checkpoint $CKPT

CONFIG=./model_ckpts/mnist/averaging/averaging_xs_ensemble_size8/hparams.yml
CKPT=./model_ckpts/mnist/averaging/averaging_xs_ensemble_size8/best_accuracy.pth
SAVE_DIR=./diversity
EXP_NAME=averaging_xs_ensemble_size8

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --config $CONFIG \
    --checkpoint $CKPT

CONFIG=./model_ckpts/mnist/averaging/averaging_xs_ensemble_size16/hparams.yml
CKPT=./model_ckpts/mnist/averaging/averaging_xs_ensemble_size16/best_accuracy.pth
SAVE_DIR=./diversity
EXP_NAME=averaging_xs_ensemble_size16

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --config $CONFIG \
    --checkpoint $CKPT

CONFIG=./model_ckpts/mnist/averaging/averaging_xs_ensemble_size32/hparams.yml
CKPT=./model_ckpts/mnist/averaging/averaging_xs_ensemble_size32/best_accuracy.pth
SAVE_DIR=./diversity
EXP_NAME=averaging_xs_ensemble_size32

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --config $CONFIG \
    --checkpoint $CKPT

CONFIG=./model_ckpts/mnist/averaging/averaging_l_ensemble_size2/hparams.yml
CKPT=./model_ckpts/mnist/averaging/averaging_l_ensemble_size2/best_accuracy.pth
SAVE_DIR=./diversity
EXP_NAME=averaging_l_ensemble_size2

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --config $CONFIG \
    --checkpoint $CKPT

CONFIG=./model_ckpts/mnist/averaging/averaging_l_ensemble_size4/hparams.yml
CKPT=./model_ckpts/mnist/averaging/averaging_l_ensemble_size4/best_accuracy.pth
SAVE_DIR=./diversity
EXP_NAME=averaging_l_ensemble_size4

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --config $CONFIG \
    --checkpoint $CKPT

CONFIG=./model_ckpts/mnist/averaging/averaging_l_ensemble_size8/hparams.yml
CKPT=./model_ckpts/mnist/averaging/averaging_l_ensemble_size8/best_accuracy.pth
SAVE_DIR=./diversity
EXP_NAME=averaging_l_ensemble_size8

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_shared_layer_cpu.py \
    --save_dir $SAVE_DIR \
    --experiment_name $EXP_NAME \
    --evaluate \
    --config $CONFIG \
    --checkpoint $CKPT
