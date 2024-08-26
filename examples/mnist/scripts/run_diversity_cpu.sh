#!/bin/bash

CUDA_DEVICE=$1


MODEL_NAME=adaboost_ensemble_size32
CONFIG=./model_ckpts/mnist/adaboost/adaboost_ensemble_size32/hparams.yml
CKPT=./model_ckpts/mnist/adaboost/adaboost_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=averaging_ensemble_size32
CONFIG=./model_ckpts/mnist/averaging/averaging_ensemble_size32/hparams.yml
CKPT=./model_ckpts/mnist/averaging/averaging_ensemble_size32/best_accuracy.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME


MODEL_NAME=bagging_ensemble_size32
CONFIG=./model_ckpts/mnist/bagging/bagging_ensemble_size32/hparams.yml
CKPT=./model_ckpts/mnist/bagging/bagging_ensemble_size32/last_ensemble_ckpt.pth

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 diversity_cpu.py \
    --evaluate \
    --checkpoint $CKPT \
    --config $CONFIG \
    --model_name $MODEL_NAME
