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


DATA_DIR=./data/hgcal22data_signal_driven_ttbar_v11/nElinks_5/
DATA_FILE=./data/hgcal22data_signal_driven_ttbar_v11/hgcal22data_signal_driven_ttbar_v11_nELinks5.npy 
SAVE_DIR=./autoencoder_synth
CLOCK=2
EXP_NAME="large_${CLOCK}ns"
CONFIG=./pareto_configs/large_pareto_model/large_pareto_model.yml
CKPT=./pareto_configs/large_pareto_model/large_pareto_model_best_loss.pth


CUDA_VISIBLE_DEVICES=0 python neq2lut.py \
   --data_dir $DATA_DIR \
   --data_file $DATA_FILE \
   --save_dir $SAVE_DIR \
   --experiment_name $EXP_NAME \
   --checkpoint $CKPT \
   --hparams_config $CONFIG \
   --add_registers \
   --clock_period $CLOCK \
   --compute_emd \
   --test_model \
   --simulate_pre_synthesis_verilog \
   # --gpu \ # Don't run on gpu