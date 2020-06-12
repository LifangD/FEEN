#!/usr/bin/env bash
export CUDA_VISIBILE_DEVICES=0
export PROJECT_DIR=/home/dlf/pyprojects/FEENDistill
export SOURCE_DIR=/home/dlf/pyprojects/InvestEventExtractor

python $PROJECT_DIR/run_cnn_lstm_crf.py \
    --do_test \
    --arch="cnnlstmcrf" \
    --do_lower_case \
    --save_best \
    --early_stop \
    --trigger \
    --partial \
    --seed=42 \
	--resume_path=$PROJECT_DIR/output/cnnlstmcrf \
    --pretrained_model=/home/dlf/pyprojects/InvestEventExtractor/pretrained_model
