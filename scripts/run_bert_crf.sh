#!/usr/bin/env bash
export PROJECT_DIR=/home/dlf/pyprojects/FEENDistill
export SOURCE_DIR=/home/dlf/pyprojects/InvestEventExtractor

python $PROJECT_DIR/run_bert_crf.py \
    --do_train \
    --arch="BERTCRF_L-4" \
    --do_lower_case \
    --save_best \
    --early_stop \
    --trigger \
    --partial \
    --seed=42 \
    --pretrained_model=$SOURCE_DIR/pretrained_model \
	--train_batch_size=10 \
    --depth=4


