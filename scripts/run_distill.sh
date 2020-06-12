#!/usr/bin/env bash

export PROJECT_DIR=/home/dlf/pyprojects/FEENDistill
export SOURCE_DIR=/home/dlf/pyprojects/InvestEventExtractor

python $PROJECT_DIR/run_distill.py \
    --do_train \
    --arch="DIS_WDL_L-4" \
	--alpha=0.9	\
    --depth=4  \
    --do_lower_case \
    --save_best \
    --early_stop \
    --trigger \
    --partial \
    --seed=42 \
    --pretrained_model=$SOURCE_DIR/pretrained_model \
    --tea_resume_path=$SOURCE_DIR/output/partial_trigger_seed42_no_lstm \
    --train_batch_size=32 \
	--gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --ex_learning_rate=3e-5 \
	--sorted=1 \
    --stu_resume_path=$SOURCE_DIR/output/partial_trigger_seed42_no_lstm \
	#--stu_resume_path=$PROJECT_DIR/output/DIS_CombineLabel_L-6\
