#!/usr/bin/env bash

#!/usr/bin/env bash
export CUDA_VISIBILE_DEVICES=0
export PROJECT_DIR=/home/dlf/pyprojects/FEENDistill
export SOURCE_DIR=/home/dlf/pyprojects/InvestEventExtractor

python $PROJECT_DIR/run_distill.py \
    --do_test \
    --arch="test" \
    --do_lower_case \
    --save_best \
    --early_stop \
    --trigger \
    --partial \
    --seed=42 \
    --pretrained_model=$SOURCE_DIR/pretrained_model \
    --resume_path=$SOURCE_DIR/output/partial_trigger_seed42_no_lstm \
    --depth=12 \



