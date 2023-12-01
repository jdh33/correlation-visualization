#!/bin/bash

# SELECTED_MODELS='rr,rfr'
# OUTPUT_SUFFIX=`date +%Y%m%d`
N_JOBS=1

python scripts/train_models.py --training_dataset "life-expectancy-who" \
        --target_variable "Life expectancy" --train_with_covar --train_with_shuffled_data \
        --timestamp --n_jobs=$N_JOBS
