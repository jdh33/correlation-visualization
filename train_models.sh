#!/bin/bash

# SELECTED_MODELS='rr,rfr'
# OUTPUT_SUFFIX=`date +%Y%m%d`
N_JOBS=1

python scripts/train_models.py --training_dataset "who_life_expectancy" \
        --target_variable "Life expectancy" --train_with_covar --train_with_shuffled_data \
        --timestamp --n_jobs=$N_JOBS

# python scripts/train_models.py --training_dataset "who_life_expectancy" \
#         --target_variable "Life expectancy" --selected_models "$SELECTED_MODELS" \
#         --train_with_covar --train_with_shuffled_data --timestamp --n_jobs=$N_JOBS
