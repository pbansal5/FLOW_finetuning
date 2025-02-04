#!/bin/bash

RUN_1="experiments/instruction_tuning/Llama-3.2-3B/20250123_221031_Llama-3.2-3B__r256__lr5e-06__train_train"
RUN_2="experiments/instruction_tuning/Llama-3.2-3B/20250123_223622_Llama-3.2-3B__r256__lr2e-05__train_train"
ALPHA=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) # Alpha values for model averaging
BASE_DIR="$SCRATCH/floss_results/model_averaging/Llama3.2_3b__Baseline__and__Full-FT" # Base directory for experiments

#### ~~~~~~~~ Do not modify below this line ~~~~~~~~ ####
MODEL_1="${RUN_1}/final_model"
TOKENIZER="${RUN_1}/tokenizer"
MODEL_2="${RUN_2}/final_model"

python average_model_weights.py \
    --model_1 ${MODEL_1} \
    --model_2 ${MODEL_2} \
    --tokenizer ${TOKENIZER} \
    --alpha ${ALPHA[@]} \
    --output ${BASE_DIR}