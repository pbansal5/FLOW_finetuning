#!/bin/bash
#SBATCH -J Analyze-Temp # Job name
#SBATCH -o slurmlogs/Analyze-LP.o%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/Analyze-LP.e%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gh                 # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 16:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name

# Source conda environment and move to project directory

source $WORK/miniconda3/install/bin/activate flow
cd /u/pbansal/flow/FLOW_finetuning/language

# NOTE: This script expects a single NODE to be used for analysis
# Job Configuration
export CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7' # Specify which GPU to use

DATASET="meta-math/MetaMathQA" # CHANGE ME
SPLIT="train" # CHANGE ME
FIELD=("query" "response") # CHANGE ME

# Note Either set the (RUN_DIR) or set (MODEL_PATH, TOKENIZER, and SAVE_PATH)
# RUN_DIR will always take priority
# RUN_DIR=" " # CHANGE ME
# RUN_DIR="experiments/instruction_tuning/Llama-3.2-3B/20250123_221031_Llama-3.2-3B__r256__lr5e-06__train_train" # CHANGE ME
 
MODEL_PATH="meta-llama/Llama-3.2-1B" # CHANGE ME
TOKENIZER="meta-llama/Llama-3.2-1B" # CHANGE ME
SAVE_PATH="meta-math/MetaMathQA" # CHANGE ME

LOSS_TYPE="both" # both, sequence, or token

BATCH_SIZE=4 # Adjust as needed
MAX_SEQ_LEN=1024 # Be sure to be consistent through experiments
DEBUG=false # true or false (Will turn off distributed training)
USE_SLURM=false # true or false (Will use SLURM for distributed training)

#### ~~~~~~~~ Do not modify below this line ~~~~~~~~ ####
if [ "$USE_SLURM" = true ]; then
    export MASTER_PORT=12802

    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
else
    export MASTER_ADDR="localhost"
    export MASTER_PORT="1231"
fi

# Calculate NUM_GPUS by counting commas and adding 1
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=0
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
fi

if [ "$DEBUG" = false  ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "NUM_GPUS: $NUM_GPUS"
else
    echo "Running in Debug Mode: Setting CUDA_VISIBLE_DEVICES to 1, NUM_GPUS to 1, and disabling wandb logging"
    export CUDA_VISIBLE_DEVICES=0
    NUM_GPUS=1
fi

if [ "$DEBUG" = false ] && [ $NUM_GPUS -gt 1 ]; then
    echo "Master Address: $MASTER_ADDR"
    echo "Master Port: $MASTER_PORT"

    echo "=== Starting Distributed Analysis ==="
    torchrun --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=$NUM_GPUS get_dataset_temperature.py \
        --data-path $DATASET \
        --dataset-split $SPLIT \
        --dataset-field "${FIELD[@]}" \
        --save-path $SAVE_PATH \
        --model $MODEL_PATH \
        --tokenizer $TOKENIZER \
        --loss-type $LOSS_TYPE \
        --bs $BATCH_SIZE \
        --max-seq-length $MAX_SEQ_LEN \
        --device "cuda" \
        # --run-dir $RUN_DIR \
else
    echo "=== Starting Analysis ==="
    python get_dataset_temperature.py \
        --data-path $DATASET \
        --dataset-split $SPLIT \
        --dataset-field "${FIELD[@]}" \
        --save-path $SAVE_PATH \
        --model $MODEL_PATH \
        --tokenizer $TOKENIZER \
        --loss-type $LOSS_TYPE \
        --bs $BATCH_SIZE \
        --max-seq-length $MAX_SEQ_LEN \
        --device "cuda" \
        # --run-dir $RUN_DIR \
fi