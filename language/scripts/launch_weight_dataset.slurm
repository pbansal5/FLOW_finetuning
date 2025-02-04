#!/bin/bash
#SBATCH -J Weight-Dataset # Job name
#SBATCH -o slurmlogs/Analyze-LP.o%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/Analyze-LP.e%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gh                 # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name

# Source conda environment and move to project directory

source $WORK/miniconda3/install/bin/activate flow
cd /work/09753/hprairie/ls6/projects/FLOW_finetuning/language

# NOTE: This script expects a single NODE to be used for analysis
# Job Configuration

export CUDA_VISIBLE_DEVICES='0' # Specify which GPU to use

DATASET="meta-math/MetaMathQA" # CHANGE ME
SPLIT="train" # CHANGE ME
FIELD=("query" "response") # CHANGE ME

MODEL="experiments/instruction_tuning/gemma-2-2b/20250109_131557_gemma-2-2b__r64__lr2e-05__train_train/final_model" # Path to rewighter model (HF/Local)
TOKENIZER="experiments/instruction_tuning/gemma-2-2b/20250109_131557_gemma-2-2b__r64__lr2e-05__train_train/tokenizer" # Path to tokenizer (HF/Local)

BASE_DIR="data" # The base directory for the reweighted dataset
RUN_NAME="Gemma-2-2b-Base" # An additional tag to add to the run name

LOSS_TYPE="both" # both, sequence, or token
# TEMPERATURE=0.31209245 # Temperature to use for token weighting
# TEMPERATURE=0.91060877 # Temperature to use for token weighting
# TEMPERATURE=1.2035236 # Temperature to use for token weighting
TEMPERATURE=0.0643
# TEMPERATURE=0.77839446

BATCH_SIZE=8 # Adjust as needed based on memory constraints
MAX_SEQ_LEN=1024 # Be sure to be consistent through experiments
DEBUG=false # true or false (Will turn off distributed training)
USE_SLURM=true # true or false (Will use SLURM for distributed training)
VALIDATE_DATESET=true # true or false (Will validate each sample of the dataset)

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

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NUM_GPUS: $NUM_GPUS"


if [ "$DEBUG" = false ] && [ $NUM_GPUS -gt 1 ]; then
    echo "Master Address: $MASTER_ADDR"
    echo "Master Port: $MASTER_PORT"

    echo "=== Starting Distributed Data Weighting ==="
    torchrun --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=$NUM_GPUS create_weighted_dataset.py \
        --data-path $DATASET \
        --dataset-split $SPLIT \
        --dataset-field "${FIELD[@]}" \
        --base-dir $BASE_DIR \
        --run-name $RUN_NAME \
        --model $MODEL \
        --tokenizer $TOKENIZER \
        --temperature $TEMPERATURE \
        --loss-type $LOSS_TYPE \
        --bs $BATCH_SIZE \
        --max-seq-length $MAX_SEQ_LEN \
        --validate-dataset $VALIDATE_DATESET \
        --device "cuda"
else
    echo "=== Starting Data Weighting ==="
    python create_weighted_dataset.py \
        --data-path $DATASET \
        --dataset-split $SPLIT \
        --dataset-field "${FIELD[@]}" \
        --base-dir $BASE_DIR \
        --run-name $RUN_NAME \
        --model $MODEL \
        --tokenizer $TOKENIZER \
        --temperature $TEMPERATURE \
        --loss-type $LOSS_TYPE \
        --bs $BATCH_SIZE \
        --max-seq-length $MAX_SEQ_LEN \
        --validate-dataset $VALIDATE_DATESET \
        --device "cuda"
fi