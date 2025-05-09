#!/bin/bash
#SBATCH -J FT-LM # Job name
#SBATCH -o slurmlogs/FT-Llama.o%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/FT-Llama.e%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gh                        # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 36:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name

# Source conda environment and move to project directory
source $WORK/miniconda3/bin/activate flow
cd $WORK/flow/FLOW_finetuning/language

# NOTE: This script expects a single NODE to be used for analysis
# Job Configuration
export CUDA_VISIBLE_DEVICES='0' # Specify which GPU to use

# ===== Model Configuration ===== #
BASE_MODEL="google/gemma-2-2b"
FINETUNE_TYPE="full" # none, full, lora, linear-probe, sfa (none will freeze the entire model, which is useful for creating a baseline model)

# ~~ LORA Configuration ~~ #
LORA_RANK=64
LORA_DROPOUT=0.0

# ~~ SFA Configuration ~~ #
SFA_UPDATE_FREQ=0.25
SFA_BETA=0.5

# ===== Data Configuration ===== #
DATASETS=("$WORK/flow/FLOW_finetuning/language/data/samples_metamath_generate_2025-03-23T00-48-09.227201.jsonl/20250323_022404_samples_metamath_generate_2025-03-23T00-48-09.227201.jsonl__gemma2-2b__tp0.0643_l_both")
# "meta-math/MetaMathQA[:50000]"
# "/u/pbansal/flow/FLOW_finetuning/language/data/MetaMathQA/20250216_154342_MetaMathQA__llama3.2__tp0.0643_l_both" \
# "/u/pbansal/flow/FLOW_finetuning/language/data/MetaMathQA/20250216_154342_MetaMathQA__llama3.2__tp0.0643_l_both") # Dataset to use (expects huggingface format or path)
SPLIT="train" # Split to use (expects huggingface format)
FIELD=("query" "response") # Field to use (expects list of two strings)

# ===== Reweighting Variables ===== #
REWEIGHTING_TYPES=("ref_logprobs")
# "ref_logprobs" "ref_logprobs") # Reweighting scheme either [sequence/token/ref_logprobs/none] (default trainer: none)

# ===== Regularization Configuration ===== #
# Note: Regularization only makes sense when using (full/linear-probe) finetuning
REGULARIZATION_LAMBDA=1e-3 # Regularization lambda to use
REGULARIZATION_TYPE="none" # Regularization type to use (none/l1/l2)
BETAS=(1e-3) # 1e-2 1e-3) # Sigmoid beta to use

# ===== Training Configuration ===== #
EPOCHS=1 # Number of epochs to train for
SCHEDULER="cosine" # Scheduler to use
WEIGHT_DECAY=0.00 # Weight decay to use
WARMUP_RATIO=0.03 # Warmup ratio to use
LR=2e-5 # Learning rate to use
BATCH_SIZE=1 # Batch size to use
MAX_SEQ_LEN=1024 # Maximum sequence length to use
GRADIENT_ACCUMULATION_STEPS=1 # Gradient accumulation steps to use
GRADIENT_CHECKPOINTING=false # Use gradient checkpointing to reduce memory usage
SEED=42 # Seed to use
# NOTE: If using Accelerate config, set the ACCELERATE_CONFIG variable other wise leave it as "" and DDP will be used in multi-gpu training
# ACCELERATE_CONFIG="configs/ddp.yaml" # Accelerate configuration file to use (expects path)
# NOTE: Accelerate hasn't been tested in a bit, aviod for now

# ===== Experiment Configuration ===== #
BASE_DIR="$WORK/flow/floss_results/experiments" # Base directory for experiments
WANDB_PROJECT="flow" # Wandb project to use
WAND_RUN="debug" # Wandb run name to use

# ===== Misc Configuration ===== #
DEBUG=false # true or false (Will turn off distributed training)
USE_SLURM=false # true or false (Will use SLURM for distributed training)

#### ~~~~~~~~ Do not modify below this line ~~~~~~~~ ####
MODEL_NAME=$(basename "$BASE_MODEL")

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
    NO_WANDB=false
else
    echo "Running in Debug Mode: Setting CUDA_VISIBLE_DEVICES to 1, NUM_GPUS to 1, and disabling wandb logging"
    export CUDA_VISIBLE_DEVICES=0
    NUM_GPUS=1
    NO_WANDB=true
fi


for i in "${!REWEIGHTING_TYPES[@]}"; do
    BETA="${BETAS[$i]}"
    REWEIGHTING_TYPE="${REWEIGHTING_TYPES[$i]}"
    DATASET="${DATASETS[$i]}"
    if [ "$DEBUG" = false ] && [ "$NUM_GPUS" -gt 1 ]; then
        echo "Master Address: $MASTER_ADDR"
        echo "Master Port: $MASTER_PORT"

        if [ -z "$ACCELERATE_CONFIG" ]; then
            echo "=== Starting Distributed Training ==="
            torchrun --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=${NUM_GPUS} train_dpo_arithmetic.py \
                --base-dir $BASE_DIR \
                --data_path $DATASET \
                --dataset_split $SPLIT \
                --dataset_field "${FIELD[@]}" \
                --model $BASE_MODEL \
                --reweight-type $REWEIGHTING_TYPE \
                --weight-regularization $REGULARIZATION_TYPE \
                --weight-regularization-lambda $REGULARIZATION_LAMBDA \
                --beta $BETA \
                --finetune-type $FINETUNE_TYPE \
                --lora_r $LORA_RANK \
                --lora_alpha $LORA_RANK \
                --lora_dropout $LORA_DROPOUT \
                --sfa-update-freq $SFA_UPDATE_FREQ \
                --sfa-beta $SFA_BETA \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS \
                --scheduler $SCHEDULER \
                --weight_decay $WEIGHT_DECAY \
                --warmup_ratio $WARMUP_RATIO \
                --max_seq_length $MAX_SEQ_LEN \
                --lr $LR \
                --seed $SEED \
                --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                --gradient-checkpointing $GRADIENT_CHECKPOINTING \
                --device "cuda" \
                --project-name $WANDB_PROJECT \
                --no-wandb $NO_WANDB \
                --run-name $WAND_RUN
        else
            echo "=== Starting Accelerated Distributed Training ==="
            accelerate launch --config_file $ACCELERATE_CONFIG train_dpo_arithmetic.py \
                --base-dir $BASE_DIR \
                --data_path $DATASET \
                --dataset_split $SPLIT \
                --dataset_field "${FIELD[@]}" \
                --model $BASE_MODEL \
                --reweight-type $REWEIGHTING_TYPE \
                --weight-regularization $REGULARIZATION_TYPE \
                --weight-regularization-lambda $REGULARIZATION_LAMBDA \
                --beta $BETA \
                --finetune-type $FINETUNE_TYPE \
                --lora_r $LORA_RANK \
                --lora_alpha $LORA_RANK \
                --lora_dropout $LORA_DROPOUT \
                --sfa-update-freq $SFA_UPDATE_FREQ \
                --sfa-beta $SFA_BETA \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS \
                --scheduler $SCHEDULER \
                --weight_decay $WEIGHT_DECAY \
                --warmup_ratio $WARMUP_RATIO \
                --max_seq_length $MAX_SEQ_LEN \
                --lr $LR \
                --seed $SEED \
                --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                --gradient-checkpointing $GRADIENT_CHECKPOINTING \
                --device "cuda" \
                --project-name $WANDB_PROJECT \
                --no-wandb $NO_WANDB \
                --run-name $WAND_RUN
        fi
    else
        if [ -z "$ACCELERATE_CONFIG" ]; then
            echo "=== Starting Training ==="
            python train_dpo_arithmetic.py \
                --base-dir $BASE_DIR \
                --data_path $DATASET \
                --dataset_split $SPLIT \
                --dataset_field "${FIELD[@]}" \
                --model $BASE_MODEL \
                --reweight-type $REWEIGHTING_TYPE \
                --weight-regularization $REGULARIZATION_TYPE \
                --weight-regularization-lambda $REGULARIZATION_LAMBDA \
                --beta $BETA \
                --finetune-type $FINETUNE_TYPE \
                --lora_r $LORA_RANK \
                --lora_alpha $LORA_RANK \
                --lora_dropout $LORA_DROPOUT \
                --sfa-update-freq $SFA_UPDATE_FREQ \
                --sfa-beta $SFA_BETA \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS \
                --scheduler $SCHEDULER \
                --weight_decay $WEIGHT_DECAY \
                --warmup_ratio $WARMUP_RATIO \
                --max_seq_length $MAX_SEQ_LEN \
                --lr $LR \
                --seed $SEED \
                --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                --gradient-checkpointing $GRADIENT_CHECKPOINTING \
                --device "cuda" \
                --project-name $WANDB_PROJECT \
                --no-wandb $NO_WANDB \
                --run-name $WAND_RUN
        else
            echo "=== Starting Accelerated Distributed Training ==="
            accelerate launch --config_file $ACCELERATE_CONFIG train_dpo_arithmetic.py \
                --base-dir $BASE_DIR \
                --data_path $DATASET \
                --dataset_split $SPLIT \
                --dataset_field "${FIELD[@]}" \
                --model $BASE_MODEL \
                --reweight-type $REWEIGHTING_TYPE \
                --weight-regularization $REGULARIZATION_TYPE \
                --weight-regularization-lambda $REGULARIZATION_LAMBDA \
                --beta $BETA \
                --finetune-type $FINETUNE_TYPE \
                --lora_r $LORA_RANK \
                --lora_alpha $LORA_RANK \
                --lora_dropout $LORA_DROPOUT \
                --sfa-update-freq $SFA_UPDATE_FREQ \
                --sfa-beta $SFA_BETA \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS \
                --scheduler $SCHEDULER \
                --weight_decay $WEIGHT_DECAY \
                --warmup_ratio $WARMUP_RATIO \
                --max_seq_length $MAX_SEQ_LEN \
                --lr $LR \
                --seed $SEED \
                --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                --gradient-checkpointing $GRADIENT_CHECKPOINTING \
                --device "cuda" \
                --project-name $WANDB_PROJECT \
                --no-wandb $NO_WANDB \
                --run-name $WAND_RUN
        fi
    fi
done
