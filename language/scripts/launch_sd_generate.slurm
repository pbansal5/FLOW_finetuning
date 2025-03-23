#!/bin/bash
#SBATCH -J Eval-LM # Job name
#SBATCH -o slurmlogs/Eval-LM.o%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/Eval-LM.e%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gh-dev                       # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 2:00:00                        # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
##SBATCH -A MLL                            # Allocation name

# Source conda environment and move to project directory (change to your own path)

source $WORK/miniconda3/bin/activate flow
cd $WORK/flow/FLOW_finetuning/language

# NOTE: This script expects a single NODE to be used for analysis
# Set Run Parameters

# NOTE: Each GPU will run a seperate evalutation
export CUDA_VISIBLE_DEVICES='0' # Specify which GPUs to use
export HF_ALLOW_CODE_EVAL="1" # Allow eval

# RUN_DIRS is a list to all folders with runs
EXPT_DIR="$WORK/flow/floss_results/experiments"
SD_DIR="$WORK/flow/floss_results/synthetic_data"
RUN_DIRS=("gemma-2-2b/20250323_055036_gemma-2-2b__r64__lr2e-05__ft_full__rw_none_beta_0.0_reg_none")
# RUN_DIRS=("gemma-2-2b/20250323_055622_gemma-2-2b__r64__lr2e-05__ft_none__rw_none_beta_0.0_reg_none")
# If model being evaluated used LoRA then set to true to merge model and LoRA weights (i.e, false/true)
# Use false if not using LoRA
NEED_TO_MERGE=(false)
# If model being evaluated used LoRA then name the HuggingFace model path to merge weights (i.e., ""/"google/gemma-2-2b")
# NOTE: Use "" if not using LoRA
BASE_MODELS=("google/gemma-2-2b")
TASKS="metamath_generate"
GEN_KWARGS="do_sample=true,temperature=0.7,top_p=0.9"
# GEN_KWARGS="do_sample=false,temperature=0"
NUM_SAMPLES=5000

BATCH_SIZE=64
METHOD="normal" # (Type of model merging done) Options: normal
WANDB_PROJECT="flow"

#### ~~~~~~~~ Do not modify below this line ~~~~~~~~ ####

# Calculate NUM_GPUS by counting commas and adding 1
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=0
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
fi

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"

# Add array length validation at start
if [ ${#RUN_DIRS[@]} -ne ${#NEED_TO_MERGE[@]} ] || [ ${#RUN_DIRS[@]} -ne ${#BASE_MODELS[@]} ]; then
    echo "Error: Arrays RUN_DIRS, NEED_TO_MERGE, and BASE_MODELS must have same length"
    exit 1
fi

for i in "${!RUN_DIRS[@]}"; do
    RUN_DIR="$EXPT_DIR/${RUN_DIRS[$i]}"
    GPU_IDX=$((i % NUM_GPUS))
    GPU_ID="${GPU_ARRAY[$GPU_IDX]}"
    MERGE="${NEED_TO_MERGE[$i]}"
    
    # Get the paths for this run
    FINAL_MODEL_PATH="$RUN_DIR/final_model"
    FINAL_TOKENIZER_PATH="$RUN_DIR/tokenizer"
    MERGED_MODEL_PATH="$RUN_DIR/merged_model" # Only used if MERGE is true
    DUMP_PATH="$SD_DIR/$TASKS"
    BASE_MODEL="${BASE_MODELS[$i]}"


    if [ "$MERGE" = true ]; then
        echo "=== Starting Merging ==="
        if [ ! -d "$FINAL_MODEL_PATH" ]; then
            echo "Error: Final model not found at $FINAL_MODEL_PATH"
            exit 1
        fi

        echo "Using merge_adapter_to_base_model_normal for method: $METHOD"
        MERGE_SCRIPT="utils.merge_adapter_to_base_model_normal"

        CUDA_VISIBLE_DEVICES=$GPU_ID python -m $MERGE_SCRIPT \
            --base_model $BASE_MODEL \
            --adapter "$FINAL_MODEL_PATH" \
            --output_path "$MERGED_MODEL_PATH" \
            --device cuda
    fi
    
    
    echo "=== Starting Evaluation #$((i+1)) on $RUN_DIR with GPU $GPU_ID ==="

    if [ "$MERGE" = true ] && [ ! -d "$MERGED_MODEL_PATH" ]; then
        echo "Error: Merged model not found at $MERGED_MODEL_PATH"
        exit 1
    fi

    if [ "$MERGE" = true ]; then
        FINAL_MODEL_PATH="$MERGED_MODEL_PATH"
    fi

    # CUDA_VISIBLE_DEVICES=$GPU_ID lm_eval --model vllm \
    #     --model_args pretrained=$FINAL_MODEL_PATH,tokenizer=$FINAL_TOKENIZER_PATH,gpu_memory_utilization=0.8,max_model_len=4096 \
    CUDA_VISIBLE_DEVICES=$GPU_ID lm_eval --model hf \
        --model_args pretrained=$FINAL_MODEL_PATH,tokenizer=$FINAL_TOKENIZER_PATH,max_length=4096,trust_remote_code=True \
        --tasks $TASKS \
        --output_path $DUMP_PATH \
        --trust_remote_code \
        --confirm_run_unsafe_code \
        --cache_requests refresh\
        --log_samples \
        --limit $NUM_SAMPLES \
        --gen_kwargs $GEN_KWARGS\
        --batch_size $BATCH_SIZE &
    
    # Wait for evaluation to finish if using all GPUs
    if [ $((GPU_IDX + 1)) -eq $NUM_GPUS ]; then
        wait
    fi
done

wait
echo "All Evaluations done!"

for i in "${!RUN_DIRS[@]}"; do
    RUN_DIR="${RUN_DIRS[$i]}"
    MERGE="${NEED_TO_MERGE[$i]}"

    MERGED_MODEL_PATH="$RUN_DIR/merged_model" # Only used if MERGE is true

    if [ "$MERGE" = true ] && [ -d "$MERGED_MODEL_PATH" ]; then
        echo "=== Cleaning up merged model ==="
        echo "Removing merged model directory: $MERGED_MODEL_PATH"
        rm -rf "$MERGED_MODEL_PATH"
        if [ $? -eq 0 ]; then
            echo "Successfully removed merged model directory"
        else
            echo "Warning: Failed to remove merged model directory"
        fi
    else
        echo "Merged model directory not found - nothing to clean up"
    fi
done

echo "Job cleaned up and finished!"
