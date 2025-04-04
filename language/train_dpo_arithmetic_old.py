import os
import torch
import re
import random

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
# from trl import DPOTrainer
from utils.softmax_dpo_trainer import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# from utils import find_all_linear_names, print_trainable_parameters
from transformers import LlamaForCausalLM, LlamaTokenizer

from Prompt import Prompt

import torch
import bitsandbytes as bnb
from accelerate import Accelerator
import fire

random.seed(1958)
def train(
    #train
    output_dir="",
    logging_dir="",
    model_name ="",
    prompt_path = "",
    dataset="",
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter
    # wandb config
    wandb_project: str = "",
    wandb_name: str = "",   # the name of the wandb run
    # training hyperparameters
    beta: float = 0.1,
    neg_num: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    eval_step = 1,  
):
    
    data_files = {
        "train": "../data/lastfm-sft-cans20/lastfm-train.json",
        "validation": "../data/lastfm-sft-cans20/lastfm-val.json",
    }



    # random 2000 samples for validation
    val_data = data["validation"].map(process_data, remove_columns=columns, \
                                        num_proc=8, batched=True).shuffle(seed=42)
    if val_data.num_rows > 2000:
        val_data = val_data.select(range(2000))
    
    print(val_data)

    device_index = Accelerator().process_index
    device_map = {"": device_index}
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


    base_model = LlamaForCausalLM.from_pretrained(model_name, 
                                                device_map=device_map, 
                                                # load_in_8bit=True,
                                                # torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint, 
                                        is_trainable=True)
    # print_trainable_parameters(base_model)
    base_model.print_trainable_parameters()

    model_ref = LlamaForCausalLM.from_pretrained(model_name,
                                                device_map=device_map, 
                                                # load_in_8bit=True,
                                                # torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)
    reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
    reference_model.print_trainable_parameters()


    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=num_train_epochs, 
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=eval_step,
        save_total_limit=100,
        evaluation_strategy="steps",
        eval_steps=eval_step,
        load_best_model_at_end=True,
        logging_steps=1,
        output_dir=output_dir,
        report_to = "wandb",
        run_name = wandb_name,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        ddp_find_unused_parameters=False,
    )

    dpo_trainer = DPOTrainer(
        base_model,
        reference_model,
        args=training_args,
        beta=beta,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
    )


    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)


    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)