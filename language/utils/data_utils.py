import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk,Dataset
import transformers
from typing import Any, Dict, List, Optional, Union, Sequence
import copy
from dataclasses import dataclass
import json 

IGNORE_INDEX = -100
# PROMPT = (
#     "Below is an instruction that describes a task. "
#     "Write a response that appropriately completes the request.\n\n"
#     "### Instruction:\n{instruction}\n\n### Response:"
# )
PROMPT = (
    "Q: {instruction}\nA: Let's think step by step."
)


def load_and_preprocess_data(task, tokenizer, args):

    if "mnli" in task:
        dataset = load_dataset("glue", "mnli")
    else:
        dataset = load_dataset("glue", task)

    def tokenize_function(examples):

        # Handle different input formats
        if "premise" in examples and "hypothesis" in examples:
            # MNLI and similar tasks
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "question" in examples and "sentence" in examples:
            # QNLI and similar tasks
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "sentence1" in examples and "sentence2" in examples:
            # MRPC, STS-B
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "question1" in examples and "question2" in examples:
            # QQP
            return tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "sentence" in examples:
            # CoLA, SST-2
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        else:
            raise ValueError(f"Unexpected format for task {task}")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    if task == "cola":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "sst2":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "mrpc":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    elif task == "qqp":
        tokenized_datasets = tokenized_datasets.remove_columns(["question1", "question2", "idx"])
    elif task == "stsb":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    elif task == "qnli":
        tokenized_datasets = tokenized_datasets.remove_columns(["question", "sentence", "idx"])
    elif task == "rte":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    elif task == "wnli":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    elif task == "mnli_matched" or task == "mnli_mismatched" or task == "mnli":
        tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis", "idx"])
    else:
        raise ValueError(f"Unexpected task {task}")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    if (
        task == "cola"
        or task == "sst2"
        or task == "mrpc"
        or task == "qqp"
        or task == "stsb"
        or task == "qnli"
        or task == "rte"
        or task == "wnli"
    ):
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"]
    elif task == "mnli_matched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_matched"]
        test_dataset = tokenized_datasets["test_matched"]
    elif task == "mnli_mismatched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_mismatched"]
        test_dataset = tokenized_datasets["test_mismatched"]

    return train_dataset, val_dataset, test_dataset


def create_dataloader(dataset, args, shuffle=True):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class IndexedDataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        index = [instance["index"] for instance in instances]
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), index=index)

@dataclass
class IndexedDPODataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_to_length(self, tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
        if tensor.size(dim) >= length:
            return tensor
        else:
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            return torch.cat(
                [
                    tensor,
                    pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                ],
                dim=dim,
            )

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        padding_value = self.tokenizer.pad_token_id
        label_pad_token_id = IGNORE_INDEX
        rejected_max_len = max([len(x) for x in batch["rejected_input_ids"]])
        max_length = max(len(batch["chosen_input_ids"]), rejected_max_len)
        concatenated_batch = {"index" : batch["index"]}
        for k in batch:
            if k.startswith("chosen") and (k.endswith("_input_ids") or k.endswith("_labels")):
                pad_value = label_pad_token_id if "labels" in k else padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = self.pad_to_length(torch.tensor(batch[k]), max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and (k.endswith("_input_ids") or k.endswith("_labels")):
                pad_value = label_pad_token_id if "labels" in k else padding_value
                # concatenated_key = k.replace("rejected", "concatenated")
                prefix = k.split("_")[0]
                concatenated_key = "concatenated" + k[len(prefix):]
                padded_tensor_of_rejected = torch.stack([
                    self.pad_to_length(torch.tensor(batch[k][idx]), max_length, pad_value=pad_value)
                    for idx in range(len(batch[k]))
                ],dim=0)
                
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key][None,:],
                        padded_tensor_of_rejected,
                    ),
                    dim=0,
                )
        return concatenated_batch

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [self.concatenated_inputs(instance) for instance in instances]
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("concatenated_input_ids", "concatenated_labels"))
        index = [instance["index"] for instance in instances]
        # input_ids = [torch.tensor(x) for x in input_ids]
        max_length = max([len(x[0]) for x in input_ids]) # each entry of x has same length.  
        input_ids = torch.cat([self.pad_to_length(input_id,max_length,self.tokenizer.pad_token_id) for input_id in input_ids],dim=0,)
        labels = torch.cat([self.pad_to_length(label,max_length,self.tokenizer.pad_token_id) for label in labels],dim=0,)
        # labels = [torch.tensor(x) for x in labels]
        # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), index=index)


@dataclass
class WeightedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):

    loss_type: str = "both"

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)

        if self.loss_type == "both":
            token_weights = [torch.tensor(instance["token_weights"]) for instance in instances]
            token_weights = torch.nn.utils.rnn.pad_sequence(token_weights, batch_first=True, padding_value=0)
            batch["token_weights"] = token_weights
            batch["sequence_weights"] = torch.tensor([instance["sequence_weights"] for instance in instances])
            batch["ref_logprobs"] = torch.tensor([instance["ref_logprobs"] for instance in instances])
        elif self.loss_type == "ref_logprobs":
            batch["ref_logprobs"] = torch.tensor([instance["ref_logprobs"] for instance in instances])
        elif self.loss_type == "sequence":
            batch["sequence_weights"] = torch.tensor([instance["sequence_weights"] for instance in instances])
        else:  # Assume Token
            token_weights = [torch.tensor(instance["token_weights"]) for instance in instances]
            token_weights = torch.nn.utils.rnn.pad_sequence(token_weights, batch_first=True, padding_value=0)
            batch["token_weights"] = token_weights
        return batch


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def train_tokenize_function_dpo(examples, tokenizer, query, chosen, rejected):
    queries = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    chosen_targets = [f"{output}{tokenizer.eos_token}" for output in examples[chosen]]
    data_chosen_dict = preprocess(queries, chosen_targets, tokenizer)
    
    # print ('in train tokenize ',examples)
    
    num_negs = len(examples[rejected][0])
    data_rejected_dict_list = []
    for idx in range(num_negs):
        rejected_targets = [f"{output[idx]}{tokenizer.eos_token}" for output in examples[rejected]]
        data_rejected_dict_list.append(preprocess(queries, rejected_targets, tokenizer))

    rejected_input_ids = [data_rejected_dict['input_ids'] for data_rejected_dict in data_rejected_dict_list]
    rejected_labels = [data_rejected_dict['labels'] for data_rejected_dict in data_rejected_dict_list]
    transpose = lambda list_ : [[x[idx] for x in list_] for idx in range(len(list_[0]))]
    rejected_input_ids = transpose(rejected_input_ids)
    rejected_labels = transpose(rejected_labels)

    data_dict = dict(chosen_input_ids=data_chosen_dict['input_ids'],
                     chosen_labels=data_chosen_dict['labels'],
                     rejected_input_ids=rejected_input_ids,
                     rejected_labels=rejected_labels,
    )
    return data_dict


def load_and_preprocess_it(tokenizer, args):
    path = args.data_path
    split_range = None
    if "[" in path and path.endswith("]"):
        main_path, range_str = path.split("[", 1)
        range_str = range_str.rstrip("]")  # Remove the closing bracket
        path = main_path  # Update path to the main dataset path

        if range_str.startswith(":"):
            start = 0
            end = int(range_str[1:])
            split_range = (start, end)
        elif range_str.endswith(":"):
            start = int(range_str[:-1])
            split_range = (start, None)  # None indicates till the end
        elif ":" in range_str:
            start, end = range_str.split(":")
            start = int(start) if start else 0
            end = int(end) if end else None
            split_range = (start, end)

    raw_train_datasets = load_dataset(path, split=args.dataset_split)

    if split_range is not None:
        start, end = split_range
        if end is None:
            raw_train_datasets = raw_train_datasets.select(range(start, len(raw_train_datasets)))
        else:
            raw_train_datasets = raw_train_datasets.select(range(start, end))

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": args.dataset_field[0], "response": args.dataset_field[1]},
    )

    return train_dataset


def load_and_preprocess_dpo(tokenizer,args):
    path = args.data_path
    split_range = None
    if "[" in path and path.endswith("]"):
        main_path, range_str = path.split("[", 1)
        range_str = range_str.rstrip("]")  # Remove the closing bracket
        path = main_path  # Update path to the main dataset path

        if range_str.startswith(":"):
            start = 0
            end = int(range_str[1:])
            split_range = (start, end)
        elif range_str.endswith(":"):
            start = int(range_str[:-1])
            split_range = (start, None)  # None indicates till the end
        elif ":" in range_str:
            start, end = range_str.split(":")
            start = int(start) if start else 0
            end = int(end) if end else None 
            split_range = (start, end)

    with open(path,'r') as f:
        data = f.readlines()
    json_data = [json.loads(x) for x in data]
    raw_train_datasets = dict(query=[x['doc'][args.dataset_field[0]] for x in json_data],
                              chosen=[x['doc'][args.dataset_field[1]] for x in json_data],
                              rejected=[x['resps'][0] for x in json_data],
                              )
    raw_train_datasets = Dataset.from_dict(raw_train_datasets)


    if split_range is not None:
        start, end = split_range
        if end is None:
            raw_train_datasets = raw_train_datasets.select(range(start, len(raw_train_datasets)))
        else:
            raw_train_datasets = raw_train_datasets.select(range(start, end))

    train_dataset = raw_train_datasets.map(
        train_tokenize_function_dpo,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, 
                   "query": "query", 
                   "chosen": "chosen", 
                   "rejected":"rejected"}
    )

    return train_dataset

def load_weighted_it(args):
    path = args.data_path
    split_range = None
    if "[" in path and path.endswith("]"):
        main_path, range_str = path.split("[", 1)
        range_str = range_str.rstrip("]")  # Remove the closing bracket
        path = main_path  # Update path to the main dataset path

        if range_str.startswith(":"):
            start = 0
            end = int(range_str[1:])
            split_range = (start, end)
        elif range_str.endswith(":"):
            start = int(range_str[:-1])
            split_range = (start, None)  # None indicates till the end
        elif ":" in range_str:
            start, end = range_str.split(":")
            start = int(start) if start else 0
            end = int(end) if end else None
            split_range = (start, end)

    # Load the dataset from disk
    dataset = load_from_disk(path)

    # If split_range is provided, apply slicing
    if split_range is not None:
        start, end = split_range
        if end is None:
            dataset = dataset.select(range(start, len(dataset)))
        else:
            dataset = dataset.select(range(start, end))

    return dataset
    # weighted_train_dataset = load_from_disk(args.data_path)

    # return weighted_train_dataset


def load_and_preprocess_cr(tokenizer, args):
    """Load and preprocess the dataset."""
    if args.data_path.endswith(".json"):
        data = load_dataset("json", data_files=args.data_path)
    else:
        data = load_dataset(args.data_path)

    # Create a wrapper function that includes all necessary arguments
    def generate_and_tokenize_prompt_wrapper(data_point):
        return generate_and_tokenize_prompt_cr(data_point, tokenizer, args)

    train_dataset = (
        data["train"]
        .shuffle()
        .map(generate_and_tokenize_prompt_wrapper, num_proc=8, remove_columns=data["train"].column_names)  # Remove original columns
    )

    return train_dataset


def generate_prompt_cr(data_point):
    """Generate prompt from data point."""
    if data_point.get("input", ""):  # Using get() with default empty string
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {data_point["instruction"]}

            ### Input:
            {data_point["input"]}

            ### Response:
            {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            {data_point["instruction"]}

            ### Response:
            {data_point["output"]}"""


def tokenize_cr(prompt, tokenizer, args, add_eos_token=True):
    """Tokenize the prompt."""
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.max_seq_length,
        padding=False,
        return_tensors=None,
    )

    if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < args.max_seq_length and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        if "chatglm" not in args.model:
            result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    if "chatglm" in args.model:
        return {"input_ids": result["input_ids"], "labels": result["labels"]}
    else:
        return result


def generate_and_tokenize_prompt_cr(data_point, tokenizer, args):
    """Generate and tokenize prompt with proper labels."""
    full_prompt = generate_prompt_cr(data_point)
    tokenized_full_prompt = tokenize_cr(full_prompt, tokenizer, args)

    if not args.train_on_inputs:
        # Create a user prompt without the response
        user_prompt = generate_prompt_cr({"instruction": data_point["instruction"], "input": data_point.get("input", ""), "output": ""})
        tokenized_user_prompt = tokenize_cr(user_prompt, tokenizer, args, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        # Replace labels with -100 for non-response tokens
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

    return tokenized_full_prompt