import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase, TrainerCallback
from utils.data_utils import IGNORE_INDEX

import transformers

@dataclass
class DPODataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    # def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    #     chosen_input_ids, chosen_labels = tuple([instance[key] for instance in instances] for key in ("chosen_input_ids", "chosen_labels"))
    #     rejected_input_ids, rejected_labels = tuple([instance[key] for instance in instances] for key in ("rejected_input_ids", "rejected_labels"))
        
    #     input_ids = [torch.tensor(x) for x in input_ids]
    #     input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    #     labels = [torch.tensor(x) for x in labels]
    #     labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

    #     max_length = max([len(x) for x in chosen_input_ids]) # each entry of x has same length.  
    #     input_ids = torch.cat([self.pad_to_length(input_id,max_length,self.tokenizer.pad_token_id) for input_id in input_ids],dim=0,)
    #     labels = torch.cat([self.pad_to_length(label,max_length,self.tokenizer.pad_token_id) for label in labels],dim=0,)

    #     max_length = max([len(x[0]) for x in input_ids]) # each entry of x has same length.  
    #     chosen_input_ids = torch.cat([self.pad_to_length(input_id,max_length,self.tokenizer.pad_token_id) for input_id in input_ids],dim=0,)
    #     labels = torch.cat([self.pad_to_length(label,max_length,self.tokenizer.pad_token_id) for label in labels],dim=0,)

    #     batch = dict(
    #         chosen_input_ids=chosen_input_ids,
    #         chosen_labels=chosen_labels,
    #         chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
    #     )
    #     batch["chosen_ref_logprobs"] = torch.tensor([instance["chosen_ref_logprobs"] for instance in instances])
    #     batch["rejected_ref_logprobs"] = torch.tensor([instance["rejected_ref_logprobs"] for instance in instances])
    #     return batch 



    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_labels"):
                # adapted from https://stackoverflow.com/questions/73256206
                if "prompt" in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = IGNORE_INDEX
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                if "prompt" in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch



    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        # for feature in features:
        #     prompt = feature["chosen_input_ids"]
        #     chosen = feature["chosen"]
        #     rejected = {}
        #     for key in feature:
        #         if key.startswith("rejected"):
        #             rejected[key] = feature[key]

        #     batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
        #     tokenized_batch.append(batch_element)

        tokenized_batch = features
        # return collated batch
        return self.collate(tokenized_batch)




def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
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



# {"doc_id": 5, "doc": {"type": "GSM_FOBAR", "query": "Mark is buying asphalt to pave a new section of road. The road will be 2000 feet long and 20 feet wide. Each truckload of asphalt will cover 800 square feet of road. If each truckload costs x, and there's a 20% sales tax, how much will Mark need to pay for asphalt?\nIf we know the answer to the above question is 4500, what is the value of unknown variable x?", "original_question": "Mark is buying asphalt to pave a new section of road. The road will be 2000 feet long and 20 feet wide. Each truckload of asphalt will cover 800 square feet of road. If each truckload costs $75, and there's a 20% sales tax, how much will Mark need to pay for asphalt?", "response": "The area of the road is the length multiplied by the width: 2000 * 20 = 40,000 square feet.\nEach truckload of asphalt will cover 800 square feet of road, so Mark will need 40,000 / 800 = 50 truckloads of asphalt.\nThe cost of each truckload is x dollars.\nThe total cost of the asphalt without sales tax is 50 * x dollars.\nThe sales tax is 20% of the total cost, so the sales tax amount is 0.2 * (50 * x) = 10 * x dollars.\nThe total cost including sales tax is the sum of the cost without sales tax and the sales tax amount: 50 * x + 10 * x = 60 * x dollars.\nWe are given that the total cost is $4500, so we can write: 60 * x = $4500.\nDividing both sides by 60, we get: x = $75.\nThe value of x is $75.\n#### 75\nThe answer is: 75"}, "target": "The area of the road is the length multiplied by the width: 2000 * 20 = 40,000 square feet.\nEach truckload of asphalt will cover 800 square feet of road, so Mark will need 40,000 / 800 = 50 truckloads of asphalt.\nThe cost of each truckload is x dollars.\nThe total cost of the asphalt without sales tax is 50 * x dollars.\nThe sales tax is 20% of the total cost, so the sales tax amount is 0.2 * (50 * x) = 10 * x dollars.\nThe total cost including sales tax is the sum of the cost without sales tax and the sales tax amount: 50 * x + 10 * x = 60 * x dollars.\nWe are given that the total cost is $4500, so we can write: 60 * x = $4500.\nDividing both sides by 60, we get: x = $75.\nThe value of x is $75.\n#### 75\nThe answer is: 75", "arguments": {"gen_args_0": {"arg_0": "Q: Mark is buying asphalt to pave a new section of road. The road will be 2000 feet long and 20 feet wide. Each truckload of asphalt will cover 800 square feet of road. If each truckload costs x, and there's a 20% sales tax, how much will Mark need to pay for asphalt?\nIf we know the answer to the above question is 4500, what is the value of unknown variable x?\nA: Let's think step by step.", "arg_1": {"do_sample": true, "until": ["Q:", "</s>", "<|im_end|>"], "temperature": 0.7, "top_p": 0.9}}}, 
# "resps": [[
#     "The road is 2000 feet long and 20 feet wide, so the total area of the road is 2000 * 20 = 40000 square feet.\nEach truckload of asphalt will cover 800 square feet of road, so Mark will need 40000 / 800 = 50 truckloads of asphalt.\nThe cost of each truckload of asphalt is x, so the total cost of the asphalt is 50 * x = 50x.\nThere's a 20% sales tax, so the total cost of the asphalt and sales tax is 50x + 0.2x = 50.2x.\nWe are given that the total cost is $4500, so we can write: 50.2x = 4500.\nDividing both sides by 50.2, we get: x = 90.\nThe value of x is $90.\n#### 90\nThe answer is: 90", 
#     "The road is 2000 feet long and 20 feet wide, so the total area of the road is 2000 * 20 = 40,000 square feet.\nEach truckload of asphalt covers 800 square feet of road, so Mark needs to buy 40,000 / 800 = 50 truckloads of asphalt.\nThe cost of each truckload of asphalt is x, so the total cost of all the truckloads is 50x.\nThere's a 20% sales tax, so the total amount Mark needs to pay is 50x + 0.2 * 50x = 50x + 10x = 60x.\nWe are given that the total amount Mark needs to pay is 4500, so we can write: 60x = 4500.\nDividing both sides by 60, we get: x = 75.\nThe value of x is 75.\n#### 75\nThe answer is: 75", 
#     "The road will be 2000 feet long and 20 feet wide, so the total area of the road is 2000 * 20 = 40,000 square feet.\nEach truckload of asphalt will cover 800 square feet, so Mark will need 40,000 / 800 = 50 truckloads of asphalt.\nThe total cost of the asphalt is the product of the number of truckloads and the cost per truckload: 50 * x.\nThere's a 20% sales tax, so the total amount Mark will need to pay is: 50 * x + 0.20 * 50 * x = 50 * x + 10x = 60x.\nWe are given that the total amount Mark will need to pay is 4500, so we can write: 60x = 4500.\nDividing both sides by 60, we get: x = 75.\nThe value of x is 75.\n#### 75\nThe answer is: 75", 
#     "Mark's road is 2000 feet long and 20 feet wide, so the total area of the road is 2000 * 20 = 40000 square feet.\nEach truckload of asphalt will cover 800 square feet of road, so the number of truckloads needed to cover the entire road is 40000 / 800 = 50 truckloads.\nEach truckload costs x, so the total cost of the asphalt without sales tax is 50 * x.\nSales tax is 20% of the total cost, so the sales tax is 20/100 * 50 * x = 10 * x.\nThe total cost of the asphalt including sales tax is 50 * x + 10 * x = 60 * x.\nWe are given that the total cost of the asphalt is $4500, so we can write: 60 * x = 4500.\nDividing both sides by 60, we get: x = 75.\nThe value of x is 75.\n#### 75\nThe"]], "filtered_resps": ["90"], "filter": "score-first", "metrics": ["exact_match"], "doc_hash": "97137c8c6dd0df49e7569113422f929456797e6b6b059b028e6b75c1f7df6697", "prompt_hash": "7d80f3ff3aad888774211b6e5e560a3d13a87f701016d928f3823f50062ee73e", "target_hash": "0bfcd7ebd79cd57e304f970d640f883a4543175dc49cc1cdbc0df5bcc66d4971", "exact_match": 0.0}
# {"doc_id": 6, "doc": {"type": "GSM_FOBAR", "query": "Evan’s dog weighs 63 pounds; it weighs x times as much as Ivan’s dog.  Together, what is the weight of the dogs?\nIf we know the answer to the above question is 72, what is the value of unknown variable x?", "original_question": "Evan’s dog weighs 63 pounds; it weighs 7 times as much as Ivan’s dog.  Together, what is the weight of the dogs?", "response": "We know that Evan's dog weighs 63 pounds.\nWe also know that Evan's dog weighs x times as much as Ivan's dog, which means Ivan's dog weighs 63/x pounds.\nThe total weight of the dogs is the sum of the weight of Evan's dog and the weight of Ivan's dog: 63 + 63/x.\nWe are given that the total weight of the dogs is 72 pounds, so we can write: 63 + 63/x = 72.\nSolving for x, we get: x = 7.\nThe value of x is 7.\n#### 7\nThe answer is: 7"}, "target": "We know that Evan's dog weighs 63 pounds.\nWe also know that Evan's dog weighs x times as much as Ivan's dog, which means Ivan's dog weighs 63/x pounds.\nThe total weight of the dogs is the sum of the weight of Evan's dog and the weight of Ivan's dog: 63 + 63/x.\nWe are given that the total weight of the dogs is 72 pounds, so we can write: 63 + 63/x = 72.\nSolving for x, we get: x = 7.\nThe value of x is 7.\n#### 7\nThe answer is: 7", "arguments": {"gen_args_0": {"arg_0": "Q: Evan’s dog weighs 63 pounds; it weighs x times as much as Ivan’s dog.  Together, what is the weight of the dogs?\nIf we know the answer to the above question is 72, what is the value of unknown variable x?\nA: Let's think step by step.", "arg_1": {"do_sample": true, "until": ["Q:", "</s>", "<|im_end|>"], "temperature": 0.7, "top_p": 0.9}}}, 
# "resps": [[
#     "Evan's dog weighs 63 pounds.\nIvan's dog weighs x times as much as Evan's dog, so Ivan's dog weighs x * 63 pounds.\nThe total weight of the dogs is the sum of the weights of Evan's dog and Ivan's dog: 63 + x * 63.\nWe are given that the total weight of the dogs is 72 pounds, so we can write: 63 + x * 63 = 72.\nSolving for x, we get: x * 63 = 72 - 63.\nx * 63 = 9.\nDividing both sides by 63, we get: x = 9 / 63.\nx = 1 / 7.\nThe value of x is 1 / 7.\n#### 1\nThe answer is: 1", 
#     "We are given that Evan's dog weighs 63 pounds and it weighs x times as much as Ivan's dog.\nTo find the weight of the dogs, we need to add the weights of both dogs.\nThe total weight of the dogs is 63 + x * Ivan's dog.\nWe are also given that the total weight of the dogs is 72 pounds, so we can write: 63 + x * Ivan's dog = 72.\nSubtracting 63 from both sides, we get: x * Ivan's dog = 15.\nDividing both sides by x, we get: Ivan's dog = 15 / x.\nWe are given that Evan's dog weighs x times as much as Ivan's dog, so we can write: x * Ivan's dog = Evan's dog.\nSubstituting the value of Ivan's dog, we get: x * (15 / x) = 63.\nSimplifying the equation, we get: 15 = 63.\nThis equation is not true, so there is no value of x that satisfies the given conditions.\nThe answer is: 6", 
#     "We know that Evan's dog weighs 63 pounds.\nWe also know that Evan's dog weighs x times as much as Ivan's dog.\nLet's denote Ivan's dog's weight as y pounds.\nWe can set up the equation as follows:\ny * x = 63\nWe are given that the total weight of the dogs is 72 pounds, so we can set up the equation as follows:\ny + 63 = 72\nLet's solve for y:\ny = 72 - 63\ny = 9\nSubstituting the value of y into the equation y * x = 63, we get:\n9 * x = 63\nTo solve for x, we divide both sides of the equation by 9:\nx = 63 / 9\nx = 7\nThe value of x is 7.\n#### 7\nThe answer is: 7", 
#     "We are given that Evan's dog weighs 63 pounds and it weighs x times as much as Ivan's dog.\nThe weight of Ivan's dog is x * 63 pounds.\nTogether, the weight of the dogs is the sum of the weight of Evan's dog and the weight of Ivan's dog: 63 + x * 63.\nWe are given that the total weight is 72 pounds, so we can write: 63 + x * 63 = 72.\nSubtracting 63 from both sides, we get: x * 63 = 72 - 63.\nDividing both sides by 63, we get: x = 0.115.\nThe value of x is 0.115.\n#### 0.115\nThe answer is: 0.115"]], "filtered_resps": ["1"], "filter": "score-first", "metrics": ["exact_match"], "doc_hash": "6e903420914b874cb314c3fbbc67b5566bcd030dfc1aff14fc6d9be9201bdb52", "prompt_hash": "758479a508c887a47e483b319fe9680e6ace379d9b340b9d4ae7f1bcafc183bd", "target_hash": "4b8fd249de211bbcfa2b271cc27d4db67b50c6af333198ac3c22c26a95e019e7", "exact_match": 0.0}
