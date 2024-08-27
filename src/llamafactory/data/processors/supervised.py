# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import get_paligemma_token_type_ids, get_pixel_values, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


def _encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:
    if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
        prompt[0]["content"] = template.image_token + prompt[0]["content"]

    messages = prompt + response
    input_ids, labels = [], []

    if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        input_ids += [image_token_id] * getattr(processor, "image_seq_length")
        labels += [IGNORE_INDEX] * getattr(processor, "image_seq_length")

    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = 1 if template.efficient_eos else 0
    if mask_history:
        encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= cutoff_len:
            break

        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if train_on_prompt:
            source_label = source_ids
        elif template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != 0:  # train on the last turn only
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        if mask_history:  # reversed sequences
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label
			
    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    # === TRINITY | gotzmann | Simply ignore all previous code, we need no special tokens for PRETRAIN examples
    if system == "":
        text = ""
        if response[0]['content'] != "":
            text = response[0]['content']
        if prompt[0]['content'] != "":
            text = prompt[0]['content']
        if text == "":
            return [], [] 
        # print("\n\n === TEXT | PRE-TRAIN ===\n\n", text)
        ##### input_ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids = [ tokenizer.eos_token_id ] + tokenizer.encode(text, add_special_tokens=False)
        # total_length = 1 if template.efficient_eos else 0
        # special_tokens_length = 1 if template.efficient_eos else 0
        # if len(input_ids) >= data_args.cutoff_len + special_tokens_length:
        #     input_ids = input_ids[:data_args.cutoff_len - special_tokens_length]
        # source_mask = source_ids
        # if template.efficient_eos:
        #    source_mask = [tokenizer.eos_token_id]
        # input_ids += source_ids + target_ids
        # labels += source_mask + target_ids
        if len(input_ids) >= cutoff_len:
            input_ids = input_ids[:cutoff_len]
        ##### labels = input_ids
        labels = [IGNORE_INDEX] + input_ids
    # gotzmann | TRINITY ===		

    return input_ids, labels


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        if processor is not None:
            model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["token_type_ids"].append(get_paligemma_token_type_ids(len(input_ids), processor))

    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    #print("-> preprocess_packed_supervised_dataset [ " + str(len(examples["prompt"])) + " ] BEFORE...") # DEBUG
    valid_num = 0
    batch_input_ids, batch_labels = [], []
    lengths = []
    length2indexes = defaultdict(list)
    lll = str(len(examples["prompt"])) # DEBUG
    for i in range(len(examples["prompt"])):
        #print("===> ### " + str(i) + " OF " + lll) # DEBUG
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=None,
            cutoff_len=data_args.cutoff_len - 1,  # reserved for the padding token
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )

        # === NEW DEBUG | gotzmann
        # NB! ^^^ _encode_supervised_example ^^^ already knows how to process PRE-TRAIN samples correct
        # if i < 3:
        #     print("\n\n=== [ INPUTS ] =======================================================================\n\n")
        #     print(format(tokenizer.decode(input_ids, skip_special_tokens=False)))
        #     print("\n\n=== [ IDS : " + str(len(input_ids)) + " ] =======================================================================\n\n")
        #     print(input_ids)
        #     print("\n\n=== [ LABELS : " + str(len(labels)) + " ] =======================================================================\n\n")
        #     print(labels)
        # gotzmann | NEW DEBUG ===

        length = len(input_ids)
        if length > data_args.cutoff_len:
            logger.warning("Dropped lengthy example with length {} > {}.".format(length, data_args.cutoff_len))
        else:
            lengths.append(length)
            length2indexes[length].append(valid_num)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            valid_num += 1

    # === KNAPSACKS | gotzmann
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    packed_input_ids, packed_attention_masks, packed_labels = [], [], []
    # packed_labels = []
    used_samples = []
    remaining_capacity = data_args.cutoff_len
    for index, length in enumerate(lengths):
        #print("===> # " + str(index) + " OF " + str(len(lengths))) # DEBUG
        if index in used_samples: continue
        # -- just fit current sample into knapsack
        if length <= remaining_capacity:
            #print("\n\nremaining == " + str(remaining_capacity))
            #print("1 DO) " + str(index) + ": [ " + str(len(packed_input_ids)) + " ] + " + str(len(batch_input_ids[index])))
            packed_input_ids += batch_input_ids[index]
            #print("1 POSLE) " + str(index) + ": [ " + str(len(packed_input_ids)) + " ]")
            packed_labels += batch_labels[index]
            remaining_capacity -= length
            used_samples.append(index)
            #print("remaining == " + str(remaining_capacity))
            # if data_args.neat_packing:
            #     packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
            # else:
            packed_attention_masks += [1] * len(batch_input_ids[index]) # TODO: neat_packing
            continue
        else:
            # -- looking for samples fitting into knapsack
            for current in range(index+1, len(lengths)):
                #print("=> #" + str(current) + " OF " + str(len(lengths)))
                # -- filling current knapsack with padding + starting new one
                if remaining_capacity < 100 or current == len(lengths)-1:
                    #print("\n\n BREAK !!!")
                    #print("\n\n +) padding: " + str(remaining_capacity))
                    # pad_length = data_args.cutoff_len - len(packed_input_ids)
                    #packed_input_ids += [tokenizer.pad_token_id] * remaining_capacity
                    #packed_labels += [IGNORE_INDEX] * remaining_capacity
                    #remaining_capacity = 0
                    #print("\n\nremaining == " + str(remaining_capacity))
                    #if len(packed_input_ids) > 8192: # DEBUG
                    #    print("\n\n TROUBLES 2 !!! > 8192\n\n")
                    break
                # -- else skipping or adding current sample into knapsack
                if current in used_samples: continue
                if lengths[current] > remaining_capacity: continue
                #print("\n\nremaining == " + str(remaining_capacity))
                #print("2 DO ) " + str(current) + ": [ " + str(len(packed_input_ids)) + " ] + " + str(len(batch_input_ids[current])))
                packed_input_ids += batch_input_ids[current]
                packed_labels += batch_labels[current]
                remaining_capacity -= lengths[current]
                #print("remaining == " + str(remaining_capacity))
                used_samples.append(current)
                packed_attention_masks += [1] * len(batch_input_ids[current]) # TODO: neat_packing
                continue
        #print("\n\nPADDING: " + str(remaining_capacity))    
        packed_input_ids += [tokenizer.pad_token_id] * remaining_capacity
        packed_labels += [IGNORE_INDEX] * remaining_capacity
        packed_attention_masks += [1] * remaining_capacity # TODO: neat_packing
        remaining_capacity = 0  
        # -- sanity check
        if len(packed_input_ids) != data_args.cutoff_len:
            print("\n\n=== packed_input_ids " + str(len(packed_input_ids)) + " === \n\n")
            #print(tokenizer.decode(packed_input_ids, skip_special_tokens=False))
            raise ValueError("The length of packed example should be identical to the cutoff length.")
        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["labels"].append(packed_labels)
        packed_input_ids, packed_labels = [], []
        remaining_capacity = data_args.cutoff_len
        if index not in used_samples:
            #print("\n\n3 DO) " + str(index) + ": [ " + str(len(packed_input_ids)) + " ] + " + str(len(batch_input_ids[index])))
            packed_input_ids += batch_input_ids[index]
            packed_labels += batch_labels[index]
            remaining_capacity -= length
            #print("3 POSLE) " + str(index) + ": [ " + str(len(packed_input_ids)) + " ]")
            #print("remaining == " + str(remaining_capacity))
            used_samples.append(index)
            packed_attention_masks += [1] * len(batch_input_ids[index]) # TODO: neat_packing
        # FIXME: Last samples migh be not added into final output        
        #model_inputs["input_ids"].append(packed_input_ids)
        #model_inputs["attention_mask"].append(packed_attention_masks)
        #model_inputs["labels"].append(packed_labels)
    # TODO: Check out all used_sampled are really used!    
    #print("<- AFTER | preprocess_packed_supervised_dataset [ " + str(len(examples["prompt"])) + " ] ...") # DEBUG    
    return model_inputs
    # gotzmann | KNAPSACKS ===

    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    knapsacks = greedy_knapsack(lengths, data_args.cutoff_len - 1)  # reserved for the padding token
    for knapsack in knapsacks:
        packed_input_ids, packed_attention_masks, packed_labels = [], [], []
        for i, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            packed_input_ids += batch_input_ids[index]
            packed_labels += batch_labels[index]
            if data_args.neat_packing:
                packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
            else:
                packed_attention_masks += [1] * len(batch_input_ids[index])

        if len(packed_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(packed_input_ids)
            packed_input_ids += [tokenizer.pad_token_id] * pad_length
            packed_labels += [IGNORE_INDEX] * pad_length
            if data_args.neat_packing:
                packed_attention_masks += [0] * pad_length
            else:
                packed_attention_masks += [1] * pad_length  # more efficient flash_attn

        if len(packed_input_ids) != data_args.cutoff_len:
            raise ValueError("The length of packed example should be identical to the cutoff length.")

        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["labels"].append(packed_labels)

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)))
