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

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)


def _encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
	neat_packing: bool = False, # gotzmann
) -> Tuple[List[int], List[int]]:
    messages = template.mm_plugin.process_messages(prompt + response, images, videos, processor)
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = len(input_ids) + (1 if template.efficient_eos else 0)
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
        
        # # Always BOS
        # input_ids = [ tokenizer.bos_token_id ] + tokenizer.encode(text, add_special_tokens=False) # + [ tokenizer.eot_token_id ]
        # labels = [ IGNORE_INDEX ] + input_ids[1:]
        # if len(input_ids) >= cutoff_len:
        #     input_ids = input_ids[:cutoff_len]
        #     labels = labels[:cutoff_len]

        # No BOS
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) >= cutoff_len:
            input_ids = input_ids[:cutoff_len]
        labels = input_ids

        # # Use BOS token to split CPT samples by default, otherwise split them with cross-contamination attention
        # if neat_packing:
        #     input_ids = tokenizer.encode(text, add_special_tokens=False)
        #     if len(input_ids) >= cutoff_len:
        #         input_ids = input_ids[:cutoff_len]
        #     labels = input_ids
        # else:
        #     input_ids = [ tokenizer.bos_token_id ] + tokenizer.encode(text, add_special_tokens=False)
        #     labels = [ IGNORE_INDEX ] + input_ids[1:]
        #     if len(input_ids) >= cutoff_len:
        #         input_ids = input_ids[:cutoff_len]
        #         labels = labels[:cutoff_len]

    # gotzmann | TRINITY ===		

    return input_ids, labels


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            # gotzmann
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][0] + examples["_response"][0])
            )
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
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
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # TODO: use `position_ids` to achieve packing
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    valid_num = 0
    batch_input_ids, batch_labels, batch_images, batch_videos = [], [], [], []
    lengths = []
    length2indexes = defaultdict(list)
    for i in range(len(examples["_prompt"])):

        # gotzmann
        
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            # gotzmann
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][0] + examples["_response"][0])
            )
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len - 1,  # reserved for the padding token
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
            neat_packing=data_args.neat_packing, # gotzmann
        )
        length = len(input_ids)
        if length > data_args.cutoff_len:
            logger.warning_rank0(f"Dropped lengthy example with length {length} > {data_args.cutoff_len}.")
        else:
            lengths.append(length)
            length2indexes[length].append(valid_num)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_images.append(examples["_images"][i] or [])
            batch_videos.append(examples["_videos"][i] or [])
            valid_num += 1

    # === KNAPSACKS | gotzmann
    from random import randint
    model_inputs = { "input_ids": [], "attention_mask": [], "labels": [] }
    packed_input_ids, packed_attention_masks, packed_labels = [], [], []
    used_samples = []
    remaining_capacity = data_args.cutoff_len
    sampleIndex = 1 # sample index within knapsack, starts from 1 [ for cross-contamination attention ]
    # firstLen = 0 # gotzmann | DEBUG
    # firstCount = 0 # gotzmann | DEBUG
    skipped = 0
    cutoff = data_args.cutoff_len
    # -- trying to build better packing with 2-step looping
    for step in range(1, 3):
        # print(f"=== STEP | {step} ===")
        for index, length in enumerate(lengths):
            if index in used_samples: continue
            total = len(batch_input_ids[index])
            # -- ignore too lengthy samples with mostly no data for active learning (most labels are -100)
            wasted = sum(label < 0 for label in batch_labels[index])
            if total > (0.80 * cutoff) and wasted > (0.80 * total):
                skipped += 1
                print(f"=== WASTED COUNT | {skipped} ===")
                used_samples.append(index)
                continue
            # -- do not allow longer samples as first ones within the packed block for the first iteration 
            if step == 1 and sampleIndex == 1 and total > 500 * cutoff and randint(0, 100) > 10: continue
            if step == 1 and sampleIndex == 2 and total > 800 * cutoff and randint(0, 100) > 20: continue
            # if step == 1 and sampleIndex == 3 and total > 1000 * cutoff and randint(0, 100) > 30: continue
            # if sampleIndex == 4 and step == 1 and total > 0.10 * cutoff and randint(0, 80): continue
            # -- just fit current sample into knapsack
            if length <= remaining_capacity:
                # if sampleIndex == 1: firstLen += len(batch_input_ids[index]); firstCount += 1 # DEBUG | metrics
                packed_input_ids += batch_input_ids[index]
                packed_labels += batch_labels[index]
                packed_attention_masks += [sampleIndex] * len(batch_input_ids[index])
                if data_args.neat_packing: sampleIndex += 1
                remaining_capacity -= length
                used_samples.append(index)
                continue
            else:
                # -- trying to shrink longer CPT samples to allow shorter samples fill beginning of the batch
                if batch_input_ids[index][0] != tokenizer.bos_token_id:
                    # shrinked_ids = 
                    # shrinked_labels = 
                    # shrinked_attention_masks = 
                    packed_input_ids += batch_input_ids[index][:remaining_capacity]
                    packed_labels += batch_labels[index][:remaining_capacity]
                    packed_attention_masks += [sampleIndex] * remaining_capacity
                    # model_inputs["input_ids"].append(packed_input_ids)
                    # model_inputs["attention_mask"].append(packed_attention_masks)
                    # model_inputs["labels"].append(packed_labels)
                    # packed_input_ids, packed_attention_masks, packed_labels = [], [], []
                    # remaining_capacity = data_args.cutoff_len
                    remaining_capacity = 0
                    used_samples.append(index)
                    # if sampleIndex == 1: firstLen += len(batch_input_ids[index]); firstCount += 1 # DEBUG | metrics
                    if data_args.neat_packing: sampleIndex += 1    
                    # i = 1
                    # print("\n\n=== CPT Sample ===\n\n")
                    # print(format(tokenizer.decode(shrinked_ids, skip_special_tokens=False)))
                    # continue
                else:    
                    # -- looking for samples fitting into knapsack
                    for current in range(index+1, len(lengths)):
                        if current in used_samples: continue
                        # -- filling current knapsack with padding + starting new one
                        # if remaining_capacity < (0.03 * data_args.cutoff_len): break
                        if remaining_capacity < 300: break
                        # -- else skipping or adding current sample into knapsack
                        if lengths[current] > remaining_capacity: continue
                        # -- ignore super short samples, it better to place them where block begins
                        # if lengths[current] < (0.05 * data_args.cutoff_len): continue
                        if lengths[current] < 300: continue
                        packed_input_ids += batch_input_ids[current]
                        packed_labels += batch_labels[current]
                        packed_attention_masks += [sampleIndex] * len(batch_input_ids[current])
                        # if sampleIndex == 1: firstLen += len(batch_input_ids[index]); firstCount += 1 # DEBUG | compute metrics
                        if data_args.neat_packing: sampleIndex += 1
                        remaining_capacity -= lengths[current]    
                        used_samples.append(current)
                        continue
            # -- padding        
            packed_input_ids += [tokenizer.pad_token_id] * remaining_capacity
            packed_labels += [IGNORE_INDEX] * remaining_capacity
            packed_attention_masks += [sampleIndex] * remaining_capacity
            # -- sanity check
            if len(packed_input_ids) != data_args.cutoff_len:
                print("\n\n=== packed_input_ids " + str(len(packed_input_ids)) + " === \n\n")
                #print(tokenizer.decode(packed_input_ids, skip_special_tokens=False))
                raise ValueError("The length of packed example should be identical to the cutoff length.")
            # -- expand total samples
            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["labels"].append(packed_labels)
            # -- reset block buffers and counter
            packed_input_ids, packed_attention_masks, packed_labels = [], [], []
            remaining_capacity = cutoff
            sampleIndex = 1
            # FIXME: Most last sample might be lost?
            if index not in used_samples:
                pass
                # print("[ WARNING ] Sample not in used samples") # DEBUG
                # packed_input_ids += batch_input_ids[index]
                # packed_labels += batch_labels[index]
                # packed_attention_masks += [i] * len(batch_input_ids[index])
                # if i == 1: 
                #     firstLen += len(batch_input_ids[index])
                #     firstCount += 1
                # if data_args.neat_packing: i += 1
                # remaining_capacity -= length
                # used_samples.append(index)
    # TODO: Check out all used_sampled are really used!
    # print("\n=== PACKING |", str(round(firstLen/firstCount)), "===")
    # === DEBUG | gotzmann | _encode_supervised_example process CPT samples correct
    # from colorama import Fore, Back, Style
    # if i < 10:
    #     print(f"\n\n============================== [ SAMPLE # {i} ] ==============================\n\n")
    #     words = tokenizer.decode(input_ids, skip_special_tokens=False)
    #     for pos, word in enumerate(words):
    #         if labels[pos] >= 0:
    #             color = Fore.GREEN if input_ids[pos] < 128000 else Fore.YELLOW
    #         else:
    #             color = Fore.LIGHTBLACK_EX
    #         print(color + word, end="")
    # gotzmann | DEBUG ===
    return model_inputs
    # gotzmann | KNAPSACKS ===

    model_inputs = defaultdict(list)
    knapsacks = greedy_knapsack(lengths, data_args.cutoff_len - 1)  # reserved for the padding token
    for knapsack in knapsacks:
        packed_input_ids, packed_attention_masks, packed_labels = [], [], []
        packed_images, packed_videos = [], []
        for i, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            packed_input_ids += batch_input_ids[index]
            packed_labels += batch_labels[index]
            packed_images += batch_images[index]
            packed_videos += batch_videos[index]
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
        model_inputs["images"].append(packed_images or None)
        model_inputs["videos"].append(packed_videos or None)

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    return # gotzmann
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print(f"labels:\n{tokenizer.decode(valid_labels, skip_special_tokens=False)}")
