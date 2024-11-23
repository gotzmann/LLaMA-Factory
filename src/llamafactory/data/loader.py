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

import os
import sys
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers.utils.versions import require_version

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import has_tokenized_data
from .aligner import align_dataset
from .data_utils import merge_dataset, split_dataset
from .parser import get_dataset_list
from .preprocess import get_preprocess_and_print_func


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .template import Template


logger = logging.get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        require_version("modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0")
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        require_version("openmind>=0.8.0", "To fix: pip install openmind>=0.8.0")
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=data_args.streaming,
            trust_remote_code=True,
        )

        # gotzmann --
        from datasets import concatenate_datasets
        from random import randint
        if training_args.num_train_epochs != int(training_args.num_train_epochs):
            raise ValueError("Use only integer numbers for [ num_train_epochs ] parameter.")
        # print("=== NUM EPOCHs = ", training_args.num_train_epochs)
        epoch_datasets = []
        epoch_datasets.append(dataset)
        for epoch in range(1, int(training_args.num_train_epochs)):
            # epoch_datasets.append(dataset.shuffle(keep_in_memory = True)) # seed=randint(0, 1980)
            epoch_datasets.append(dataset.shuffle())
            # print("=== EPOCH #", epoch)
        dataset = concatenate_datasets(epoch_datasets)
        # dataset.to_json("./merged.jsonl", force_ascii=False) # DEBUG
        training_args.num_train_epochs = 1.0 # NB!
        # -- gotzmann

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: Optional[Sequence[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Gets the merged datasets in the standard format.
    """
    if dataset_names is None:
        return None

    datasets = []
    for dataset_attr in get_dataset_list(dataset_names, data_args.dataset_dir):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args))

    return merge_dataset(datasets, data_args, seed=training_args.seed)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None

    preprocess_func, print_function = get_preprocess_and_print_func(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
		# batch_size=500, # gotzmann
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            print_function(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    """
    # Load tokenized dataset
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            dataset_dict: "DatasetDict" = load_from_disk(data_args.tokenized_path)
            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")

            dataset_module: Dict[str, "Dataset"] = {}
            if "train" in dataset_dict:
                dataset_module["train_dataset"] = dataset_dict["train"]

            if "validation" in dataset_dict:
                dataset_module["eval_dataset"] = dataset_dict["validation"]

            if data_args.streaming:
                dataset_module = {k: v.to_iterable_dataset() for k, v in dataset_module.items()}

            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args, stage)

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        eval_dataset = _get_preprocessed_dataset(
            eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
        )
		
        # NEW DEBUG | gotzmann --

        rank = int(os.environ.get('LOCAL_RANK', -1))
        if rank == 0:
            print("\n\n=== Writing [ 10 ] blocks to disk... ===\n\n")

        num = 0
        for block in iter(dataset):

            if rank != 0: continue

            if num >= 10: break
            # continue # DEBUG

            input_ids = block["input_ids"]
            labels = block["labels"]
            attention = block["attention_mask"]

            # === DEBUG | gotzmann | _encode_supervised_example process CPT samples correct
            from colorama import Fore, Back, Style
            prev_attention = 0
            print(Fore.LIGHTYELLOW_EX + f"\n\n============================== [ SAMPLE # {num} ] ==============================") # Fore.WHITE
            for pos, word in enumerate(input_ids):
                word = tokenizer.decode(input_ids[pos], skip_special_tokens=False)
                if prev_attention != attention[pos]:
                    print(Fore.LIGHTMAGENTA_EX + "\n\n[ " + str(attention[pos]) + " ]\n\n", end="")
                    # logger.info_rank0(Fore.LIGHTMAGENTA_EX + "\n\n[ " + str(attention[pos]) + " ]\n\n")
                    prev_attention = attention[pos]
                if labels[pos] >= 0:
                    color = Fore.LIGHTGREEN_EX if input_ids[pos] < 128000 else Fore.LIGHTYELLOW_EX
                else:
                    color = Fore.LIGHTBLACK_EX
                print(color + word, end="")
                # logger.info_rank0(color + word)
            print(Style.RESET_ALL)  
            # gotzmann | DEBUG ===

            sample = format(tokenizer.decode(input_ids, skip_special_tokens=False))
            f = open('./batches/input_ids.' + str(num), 'w')
            f.write(sample)
            f.close()

            labels = ', ' . join(map(str, labels))
            f = open('./batches/labels.' + str(num), 'w')
            f.write(labels)
            f.close()

            attention_mask = ', ' . join(map(str, attention))
            f = open('./batches/attention_mask.' + str(num), 'w')
            f.write(attention_mask)
            f.close()

            num += 1
		
        # gotzmann | DEBUG

        # print ("\n\n=== SEARCHING FOR DUBS... ===\n\n")
        # hashes = []
        # samples = []
        # num = 0
        # for block in iter(dataset):
        #     sample = format(tokenizer.decode(block["input_ids"], skip_special_tokens=False))
        #     sampleHash = hash(sample)
        #     #print("\n\n=== SAMPLE # {} ===\n\n".format(num), sample)
        #     #print("\n\n=== HASH # {} ===\n\n".format(num), sampleHash)
        #     samples += [ sample ]
        #     hashes += [ sampleHash ]
        #     num += 1
        #     if num > 100:
        #         break

        # for z in range(len(samples)):
        #     for dub in range(z+1, len(samples)):
        #         if hashes[z] == hashes[dub]:
        #             print("\n\n=== DUB FOUND !!! {} == {} === \n\n".format(z, dub))
        #             print(samples[dub])
        #             #exit()
		
		# -- NEW DEBUG

        if data_args.val_size > 1e-6:
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming:
                    dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if data_args.streaming:
                    eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["validation"] = eval_dataset

            dataset_dict = DatasetDict(dataset_dict)

        if data_args.tokenized_path is not None:
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please restart the training with `tokenized_path: {data_args.tokenized_path}`.")

            sys.exit(0)

        dataset_module = {}
        if "train" in dataset_dict:
            dataset_module["train_dataset"] = dataset_dict["train"]

        if "validation" in dataset_dict:
            dataset_module["eval_dataset"] = dataset_dict["validation"]

        return dataset_module