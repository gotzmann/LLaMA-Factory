from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...extras.constants import IGNORE_INDEX, IMAGE_TOKEN
from ...extras.logging import get_logger
from .mm_utils import get_paligemma_token_type_ids, get_pixel_values


if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


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

        if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
            examples["prompt"][i][0]["content"] = IMAGE_TOKEN + examples["prompt"][i][0]["content"]

        messages = examples["prompt"][i] + examples["response"][i]
        input_ids, labels = [], []

        if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
            image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            input_ids += [image_token_id] * getattr(processor, "image_seq_length")
            labels += [IGNORE_INDEX] * getattr(processor, "image_seq_length")

        for turn_idx, (source_ids, target_ids) in enumerate(
            template.encode_multiturn(
                tokenizer,
                messages,
                examples["system"][i],
                examples["tools"][i],
                data_args.cutoff_len,
                data_args.reserved_label_len,
            )
        ):
            if data_args.train_on_prompt:
                source_mask = source_ids
            elif turn_idx != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        if processor is not None:
            model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["token_type_ids"].append(get_paligemma_token_type_ids(len(input_ids), processor))

    return model_inputs

# gotzmann: FIXME: process LLaMA v2 and v3 EOS different

def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    block_size = data_args.cutoff_len
    sampleCount = len(examples["prompt"])
    input_ids, labels, skip_samples = [], [], []
    all_source_ids = [ [] for i in range(sampleCount) ]
    all_target_ids = [ [] for i in range(sampleCount) ]
    total_tokens = [ 0 for i in range(sampleCount) ]

    # print("\nDo Tokeinizer Fast? [ ", tokenizer.is_fast, " ]")
    if tokenizer.is_fast: # gotzmann
        print("[ ERROR ] Fast Tokenizer is ON - stopping here, please use Legacy!")
        # exit(0)

    MAX_ALLOWED = 0 # 10 # -- set max allowed for debug or zero for production

    # -- pre-tokenize all dataset

    print("Preprocessing [ ", len(examples["prompt"]), " ] samples...")
    for i in range(sampleCount):

        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        if MAX_ALLOWED != 0 and i >= MAX_ALLOWED: break

        ##### turn = 0

        # === PT | PRE-TRAIN MODE === NB! We expect only one GPT sample here, not conversations!

        if examples["system"][i] == "":

            source_ids = []
            gpt = examples["response"][i][0]['content']
            target_ids = tokenizer.encode(gpt, add_special_tokens=False)

            all_source_ids[i].append(source_ids)
            all_target_ids[i].append(target_ids)
            total_tokens[i] += len(source_ids) + len(target_ids)

        # === STF | FINE TUNE MODE ===

        else:

            ##### turn = 0
            for source_ids, target_ids in template.encode_multiturn(
                tokenizer, examples["prompt"][i] + examples["response"][i], examples["system"][i], examples["tools"][i]
            ):
                
                all_source_ids[i].append(source_ids)
                all_target_ids[i].append(target_ids)
                total_tokens[i] += len(source_ids) + len(target_ids)
                ##### turn += 1      

                # print("\n\n=== sample #{} source_ids ===\n\n{}".format(i, source_ids))  
                # print("\n\n=== sample #{} target_ids ===\n\n{}".format(i, target_ids))

        # all_target_ids[i][turn-1] += [ 128001, 128009 ] # v3 Dirty Fix again
        # total_tokens[i] += 1 # v3

        # print("\n\n=== all_source_ids #{} ===\n\n{}".format(i, all_source_ids[i]))
        # print("\n\n=== all_target_ids #{} ===\n\n{}".format(i, all_target_ids[i]))
        # print("\n\n=== all_target_ids #{} ===\n\n", i, all_target_ids[i])
        # #print("\n\n=== DECODE ===\n", tokenizer.decode(all_source_ids, skip_special_tokens=False))
        # #exit()

    # -- main loop

    for i in range(sampleCount):

        if MAX_ALLOWED != 0 and i >= MAX_ALLOWED: break
        if i in skip_samples: continue

        #print("\n\n=== [ PROMPT + RESPONSE ] =======================================================================\n\n")
        #print(examples["prompt"][i])
        #print("\n\n")
        #print(examples["response"][i])

        expected_length = len(input_ids) + total_tokens[i]
        total_blocks = len(input_ids) // block_size
        expected_blocks = expected_length // block_size

        # -- draft solution for sample padding / shrinking / aligning

        # easy flow, just add new sample
        if total_blocks == expected_blocks:

            for turn in range(len(all_source_ids[i])):
                source_mask = [IGNORE_INDEX] * len(all_source_ids[i][turn])
                input_ids += all_source_ids[i][turn] + all_target_ids[i][turn]
                labels += source_mask + all_target_ids[i][turn]
                # print("=== SAMPLE ADDED EASY # ", i)
                # print("\n\n=== [ INPUT AFTER ] ============================================\n\n{}\n\n".
                #      format(tokenizer.decode(input_ids, skip_special_tokens=False)))
            
            # TODO: Thing twice again
            # go to the next sample
            continue

        # otherwise pad block space with pad token, add sample to a new block
        # TODO: Introduce better strategy - shrink (?) samples longer than [ block_size ]
        # TODO: Shrink longer samples around meaningful parts: sentences and paragraphs
        # TODO: Optimize block filling depending on different sample size of the whole dataset
            
        else:

            # -- trying to fill long padding space with shorter samples 
            #    skipping this when padding space less than 100 tokens

            total_blocks = len(input_ids) // block_size
            padding_length = (total_blocks + 1) * block_size - len(input_ids)

            for short in range(i + 1, sampleCount):

                if padding_length < 100: break
                if short in skip_samples: continue
                if total_tokens[short] > padding_length: continue

                ## -- we've found short sample, use it!

                for turn in range(len(all_source_ids[short])):
                    source_mask = [IGNORE_INDEX] * len(all_source_ids[short][turn])
                    input_ids += all_source_ids[short][turn] + all_target_ids[short][turn]
                    labels += source_mask + all_target_ids[short][turn]
                    # print("\n\n === [ SHORT SAMPLE # {} OF LEN {} WAS ADDED ] ===".format(short, total_tokens[short]))
                    # print("\n\n=== [ INPUT AFTER ] ============================================\n\n{}\n\n".
                    #     format(tokenizer.decode(input_ids, skip_special_tokens=False)))

                padding_length = (total_blocks + 1) * block_size - len(input_ids)
                skip_samples += [ short ]
            
            # -- then fill remaining space with padding

            # add right padding to the current block
            input_ids += padding_length * [ tokenizer.eos_token_id ]
            labels += padding_length * [ -100 ]

            # -- shrink long sample when needed                                    

            # FIXME: Does it works with long multi-turn samples?
            # FIXME: What if source ids longer than block size?

            source_ids = all_source_ids[i][0]
            target_ids = all_target_ids[i][0]
            
            if total_tokens[i] > block_size:

                # -- if there just one long sample for pretrain, just shrink it for the block size, no BOS / EOS needed
                if len(source_ids) == 0:
                    target_ids = target_ids[:block_size]
                else:       

                    allow_length = block_size - len(source_ids) - 1 # v2
                    # allow_length = block_size - len(source_ids) - 2 # v3

                    # it's not expected for longer samples but who knows
                    if allow_length < 100 or allow_length > len(target_ids):
                        print("[ ERROR ] allow_length is less than 100 or bigger than target length")
                        continue

                    # v2 for tail in range(allow_length, 0, -1):
                    for tail in range(allow_length, 2, -1): # v3
                        # v2 if target_ids[tail] == 13:
                        if len(target_ids) > tail and target_ids[tail] == 271: # /n/n
                        # v2    if tail > 0 and target_ids[tail-1] == 13:
                        # v2        tail -= 1
                        # v2    allow_length = tail # replace last 0x13 or double 0x13 with EOS
                            allow_length = tail - 1 # replace last 0x13 or double 0x13 with EOS
                            break
                    #print("\n\n=== [ CUT LENGTH = ", cut_length, " ] ===\n\n")
                    target_ids = target_ids[:allow_length] + [ tokenizer.eos_token_id ]
                    # print("\n\n=== tokenizer.eos_token_id ===\n", tokenizer.eos_token_id)
                    # print("\n")
                    # v3 target_ids = target_ids[:allow_length] + [ 128001, 128009 ] 

            # -- add long sample -> it opens a new block
                
            # NB! Just ignore multi-turns for now    
            source_mask = [IGNORE_INDEX] * len(source_ids)
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            
            # print("\n\n=== SAMPLE ADDED LONG # {}".format(i))
            # print("\n\n=== [ INPUT AFTER ] ============================================\n\n{}\n\n".
            #     format(tokenizer.decode(input_ids, skip_special_tokens=False)))
            
            continue

        # if MAX_ALLOWED !=0:

            # print("\n\n=== [ INPUTS ] =======================================================================\n\n")
            # print(format(tokenizer.decode(input_ids, skip_special_tokens=False)))
            # print("\n\n=== [ IDS ] =======================================================================\n\n")
            # print(input_ids)
            # print("\n\n=== [ LABELS ] =======================================================================\n\n")
            # print(labels)
            # # exit()
            
        # DEBUG
        #print("\n\n=== [ INPUT AFTER ] ============================================\n\n{}".format(tokenizer.decode(input_ids, skip_special_tokens=False)))    
        #f = open('input.'+str(i), 'w')
        #f.write(tokenizer.decode(input_ids, skip_special_tokens=False))
        #f.close()            


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    # model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    # input_ids, labels = [], []
    # for i in range(len(examples["prompt"])):
    #     if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
    #         logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
    #         continue

    #     messages = examples["prompt"][i] + examples["response"][i]
    #     for source_ids, target_ids in template.encode_multiturn(
    #         tokenizer, messages, examples["system"][i], examples["tools"][i]
    #     ):
    #         if data_args.train_on_prompt:
    #             source_mask = source_ids
    #         elif len(input_ids) != 0 and template.efficient_eos:
    #             source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
    #         else:
    #             source_mask = [IGNORE_INDEX] * len(source_ids)

    #         input_ids += source_ids + target_ids
    #         labels += source_mask + target_ids
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id] # v2
        # v3 input_ids += [ 128001, 128009 ]
        labels += [tokenizer.eos_token_id] # v2
        # v3 labels += [ 128001, 128009 ]

    total_length = len(input_ids)
    block_size = data_args.cutoff_len
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (total_length // block_size) * block_size
    # split by chunks of cutoff_len
    for i in range(0, total_length, block_size):
        if not all(label == IGNORE_INDEX for label in labels[i : i + block_size]):
            model_inputs["input_ids"].append(input_ids[i : i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i : i + block_size])

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    return # gotzmann
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)))
