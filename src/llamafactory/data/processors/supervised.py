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


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    input_ids, labels = [], []
    block_size = data_args.cutoff_len
    skip_samples = []
    all_source_ids = [ [] for i in range(len(examples["prompt"])) ]
    #all_source_length = []
    #print(all_source_ids)
    all_target_ids = [ [] for i in range(len(examples["prompt"])) ]
    total_tokens = [ 0 for i in range(len(examples["prompt"])) ]
    #all_target_length = []

    # print("\nDo Tokeinizer Fast? [ ", tokenizer.is_fast, " ]")
    if tokenizer.is_fast: # gotzmann
        print("[ ERROR ] Fast Tokenizer is ON - stopping here, please use Legacy!")
        # exit(0)

    print("Preprocessing [ ", len(examples["prompt"]), " ] samples...")

    # -- set max allowed for debug or zero for production

    MAX_ALLOWED = 0
    sampleCount = len(examples["prompt"])

    # -- pre-tokenize all dataset

    for i in range(sampleCount):

        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        if MAX_ALLOWED != 0 and i >= MAX_ALLOWED: break

        turn = 0

        # print("encode_multiturn") # DEBUG
        # Special case of embedding PT style into SFT
        #print(target_ids)
        #exit(0)
        #print(i)
        #print(examples["system"][i])

        # === PT | PRE-TRAIN MODE ===

        # NB! We expect only one GPT sample here, not conversations!
        #print(examples["system"][i])
        #exit(0)
        if examples["system"][i] == "": # and len(examples["prompt"][i]) == 0 and examples["prompt"][i][0] == "":

            # print("[ PT ]") # DEBUG
            # _, target_ids = template.encode_oneturn(tokenizer, examples["response"][i])
            #_, target_ids = template.encode_multiturn(tokenizer, examples["response"][i])
            gpt = examples["response"][i][0]['content']
            # print(gpt)
            # FIXME: need to manually add BOS when system prompt is empty [ 1 == BOS | 2 == EOS ]
            # target_ids = [ 1 ] + tokenizer.encode(gpt, add_special_tokens=False) + [ 2 ]
            target_ids = tokenizer.encode(gpt, add_special_tokens=False)
            # print("=== [ target_ids ] ===") # DEBUG
            # print(target_ids) # DEBUG
            # print("=== [ source_ids ] ===") # DEBUG
            source_ids = []
            # print(source_ids) # DEBUG

            all_source_ids[i].append(source_ids)
            all_target_ids[i].append(target_ids)
            total_tokens[i] += len(source_ids) + len(target_ids)

        # === STF | FINE TUNE MODE ===

        else:

            for source_ids, target_ids in template.encode_multiturn(
                tokenizer, examples["prompt"][i] + examples["response"][i], examples["system"][i], examples["tools"][i]
            ):
                
                # if source_ids[0] == 

                ###if turn == 0:
                    ###buffer_source_ids = [ source_ids ]
                    #print("\n\n=== TURN = ", turn)
                    #print("\n\n=== LEN = ", len(buffer_source_ids))
                    #print("\n\n=== buffer_source_ids ===\n\n", buffer_source_ids)
                    #all_source_length[i][z] = len(source_ids)
                    ##### all_target_ids[i] += [ target_ids ] # .append(target_ids) 
                    #####buffer_target_ids += target_ids
                    #all_target_length[i][z] = len(target_ids)
                ###else:
                #########################buffer_source_ids += [ source_ids ]
                #all_source_ids[i] += [ source_ids ]
                all_source_ids[i].append(source_ids)
                #print("\n\n=== TURN = ", turn)
                #print("\n\n=== LEN = ", len(buffer_source_ids))
                #print("\n\n=== buffer_source_ids ===\n\n", buffer_source_ids)
                #all_source_length[i][z] = len(source_ids)
                ##### all_target_ids[i] += [ target_ids ] # .append(target_ids) 
                #######################################buffer_target_ids += [ target_ids ]
                #####all_target_ids[i] += [ target_ids ]
                # Dirty Fix: Remove EOS from the middle of nowhere
                ###############print("\n\n=== I = {} | TURN = {} | LEN = {} ===\n\n".format(i, turn, len(examples["prompt"][i])))
                ###########################if turn < len(examples["prompt"][i]) and target_ids[-1] == tokenizer.eos_token_id:
            
                ##### if target_ids[-1] == tokenizer.eos_token_id:
                #####     target_ids.pop()
                all_target_ids[i].append(target_ids)
                total_tokens[i] += len(source_ids) + len(target_ids)
                turn += 1      

        ##### all_target_ids[i][turn-1] += [ tokenizer.eos_token_id ] # Dirty Fix again
        ##### total_tokens[i] += 1

        ##################################all_source_ids[i] = buffer_source_ids
        ########################################all_target_ids[i] = buffer_target_ids
        ## print("\n\n=== all_source_ids {} ===\n\n{}".format(i, all_source_ids))
        ## print("\n\n=== all_target_ids {} ===\n\n{}".format(i, all_target_ids))
        #print("\n\n=== all_target_ids {} ===\n\n", i, all_target_ids)
        #exit()

    # -- main loop

    for i in range(sampleCount):

        if MAX_ALLOWED != 0 and i >= MAX_ALLOWED: break

        if i in skip_samples:
            #print("\n\n=== SKIPPING # {}".format(i))
            #print("\n\n=== SKIP SAMPLES ===", skip_samples)
            continue

        #print("\n\n=== [ PROMPT + RESPONSE ] =======================================================================\n\n")
        #print(examples["prompt"][i])
        #print("\n\n")
        #print(examples["response"][i])

        # TODO: Why there no such samples?
        # if len(examples["prompt"][i]) > 1:
        #    exit()

        #if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            #print("\n\n=== len(examples[prompt][i]) % 2 != 1 or len(examples[response][i]) != 1 ===\n\n")
            #print(examples["prompt"][i])
            #print("\n\n")
            #print(examples["response"][i])
        #    continue

        #messages = examples["prompt"][i] + examples["response"][i]
        ##### for source_ids, target_ids in template.encode_multiturn(
            #tokenizer, messages, examples["system"][i], examples["tools"][i]
        #####     tokenizer, examples["prompt"][i] + examples["response"][i], examples["system"][i], examples["tools"][i]
        ##### ):
        ### for turn in range(len(all_source_ids[i])):

            ### source_ids = all_source_ids[i][turn]
            ### target_ids = all_target_ids[i][turn]
        
        #####source_ids  = []
        #####target_ids  = []
        #####source_mask = []

        ## for turn in range(len(all_source_ids[i])):
        
            #####source_ids  += all_source_ids[i][turn]
            #####target_ids  += all_target_ids[i][turn]
            #####source_mask += [IGNORE_INDEX] * len(all_source_ids[i][turn])

            ### if turn > 0:
            ###     print("\n\n=== TURN2 = ", turn)

            # DEBUG | skip_special_tokens=True
            ## print("\n\n=== [ SAMPLE # {} | TURN # {} ] ============================================\n\n{}\n\n".
            ##    format(i, turn, tokenizer.decode(all_source_ids[i][turn] + all_target_ids[i][turn], skip_special_tokens=False)))    
 
            # if data_args.train_on_prompt:
            #     source_mask += all_source_ids[i][turn]
            # elif len(input_ids) != 0 and template.efficient_eos:
            #     source_mask += [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(all_source_ids[i][turn]) - 1)
            # else:
            #     source_mask += [IGNORE_INDEX] * len(all_source_ids[i][turn])

        ##### sample_length = len(source_ids) + len(target_ids)
        ##### expected_length = len(input_ids) + sample_length
        expected_length = len(input_ids) + total_tokens[i]
        #print("===    TOTAL LENGTH = ", len(input_ids))
        #print("=== + SAMPLE LENGTH = ", sample_length)
        #print("=== = BECOME LENGTH = ", expected_length)
        #total_length = len(input_ids)
        total_blocks = len(input_ids) // block_size
        expected_blocks = expected_length // block_size

        # -- draft solution for sample padding / shrinking / aligning

        # easy flow, just add new sample
        if total_blocks == expected_blocks:
            for turn in range(len(all_source_ids[i])):
                source_mask = [IGNORE_INDEX] * len(all_source_ids[i][turn])
                ##### input_ids += source_ids + target_ids
                input_ids += all_source_ids[i][turn] + all_target_ids[i][turn]
                ##### labels += source_mask + target_ids
                labels += source_mask + all_target_ids[i][turn]
                #padding_length = (total_blocks + 1) * block_size - len(input_ids)
                ## print("=== SAMPLE ADDED EASY # ", i)
                ## print("\n\n=== [ INPUT AFTER ] ============================================\n\n{}\n\n".
                ##     format(tokenizer.decode(input_ids, skip_special_tokens=False)))
            
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
                    
                    # print("\n\n=== [ INPUT AFTER ] ============================================\n\n{}\n\n".
                    #     format(tokenizer.decode(input_ids, skip_special_tokens=False)))

                ## print("\n\n === [ SHORT SAMPLE # {} OF LEN {} WAS ADDED ] ===".format(short, total_tokens[short]))
                padding_length = (total_blocks + 1) * block_size - len(input_ids)
                skip_samples += [ short ]

            # FIXME: Is it possible to fill more input_ids than padding space allowed? Check for multi-turn conversations
            ##### for short_source_ids, short_target_ids in template.encode_multiturn(
            #####     tokenizer, examples["prompt"][j] + examples["response"][j], examples["system"][j], examples["tools"][j]
            ##### ):

            # TODO: Not sure how it works when [ z ] is more than one turn interation
            ### for z in range(len(all_source_ids[j])):

                ### short_source_ids = all_source_ids[j][z]
                ### short_target_ids = all_target_ids[j][z]
            
            #####short_source_ids = all_source_ids[j]
            #####short_target_ids = all_target_ids[j]

            #####if len(input_ids) != 0 and template.efficient_eos:
            #####    short_source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(short_source_ids) - 1)
            #####else:
            #####    short_source_mask = [IGNORE_INDEX] * len(short_source_ids)

            #####skip_samples += [ j ]
            #####input_ids += short_source_ids + short_target_ids
            #####labels += short_source_mask + short_target_ids
            #####padding_length = (total_blocks + 1) * block_size - len(input_ids)
            #print("=== SAMPLE FOUND HARD # ", j)
            #print("=== ADD SKIP # ", j)
            
            # -- then fill remaining space with padding

            # add right padding to the current block
            #print("\n\n=== [ !!! PADDING !!! ] ===")
            #padding_length = (total_blocks + 1) * block_size - len(input_ids)
            #print("\n\n=== [ PADDING LENGTH = ", padding_length, " ] ===\n\n")
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

                    allow_length = block_size - len(source_ids) - 1

                    # it's not expected for longer samples but who knows
                    if allow_length < 100 or allow_length > len(target_ids):
                        print("[ ERROR ] allow_length is less than 100 or bigger than target length")
                        continue

                    for tail in range(allow_length, 0, -1):
                        # print (target_ids[i], " ?==? ", tokenizer.eos_token_id, " || ", target_ids[i] == tokenizer.eos_token_id)
                        if target_ids[tail] == 13:
                            if tail > 0 and target_ids[tail-1] == 13:
                                tail -= 1
                            allow_length = tail # replace last 0x13 or double 0x13 with EOS
                            break
                    #print("\n\n=== [ CUT LENGTH = ", cut_length, " ] ===\n\n")
                    target_ids = target_ids[:allow_length] + [ tokenizer.eos_token_id ]
                    #print("\n\n=== [ SAMPLE # {} LENGTH = ".format(i), total_tokens[i], " || REFRAMED LENGTH = ", len(source_ids) + len(target_ids), " ] ===\n\n")
                    #print("=== SAMPLE SHRUNK # ", j)  

            # -- add long sample -> it opens a new block

            # NB! Multi-turn code, need some attention later    
            # for turn in range(len(all_source_ids[i])):
            #    source_mask = [IGNORE_INDEX] * len(all_source_ids[i][turn])
            #    input_ids += all_source_ids[i][turn] + all_target_ids[i][turn]
            #    labels += source_mask + all_target_ids[i][turn]
                
            # NB! Just ignore multi-turns for now    
            source_mask = [IGNORE_INDEX] * len(source_ids)
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            # print("\n\n=== SAMPLE ADDED LONG # {}".format(i))
            ##### padding_length = (total_blocks + 1) * block_size - len(input_ids)
            # print("\n\n=== [ INPUT AFTER ] ============================================\n\n{}\n\n".
            #    format(tokenizer.decode(input_ids, skip_special_tokens=False)))

        if MAX_ALLOWED !=0:

            print("\n\n=== [ INPUTS ] =======================================================================\n\n")
            print(format(tokenizer.decode(input_ids, skip_special_tokens=False)))
            print("\n\n=== [ IDS ] =======================================================================\n\n")
            print(input_ids)
            print("\n\n=== [ LABELS ] =======================================================================\n\n")
            print(labels)
            # exit()
            
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
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

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
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)))
