# "final": { "file_name": "/home/git/final/final-x2.jsonl", "formatting": "sharegpt" },

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
adapter_model_name = "/home/exported"
output_dir = "/home/merged"

##### model, tokenizer = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)
##### model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(base_model_name)
#####        device_map="auto", offload_folder="offload", # allow bigger models not fitting RAM swap to disk
model = PeftModel.from_pretrained(model, adapter_model_name)

model = model.merge_and_unload(progressbar=True, safe_merge=True, adapter_names=None)
model.to(dtype=torch.float16)
# Need .bin format? safe_serialization=False
model.save_pretrained(output_dir, max_shard_size="10GB", safe_serialization=True, progressbar=True)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_dir)
