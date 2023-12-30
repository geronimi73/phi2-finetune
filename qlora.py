import torch
import os
import wandb
import uuid
import json
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback, set_seed
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict, Dataset,load_from_disk
from functools import partial

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    # if args.bits == 4: 
    trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


accelerator = Accelerator()

set_seed(42)

run_id = str(uuid.uuid4())
modelpath="microsoft/phi-2"
dataset_name="g-ronimo/riddles_evolved"
lr=0.00002
bs=1            # batch size
bs_eval=16        # batch size for evals
ga_steps=16     # gradient acc. steps
epochs=20
max_length=1024
output_dir=f"out"

lora_config = LoraConfig(
    r=32, 
    lora_alpha=32, 
    target_modules = ['Wqkv','out_proj'],
    # target_modules = ['fc1', 'fc2', 'Wqkv', 'out_proj'],
    lora_dropout=0.1, 
    bias="none", 
    modules_to_save = ["lm_head", "embed_tokens"],
    task_type="CAUSAL_LM"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    device_map={"": accelerator.process_index},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)    # fast tokenizer sometimes ignores the added tokens

# Add tokens <|im_start|> and <|im_end|>, latter is special eos token, 
tokenizer.add_tokens(["<|im_start|>", "<PAD>"])
tokenizer.pad_token = "<PAD>"
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
model.resize_token_embeddings(
    new_num_tokens=len(tokenizer),
    pad_to_multiple_of=64)   # phi2 default is 64, see configuration_phi.py
model.config.eos_token_id = tokenizer.eos_token_id

# Add adapters to model
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model = get_peft_model(model, lora_config)
model.config.use_cache = False

# Print stats
if accelerator.is_main_process:
    print_trainable_parameters(model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

# Load dataset
dataset = load_dataset(dataset_name)
dataset = dataset["train"].train_test_split(test_size=0.1)

# Format (chatML) and tokenize dataset
templates=[
    "<|im_start|>assistant\n{msg}<|im_end|>",
    "<|im_start|>user\n{msg}<|im_end|>"
]
IGNORE_INDEX=-100

def tokenize(input, max_length):
    input_ids, attention_mask, labels = [], [], []

    for i,msg in enumerate(input["messages"]):
        isHuman = i%2==0
        msg_chatml=templates[isHuman].format(msg=msg)
        msg_tokenized=tokenizer(msg_chatml, truncation=False, add_special_tokens=False)
    
        input_ids+=msg_tokenized["input_ids"]
        attention_mask+=msg_tokenized["attention_mask"]
        labels+=[IGNORE_INDEX]*len(msg_tokenized["input_ids"]) if isHuman else msg_tokenized["input_ids"]

    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": labels[:max_length],
    }

dataset_tokenized = dataset.map(
    partial(tokenize, max_length=max_length), 
    batched=False, 
    num_proc=os.cpu_count()//accelerator.num_processes,    # multithreaded
    remove_columns=dataset["train"].column_names  # don't need this anymore, we have tokens from here on
)

# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
def collate(elements):
    tokens=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokens])

    for i,sample in enumerate(elements):
        input_ids=sample["input_ids"]
        labels=sample["labels"]
        attention_mask=sample["attention_mask"]

        pad_len=tokens_maxlen-len(input_ids)

        input_ids.extend( pad_len * [tokenizer.pad_token_id] )   
        labels.extend( pad_len * [IGNORE_INDEX] )    
        attention_mask.extend( pad_len * [0] ) 

    batch={
        "input_ids": torch.tensor( [e["input_ids"] for e in elements] ),
        "labels": torch.tensor( [e["labels"] for e in elements] ),
        "attention_mask": torch.tensor( [e["attention_mask"] for e in elements] ),
    }

    return batch

steps_per_epoch=len(dataset_tokenized["train"])//(accelerator.num_processes*bs*ga_steps)

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs_eval,
    evaluation_strategy="steps",
    logging_steps=1,
    eval_steps=steps_per_epoch//2,    # 2 evals per epoch
    save_steps=steps_per_epoch,     # save once per epoch
    gradient_accumulation_steps=ga_steps,
    num_train_epochs=epochs,
    lr_scheduler_type="constant",
    optim="paged_adamw_32bit",      # val_loss will go nan with paged_adamw_8bit
    learning_rate=lr,
    group_by_length=False,
    bf16=True,        
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=collate,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
)

if accelerator.is_main_process:
    run = wandb.init(
        project="phi2",
        name=modelpath.split("/")[1]+"_"+dataset_name+f"_bs-{bs}_LR-{lr}_maxlen-{max_length}_{run_id}",
        config={
            "model_name": modelpath,
            "run_id": run_id,
            "dataset": dataset_name,
            "output_dir": output_dir,
            "lr": lr,
            "max_length": max_length,
            "train_batch_size": bs,
            "validation_batch_size": bs,
            "ga_steps": ga_steps,
            "lora_config": lora_config, 
            "training_args": args,
            "GPUs": accelerator.num_processes,
        }
    )

trainer.train()