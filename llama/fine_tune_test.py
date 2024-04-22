import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

import tempfile
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

import sys
sys.path.append('../')
import json
import os
from typing import Dict, List, Optional, Sequence, Union

import datetime 
import fire
import numpy as np
import torch

import weak_to_strong.logger as logger
from weak_to_strong.utils import get_tokenizer
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset, tokenize_dataset)
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model

os.chdir('../')

access_token = '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'
download_directory = '/n/groups/patel/aashna/weak-to-strong-expts/tmp'
# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

# Load dataset
seed = 0
dataset = load_dataset('medqa', seed=seed, split_sizes=dict(train=10, test=1000))

# Split the training dataset in half
train_ds, test_ds = dataset["train"], dataset["test"]
split_data = train_ds.train_test_split(test_size=0.5, seed=seed)
train1_ds, train2_ds = split_data["train"], split_data["test"]
print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
   # num_labels=2,
    quantization_config=quant_config,
    device_map={"": 0}, 
    cache_dir=download_directory, 
   # token=access_token
)

model.config.use_cache = False
model.config.pretraining_tp = 1

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="SEQ_CLS",
)

new_model = '/n/groups/patel/aashna/weak-to-strong-expts/llama/medQA'
if os.path.exists(new_model):
    print('Loading PreTrained Model')
    model = AutoModelForCausalLM.from_pretrained(new_model, cache_dir=download_directory) # token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(new_model)
    
else:
    print("Training")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir=download_directory)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_params = TrainingArguments(
        output_dir="results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train1_ds,
        peft_config=peft_params,
        dataset_text_field="txt",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    
    torch.cuda.empty_cache()
    trainer.train()

    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)
    
logging.set_verbosity(logging.CRITICAL)
prompt = """A 39-year-old woman is brought to the emergency department because of fevers, 
    chills, and left lower quadrant pain. Her temperature is 39.1°C (102.3°F), pulse is 126/min, 
    respirations are 28/min, and blood pressure is 80/50 mm Hg. There is blood oozing around the 
    site of a peripheral intravenous line. Pelvic examination shows mucopurulent discharge from 
    the cervical os and left adnexal tenderness. Laboratory studies show:\nPlatelet count 
    14,200/mm3\nFibrinogen 83 mg/mL (N = 200–430 mg/dL)\nD-dimer 965 ng/mL (N < 500 ng/mL) 
    When phenol is applied to a sample of the patient's blood at 90°C, a phosphorylated N-acetylglucosamine 
    dimer with 6 fatty acids attached to a polysaccharide side chain is identified. A blood 
    culture is most likely to show which of the following? A: "Lactose-fermenting, gram-negative rods 
    forming pink colonies on MacConkey agar. If correct, respond with 1. If incorrect, respond with 0."""
    
# pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer) #, max_length=512)
# result = pipe(f"<s>[INST] {prompt} [/INST]")
pipe = pipeline("text-classification")
print(pipe([prompt]))

result = pipe([prompt])
print(result)
print(result[0])

# def evaluate_model(model, tokenizer, test_dataset):
#     # Define a pipeline for text classification
#     pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)

#     # Initialize variables for evaluation
#     total_samples = len(test_dataset)
#     correct_predictions = 0

#     # Iterate over the test dataset and evaluate
#     for example in test_dataset:
#         text = example["txt"]
#         label = example["hard_label"]

#         # Make predictions
#         prediction = pipe(text)[0]

#         # Check if the predicted label matches the true label
#         if prediction["label"] == label:
#             correct_predictions += 1

#     # Calculate accuracy
#     accuracy = correct_predictions / total_samples

#     print(f"Accuracy on test set: {accuracy:.2%}")

# evaluate_model(model, tokenizer, test_ds)