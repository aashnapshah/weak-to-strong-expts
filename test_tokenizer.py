import json
import os
from typing import Dict, List, Optional, Sequence, Union

import datetime 
import fire
import numpy as np
import torch
import pandas as pd
import weak_to_strong.logger as logger
from weak_to_strong.utils import get_tokenizer
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset, tokenize_dataset)
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model

# Load dataset
seed = 0
dataset = load_dataset('medqa', seed=seed, split_sizes=dict(train=10173, test=1000))

tokenizer = get_tokenizer('EleutherAI/pythia-70m') 

# Split the training dataset in half
train_ds, test_ds = dataset["train"], dataset["test"]
max_ctx = 512
split_data = train_ds.train_test_split(test_size=0.5, seed=seed)
train1_ds, train2_ds = split_data["train"], split_data["test"]
print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))

def check_tokenization(data, tokenizer, max_ctx):
    original_lengths = []
    tokenized_lengths = []

    train_ds = tokenize_dataset(data, tokenizer, max_ctx)
    print(data, train_ds)
    
    for _, sample in pd.DataFrame(train_ds).iterrows():
        # Get the original length of the sample
        tokens = tokenizer.encode(sample['txt'], max_length=1028) #, truncation=False)

        # Count the tokens
        num_tokens = len(tokens)
        # Print the number of tokens
        #print("Number of tokens:", num_tokens)
        
        # Check if any words are being cut off
        if num_tokens > 512:
            #print()
            print("Some words are being cut off.")
        else:
            print()
            #print("No words are being cut off.")

        # Print the tokens (for checking truncated words)
        #print("Tokens:", tokens)

    return original_lengths, tokenized_lengths

# Check tokenization for train dataset
train_original_lengths, train_tokenized_lengths = check_tokenization(train_ds, tokenizer, max_ctx)

# Check tokenization for test dataset
test_original_lengths, test_tokenized_lengths = check_tokenization(test_ds, tokenizer, max_ctx)

# Print statistics
print("Train Dataset:")
print(f"Max Original Length: {max(train_original_lengths)}")
print(f"Max Tokenized Length: {max(train_tokenized_lengths)}")
print()
print("Test Dataset:")
print(f"Max Original Length: {max(test_original_lengths)}")
print(f"Max Tokenized Length: {max(test_tokenized_lengths)}")
print()


# Your text with 2095 words
text = train_ds['txt'][0]
print(text)

# Tokenize the text
tokens = tokenizer.encode(text, max_length=512, truncation=False)

# Count the tokens
num_tokens = len(tokens)

# Check if any words are being cut off
if num_tokens > 512:
    print("Some words are being cut off.")
else:
    print("No words are being cut off.")

# Print the number of tokens
print("Number of tokens:", num_tokens)

# Print the tokens (for checking truncated words)
print("Tokens:", tokens)
