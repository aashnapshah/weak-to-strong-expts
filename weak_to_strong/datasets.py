import functools
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Optional

from datasets import Dataset as HfDataset
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset, Features, Value, Sequence
import pandas as pd
import numpy as np

import glob
import json 
import random

@dataclass
class DatasetConfig:
    # split -> unshuffled dataset of items
    loader: Callable[[str], HfDataset]
    # formats items to have keys 'txt' and 'hard_label', takes a random.Random rng
    formatter: Callable[[Any], Any]
    file_path: Optional[str] = None
    folder: Optional[str] = None

# mapping from dataset name to load function and format function
_REGISTRY: dict[str, DatasetConfig] = {}


def register_dataset(name: str, config: DatasetConfig):
    _REGISTRY[name] = config

def load_dataset(ds_name: str, seed: int = 0, split_sizes: dict = None):
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)

    if ds_name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {ds_name}, please register")

    cfg = _REGISTRY[ds_name]
    results = {}
    print('Loading dataset:', ds_name)
    # Check if cfg has a file_path attribute and it ends with csv
    if hasattr(cfg, 'file_path') and cfg.file_path:
        # Load dataset from CSV file
        file_path = cfg.file_path  # Assuming you have a file path stored in cfg
        data = pd.read_csv(file_path).sample(frac=1, random_state=seed)  # Shuffle the data
        used_indices = set()  # Keep track of used indices

        if 'train' in list(data.columns):
            ds_train = data[data['train'] == 1]
            ds_test = data[data['train'] == 0]

            ## means that train/test is already built in. now, split_sizes becomes how much to subsample
            n_docs_train = split_sizes['train']
            n_docs_test = split_sizes['test']

            if split_sizes['train'] is not None:
                if len(ds_train) >= n_docs_train:
                    ds_train = ds_train.sample(n=split_sizes['train'], random_state=seed)
                else:
                    print(f"Warning: training dataset has {len(ds_train)} points and asked to subsample {n_docs_train}, using all")

            ds_train = cfg.formatter(ds_train, np.random.default_rng(seed))
            results['train'] = ds_train

            if split_sizes['test'] is not None:
                if len(ds_test) >= n_docs_test:
                    ds_test = ds_test.sample(n=split_sizes['test'], random_state=seed)
                else:
                    print(f"Warning: training dataset has {len(ds_test)} points and asked to subsample {n_docs_test}, using all")

            ds_test = cfg.formatter(ds_test, np.random.default_rng(seed))
            results['test'] = ds_test
            
    elif hasattr(cfg, 'folder') and cfg.folder:
        # Load dataset from JSON files
        train_files = glob.glob(cfg.folder + '/*train*.jsonl')[0]
        test_files = glob.glob(cfg.folder + '/*test*.jsonl')[0]
        for split, ndocs in split_sizes.items():
            if split == 'train':
                data = pd.read_json(train_files, lines=True)
                if ndocs is not None and len(data) >= ndocs:
                    data = data.sample(n=ndocs, random_state=seed)
            elif split == 'test':
                data = pd.read_json(test_files, lines=True)
                if ndocs is not None and len(data) >= ndocs:
                    data = data.sample(n=ndocs, random_state=seed)
            data = cfg.formatter(data, Random(seed))
            results[split] = data
                   
    else:
        # Load dataset from Hugging Face dataset loader
        for split, n_docs in split_sizes.items():
            ds = cfg.loader(split)
            print(f"Loaded {(ds)} examples for {ds_name} {split}")
            try:
                ds = ds.select(range(n_docs))
            except IndexError as e:
                print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
            ds = ds.map(functools.partial(cfg.formatter, rng=Random(seed)))
            ds = ds.map(
                lambda ex: {"soft_label": [1 - float(ex["hard_label"]), float(ex["hard_label"])]}
            )
            ds = ds.shuffle(seed=seed)  # shuffling a bit pointless for test set but wtv
            results[split] = ds
        
    return results
    

def tokenize_dataset(
    raw_ds: HfDataset,
    tokenizer: Callable,
    max_ctx: int,
):
    """
    This function prepares the dataset for training. It takes the raw dataset, a formatting function,
    a tokenizer, a maximum context length

    Parameters:
    raw_ds: The raw dataset to be processed.
    tokenizer: The tokenizer to be used on the formatted dataset.
    max_ctx: The maximum context length for the tokenizer.

    Returns:
    ds: The processed and shuffled dataset ready for training.
    """

    def process_function(res):
        toks = tokenizer(res["txt"])
        return dict(
            input_ids=toks["input_ids"],
        )

    ds = raw_ds.map(process_function, batched=False).filter(lambda x: len(x["input_ids"]) < max_ctx)
    return ds


def hf_loader(*hf_name, split_names=None):
    if split_names is None:
        split_names = dict()
    return lambda split: hf_load_dataset(*hf_name, split=split_names.get(split, split))


##########
# ACTUAL DATASETS
##########


def format_amazon_polarity(ex, rng):
    return dict(txt=f"{ex['title']} {ex['content']}", hard_label=ex["label"])


register_dataset(
    "amazon_polarity",
    DatasetConfig(loader=hf_loader("amazon_polarity"), 
    formatter=format_amazon_polarity),
)


def format_sciq(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    txt = f"Q: {ex['question']} A: {ans}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "sciq",
    DatasetConfig(loader=hf_loader("sciq"), 
    formatter=format_sciq),
)


def format_anthropic_hh(ex, rng):
    hard_label = int(rng.random() < 0.5)
    txt = ex["chosen"] if hard_label else ex["rejected"]
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "anthropic_hh",
    DatasetConfig(loader=hf_loader("Anthropic/hh-rlhf"), 
    formatter=format_anthropic_hh),
)


def format_cosmosqa(ex, rng):
    true_answer = ex["answer" + str(ex["label"])]
    if "None of the above choices ." in true_answer:
        hard_label = 0
    else:
        assert "None of the above choices" not in true_answer, true_answer
        hard_label = int(rng.random() < 0.5)
    if hard_label:
        answer = true_answer
    else:
        candidate_answers = [ex["answer" + str(i)] for i in range(4)]
        answer = rng.choice([x for x in candidate_answers if x != true_answer])
    txt = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "cosmos_qa",
    DatasetConfig(
        loader=hf_loader("cosmos_qa", split_names=dict(test="validation")),
        formatter=format_cosmosqa,
    ),
)


def format_boolq(ex, rng):
    hard_label = int(ex["answer"])
    txt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "boolq",
    DatasetConfig(
        loader=hf_loader("boolq", split_names=dict(test="validation")), 
        formatter=format_boolq
    ),
)

def format_pneumothorax(data, seed):
    """
    Formatting function for the dataset.
    """
    # Example formatting logic, modify as per your dataset structure
    txt = (data['report_small'] + ' ' + data['question']).tolist()
    hard_label = data['Pneumothorax'].astype(int).tolist()
    soft_label = [[1 - float(label), float(label)] for label in hard_label]

    # Define features for the dataset
    features = Features({
        'txt': Value('string'),
        'hard_label': Value('int64'),
        'soft_label': Sequence(Value('float64'))  # List of floats representing soft label probabilities
    })

    # Create a Hugging Face Dataset from the formatted data
    hf_dataset = Dataset.from_dict({
        'txt': txt,
        'hard_label': hard_label,
        'soft_label': soft_label,
    }, features=features)
    return hf_dataset

# Registering the CSV dataset
register_dataset(
    "pneumothorax",
    DatasetConfig(
        file_path= "/Users/aashnashah/Dropbox/Research/weak-to-strong-expts/data/pneumothorax.csv",
        loader=hf_loader("pneumothorax"),                        
        formatter=format_pneumothorax
    ),
)

def format_sciq(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    txt = f"Q: {ex['question']} A: {ans}"
    return dict(txt=txt, hard_label=hard_label) 


def format_medqa(ex, rng):
    # Remove correct answer from options
    ex['options'] = ex.apply(lambda row: {key: val for key, val in row['options'].items() if val != row['answer']}, axis=1)
    ex['options'] = ex['options'].apply(lambda x: {'distractor{}'.format(i+1): val for i, (key, val) in enumerate(sorted(x.items()))})
    ex['distractor1'] = ex['options'].apply(lambda x: x['distractor1']).tolist()
    ex['distractor2'] = ex['options'].apply(lambda x: x['distractor2']).tolist()
    ex['distractor3'] = ex['options'].apply(lambda x: x['distractor3']).tolist()
    ex['correct_answer'] = ex['answer'].tolist()

    # Generate random hard_label
    ex['hard_label'] = ex.apply(lambda row: int(rng.random() < 0.5), axis=1)
    ex['ans'] = ex.apply(lambda row: row["correct_answer"] if row["hard_label"] else rng.choice([row["distractor1"], row["distractor2"], row["distractor3"]]), axis=1)
    
    # Create hard and soft label columns
    txt = ('Q: ' + ex['question'] + ' A: ' + ex['ans']).tolist()
    hard_label = ex['hard_label'].tolist()
    soft_label = [[1 - label, label] for label in hard_label]

    # Define features for the dataset
    features = Features({
        'txt': Value('string'),
        'correct_answer': Value('string'),
        'hard_label': Value('int64'),
        'soft_label': Sequence(Value('float64'))  # List of floats representing soft label probabilities
    })

    # Create a Hugging Face Dataset from the formatted data
    hf_dataset = Dataset.from_dict({
        'txt': txt,
        'correct_answer': ex['correct_answer'].tolist(),
        'hard_label': hard_label,
        'soft_label': soft_label,
    }, features=features)
    return hf_dataset

    
    
    # return dict(txt=txt, hard_label=hard_label)

    # txt = data["question"].tolist() 
    # options = data["options"].tolist()
    # hard_label = data["answer"].tolist()
    # #distractors = [value for key, value in data["options"].items() if value != correct_answer]

    # print(txt)
    # print(options)
    # print(hard_label)
    # # Define features for the dataset
    # features = Features({
    #     'txt': Value('string'),
    #     'hard_label': Value('string'),
    # })

    # # Create a Hugging Face Dataset from the formatted data
    # hf_dataset = Dataset.from_dict({
    #     'txt': txt,
    #     'hard_label': hard_label,
    # }, features=features)
    # return hf_dataset


    
    # return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "medqa",
    DatasetConfig(
        folder= "data/medQA_4_options/",
        loader=hf_loader("medqa"),                     
        formatter=format_medqa
    ),
)

def format_report(data, rng):
    hard_label = data['hard_label'].astype(int).tolist()
    soft_label = [[1 - float(label), float(label)] for label in hard_label]
    txt = []
    for _, row in data.iterrows():
        h = int(row['hard_label'])
        answer = None
        if h:
            answer = row['label']
        else:
            answer = rng.choice([row['distraction1'], row['distraction2']])

        txt.append(f"Report: {row['report_small']}\n\nAnswer: {answer}")


    # Define features for the dataset
    features = Features({
        'txt': Value('string'),
        'hard_label': Value('int64'),
        'soft_label': Sequence(Value('float64'))  # List of floats representing soft label probabilities
    })

    # Create a Hugging Face Dataset from the formatted data
    hf_dataset = Dataset.from_dict({
        'txt': txt,
        'hard_label': hard_label,
        'soft_label': soft_label,
    }, features=features)

    return hf_dataset


register_dataset(
    "cxr-report",
    DatasetConfig(
        file_path= "data/report_multiclass.csv",
        loader=hf_loader("cxr-report"),
        formatter=format_report,
    ),
)

register_dataset(
    "cxr-report-2",
    DatasetConfig(
        file_path= "data/report_multiclass_v2.csv",
        loader=hf_loader("cxr-report"),
        formatter=format_report,
    ),
)

def format_bbq(data, rng):
    hard_label = data['hard_label'].astype(int).tolist()
    soft_label = [[1 - float(label), float(label)] for label in hard_label]
    txt = []
    for _, row in data.iterrows():
        h = int(row['hard_label'])
        answer = None
        if h:
            answer = row[f'ans{row["label"]}']
        else:
            answer_idx = rng.choice([i for i in range(3) if i != int(row['label'])])
            answer = row[f'ans{answer_idx}']

        txt.append(f"Context: {row['context']}\nQuestion: {row['question']}\nAnswer: {answer}")

    # Define features for the dataset
    features = Features({
        'txt': Value('string'),
        'hard_label': Value('int64'),
        'soft_label': Sequence(Value('float64'))  # List of floats representing soft label probabilities
    })

    # Create a Hugging Face Dataset from the formatted data
    hf_dataset = Dataset.from_dict({
        'txt': txt,
        'hard_label': hard_label,
        'soft_label': soft_label,
    }, features=features)

    return hf_dataset


register_dataset(
    "bbq",
    DatasetConfig(
        file_path= "data/bbq.csv",
        loader=hf_loader("bbq"),
        formatter=format_bbq,
    ),
)

def format_shadr(data, rng):
    hard_label = data['hard_label'].astype(int).tolist()
    soft_label = [[1 - float(label), float(label)] for label in hard_label]
    txt = []
    for _, row in data.iterrows():
        h = int(row['hard_label'])
        answer = None
        if h:
            answer = row['label']
        else:
            answer = rng.choice([row['distraction1'], row['distraction2']])

        txt.append(f"Sentence: {row['text']}\n\n Is the following social determinant of health is associated with the sentence above? Answer: {answer}")


    # Define features for the dataset
    features = Features({
        'txt': Value('string'),
        'hard_label': Value('int64'),
        'soft_label': Sequence(Value('float64'))  # List of floats representing soft label probabilities
    })

    # Create a Hugging Face Dataset from the formatted data
    hf_dataset = Dataset.from_dict({
        'txt': txt,
        'hard_label': hard_label,
        'soft_label': soft_label,
    }, features=features)

    return hf_dataset


register_dataset(
    "shadr",
    DatasetConfig(
        file_path= "data/shadr.csv",
        loader=hf_loader("shadr"),
        formatter=format_shadr,
    ),
)

VALID_DATASETS: list[str] = list(_REGISTRY.keys())

# """ TEST 
# from datasets import disable_caching
# disable_caching()

# from weak_to_strong.datasets import load_dataset, VALID_DATASETS
# import numpy as np

# ds_name = "pneumothorax"
# print(VALID_DATASETS)

# ds = load_dataset(ds_name, split_sizes=dict(train=500, test=10))
# train = list(ds['train'])
# test = list(ds['test'])
# print(test[0])
# print(np.mean([x['hard_label'] for x in train]))
# """
