import functools
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Optional

from datasets import Dataset as HfDataset
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset, Features, Value, Sequence
import pandas as pd


@dataclass
class DatasetConfig:
    # split -> unshuffled dataset of items
    loader: Callable[[str], HfDataset]
    # formats items to have keys 'txt' and 'hard_label', takes a random.Random rng
    formatter: Callable[[Any], Any]
    file_path: Optional[str] = None 

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
    # Check if cfg has a file_path attribute
    if hasattr(cfg, 'file_path') and cfg.file_path:
        # Load dataset from CSV file
        file_path = cfg.file_path  # Assuming you have a file path stored in cfg
        data = pd.read_csv(file_path).sample(frac=1, random_state=seed)  # Shuffle the data
        used_indices = set()  # Keep track of used indices

        if split_sizes['train'] is None:
            ## means that train/test is already built in
            results['train'] = data[data['train'] == 1]
            results['test'] = data[data['train'] == 0]

        else:
            for split, n_docs in split_sizes.items():
                print(f"Creating split: {split}")
                if n_docs is not None and len(data) <= n_docs:
                    remaining_indices = set(data.index) - used_indices  # Exclude used indices
                    sample_indices = data.loc[list(remaining_indices)].sample(n=n_docs, random_state=seed).index
                    ds = data.loc[sample_indices].copy()  # Make a copy to avoid modifying original data
                    used_indices.update(sample_indices)  # Update used indices
                else:
                    print(f"Warning: {ds_name} has less than {n_docs} docs, using all")
                    ds = data.copy()  # Use all data if n_docs is None or exceeds dataset size
                
                ds = cfg.formatter(ds, seed)
                results[split] = ds 
                   
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


def format_reportmulti(ex, rng):
    hard_label = int(ex['hard_label'])
    if hard_label:
        answer = ex['label']
    else:
        answer = rng.choice([ex['distraction1'], ex['distraction2']])

    txt = f"Report: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {answer}"

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


register_dataset(
    "reportmulti",
    DatasetConfig(
        file_path= "/Users/nathanjo/Dropbox (MIT)/MLHC_final_project/report_multiclass.csv",
        loader=hf_loader("reportmulti"),
        formatter=format_reportmulti,
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
