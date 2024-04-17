from datasets import disable_caching
disable_caching()

from weak_to_strong.datasets import load_dataset, VALID_DATASETS
import numpy as np
print(VALID_DATASETS)

ds_name = "boolq"
print(ds_name)
ds = load_dataset(ds_name, split_sizes=dict(train=500, test=10))

ds_name = "pneumothorax"
ds = load_dataset(ds_name, split_sizes=dict(train=500, test=10))

print(ds)
train = list(ds['train'])
test = list(ds['test'])
print(test[0])
print(np.mean([x['hard_label'] for x in train]))
