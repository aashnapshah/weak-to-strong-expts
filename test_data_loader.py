from datasets import disable_caching
disable_caching()
import sys
sys.path.append('../')

from weak_to_strong.datasets import load_dataset, VALID_DATASETS
import numpy as np
print(VALID_DATASETS)

ds_name = "shadr"
print(ds_name)
ds = load_dataset(ds_name, split_sizes=dict(train=10000, test=1000))

# ds_name = "medqa"
# ds = load_dataset(ds_name, split_sizes=dict(train=1000, test=10), seed=42) 
# ds2 = load_dataset(ds_name, split_sizes=dict(train=1000, test=10), seed=42) 

print(ds)
train = list(ds['train'])
test = list(ds['test'])
print(test[0]['txt'])
print(np.mean([x['hard_label'] for x in train]))

# train2 = list(ds2['train'])
# test2 = list(ds2['test'])
# print(test2[0])
# print(np.mean([x['hard_label'] for x in train2]))

# # print(train2 == train)
# # print(test2 == test)
# # print(train2[0] == train[0])
# # print(test2[0] == test[0])
# # print(train2[0]['hard_label'] == train[0]['hard_label'])

