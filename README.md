# Code for Mozilla Low-Resource ASR Challenge

## Getting data

```python

from get_data import get_data

data = get_data(split='train', langs=None) # all training data

data = get_data(split='dev', langs='el-CY') # dev data for Cypriot Greek
```
