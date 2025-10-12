# Code for Mozilla Low-Resource ASR Challenge

## Getting data

```python

from get_data import get_data

data = get_data(split='train', langs=None) # all training data

data = get_data(split='dev', langs='el-CY') # dev data for Cypriot Greek

data[:5]['meta'] # metadata for first 6 rows (pd.DataFrame)
data[0]['audios'] # absolute path to first audio (list[str])
data[7:10]['transcriptions'] # transcriptions of rows 7-10 (list[str])
```

## Initial Trainer
This just takes the data for a specific language or all of the languages and finetunes a whisper model (or whatever you set in the config) on that language.

Model and languages set in the config. To train on all languages set language to `all`

Run on HPC using slurm
```bash
sbatch scripts/slurm/train.sh
```

run locally using `scripts/trainer.py`

