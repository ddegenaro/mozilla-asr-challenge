import os
from typing import Union, Iterable

import pandas as pd
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.join(
    '/home', 'drd92', 'mozilla-asr-challenge', 'mcv-sps-st-09-2025'
)

LANGUAGES = {
    'aln',
    'bew',
    'bxk',
    'cgg',
    'el-CY',
    'hch',
    'kcn',
    'koo',
    'led',
    'lke',
    'lth',
    'meh',
    'mmc',
    'pne',
    'ruc',
    'rwm',
    'sco',
    'tob',
    'top',
    'ttj',
    'ukv'
}

class SpeechDataset(Dataset):

    def __init__(self, split: str, langs: Iterable[str], df: pd.DataFrame):
        super().__init__()
        self.split = split
        self.langs = langs
        self.df = df

    def __len__(self):
        """Returns the length of the dataset in audio-text pairs.

        Returns:
            int: _The length._
        """
        return len(self.df)
    
    def __getitem__(self, index):
        """Indexing operations.

        Args:
            index: _The row(s) to select._

        Returns:
            dict: _The keys are `meta`, `audios` (absolute paths to audio files) and `transcriptions`._
        """

        if type(index) == int:
            index = [index]
        
        rows = self.df.loc[index]

        audio_files = rows['audio_file']
        langs = []
        for x in audio_files:
            if x.split('-')[2] == "el":
                langs.append("el-CY")
            else:
                langs.append(x.split('-')[2])

        audios = [
            os.path.join(ROOT, f'sps-corpus-1.0-2025-09-05-{lang}', "audios", audio_file)
            for audio_file, lang in zip(audio_files, langs)
        ]

        transcriptions = rows['transcription'].tolist()

        return {
            'meta': rows,
            'audios': audios,
            'transcriptions': transcriptions
        }

def get_data(
    split: str = 'train',
    langs: Union[str, Iterable[str]] = None,
    clean: bool = True,
    log: bool = False
) -> Union[SpeechDataset, DataLoader]:
    """Indexing operations.

    Args:
        split (_str_): _The split to return (either `train` or `dev`)._
        langs (_Union[str, Iterable[str]]_): _The languages to include. Defaults to `None`, which inludes all languages._
        clean (_bool_): _Whether to delete rows with empty transcriptions. Defaults to `True`._
        log (_bool_): _Whether to produce log messages about row counts. Defaults to `False`._

    Returns:
        DataLoader: _A dataset that supports PyTorch `Dataset` functionality (or pre-wrapped in a `DataLoader`)._
    """

    assert split in ('train', 'dev'), f'Unknown split: {split}, must be train or dev.'

    if type(langs) == str:
        langs = [langs]
    elif langs is None:
        langs = LANGUAGES

    dfs = []

    for lang in langs:
        assert lang in LANGUAGES, f'No such lang: {lang}'
        lang_dir = os.path.join(ROOT, f'sps-corpus-1.0-2025-09-05-{lang}')

        lang_df = pd.read_csv(
            os.path.join(lang_dir, f'ss-corpus-{lang}.tsv'),
            sep='\t'
        )

        all_len = len(lang_df)
        if log:
            print(f'Found {all_len} rows for {lang}.')

        lang_df = lang_df.loc[lang_df['split'] == split]
        old_len = len(lang_df)
        if log:
            print(f'{old_len} rows are from split {split}.')

        if clean:
            lang_df = clean_df(lang_df)
            new_len = len(lang_df)
            if log and (old_len > new_len):
                print(f'Purged {old_len - new_len} rows. New length: {new_len}\n')
        elif log:
            print('\n')

        dfs.append(lang_df[lang_df['split'] == split])

    df = pd.concat(dfs, ignore_index=True)

    return SpeechDataset(split, langs, df)



def clean_df(lang_df: pd.DataFrame):

    lang_df['transcription'] = lang_df['transcription'].fillna('')
    lang_df = lang_df.loc[lang_df['transcription'].str.strip() != '']

    return lang_df