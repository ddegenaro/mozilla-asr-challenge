import os
import json
from typing import Union, Iterable

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import WhisperProcessor

from utils.clean_transcript import clean
from utils.lang_maps import ROOT, LANGUAGES, HR_ROOT, HR_MAP, ALL_TARGETS

class SpeechDataset(Dataset):

    def __init__(
        self,
        split: str,
        langs: Iterable[str],
        df: pd.DataFrame
    ):
        super().__init__()
        self.split = split
        self.langs = langs
        self.df = df
        self.config = json.load(open('config.json', 'r', encoding='utf-8'))
        self.processor = WhisperProcessor.from_pretrained(
            self.config['whisper_model']
        )

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

        rows = pd.DataFrame(self.df.loc[index]).transpose()

        audio_files = rows['audio_file']
        
        langs = []
        for audio_file in audio_files:
            if "el-CY" in audio_file:
                lang = "el-CY"
            else:
                lang = audio_file.replace('_', '-').split('-')[2]
            langs.append(lang)

        audio_paths = self.make_paths(langs, audio_files)
        audios = [
            librosa.load(audio_path, offset=0, duration=30, mono=True, sr=16_000)[0]
            for audio_path in audio_paths
        ]
        transcriptions = [clean(t) for t in rows['transcription']]

        inputs = self.processor(audio=audios, sampling_rate=16_000, return_tensors='pt')
        input_features = inputs.input_features[0].to(dtype=torch.float16 if self.config['lora'] else torch.float32)
        labels = self.processor.tokenizer(transcriptions, max_length=200, truncation=True)
        labels = {
            'input_ids': labels['input_ids'][0],
            'attention_mask': labels['attention_mask'][0]
        }

        return {
            'meta': rows,
            'audio_paths': audio_paths,
            'audios': audios,
            'input_features': input_features,
            'labels': labels,
            'transcriptions': transcriptions
        }

    def make_paths(
        self,
        langs: list[str],
        audio_files: list[str]
    ) -> list[str]:
        audio_paths = []
        for lang, audio_file in zip(langs, audio_files):
            if lang in LANGUAGES:
                audio_paths.append(
                    os.path.join(ROOT, f'sps-corpus-1.0-2025-09-05-{lang}', "audios", audio_file)
                )
            else:
                audio_paths.append(
                    os.path.join(HR_ROOT, lang, "clips", audio_file)
                )
        return audio_paths



def get_data(
    split: str = 'train',
    langs: Union[str, Iterable[str]] = None,
    clean: bool = True,
    log: bool = False,
    df_only: bool = False,
    vote_upsampling: bool = False
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
        langs = ALL_TARGETS

    dfs = []

    for lang in langs:
        assert lang in ALL_TARGETS, f'No such lang: {lang}'

        if lang in LANGUAGES:
            lang_dir = os.path.join(ROOT, f'sps-corpus-1.0-2025-09-05-{lang}')

            lang_df = pd.read_csv(
                os.path.join(lang_dir, f'ss-corpus-{lang}.tsv'),
                sep='\t', quoting=3
            )
        else:
            lang_dir = os.path.join(HR_ROOT, lang)
            if split == 'all':
                lang_df = pd.concat([
                    pd.read_csv(os.path.join(lang_dir, f'{split}.tsv'), sep='\t', quoting=3)
                    for split in ('train', 'dev', 'test', 'other', 'validated')
                ])
            else:
                loc = os.path.join(lang_dir, f'{split}.tsv')
                lang_df = pd.read_csv(loc, sep='\t', quoting=3)
            lang_df.drop_duplicates(subset=['path'], inplace=True)
            lang_df.rename(
                columns={
                    'path': 'audio_file',
                    'locale': 'language',
                    'sentence': 'transcription'
                }, inplace=True
            )
            lang_df['votes'] = lang_df['up_votes'] - lang_df['down_votes']
            durations = pd.read_csv(
                os.path.join(lang_dir, 'clip_durations.tsv'), sep='\t', quoting=3
            )
            durations.rename(
                columns = {
                    'clip': 'audio_file',
                    'duration[ms]': 'duration_ms'
                }, inplace=True
            )
            lang_df = pd.merge(
                lang_df, durations
            )

        # remove "reported" audios
        if os.path.exists(os.path.join(lang_dir, f'ss-reported-audios-{lang}.tsv')):
            try:
                reported_df = pd.read_csv(
                    os.path.join(lang_dir, f'ss-reported-audios-{lang}.tsv'),
                    sep='\t', quoting=3
                )
                reported_audio_files = reported_df["audio_file"].to_list()
                lang_df = lang_df[~lang_df['audio_file'].isin(reported_audio_files)]
            except Exception as e:
                print(e)
                print("couldn't read reported audio files for: ", lang)

        all_len = len(lang_df)
        if log:
            print(f'Found {all_len} rows for {lang}.')

        if lang in LANGUAGES:
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

        if lang in LANGUAGES:
            dfs.append(lang_df[lang_df['split'] == split])
        else:
            dfs.append(lang_df)

    df = pd.concat(dfs, ignore_index=True)
    
    if vote_upsampling:
        addl_dfs = []
        for vote_count in df['votes'].unique():
            addl_df = df[df['votes'] == vote_count]
            for _ in range(vote_count):
                addl_dfs.append(addl_df)
        final_df = pd.concat([df] + addl_dfs).reset_index(drop=True)
    else:
        final_df = df

    if df_only:
        return final_df
    else:
        return SpeechDataset(split, langs, final_df)



def get_data_high_resource(
    split: str = 'all',
    langs: Union[str, Iterable[str]] = None,
    clean: bool = True,
    log: bool = False,
    df_only: bool = False,
    multilingual_drop_duplicates = True,
    vote_upsampling: bool = False,
    limit_rows: int = 100_000
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

    assert split in ('train', 'dev', 'test', 'other', 'validated', 'all'), f'Unknown split: {split}, must be train, dev, test, other, validated, or all.'

    if type(langs) == str:
        langs = [langs]
    elif langs is None:
        langs = LANGUAGES

    dfs = []

    for lang in sorted(langs):
        assert lang in ALL_TARGETS, f'No such lang: {lang}'
        hr_lang_dirs = sorted([
            os.path.join(HR_ROOT, hr_lang) for hr_lang in HR_MAP[lang] if hr_lang != 'msi'
        ])

        hr_lang_dfs = []
        for i, hr_lang_dir in enumerate(hr_lang_dirs):
            if log:
                print(f'Extracting data from {HR_MAP[lang][i]}...')
            if split == 'all':
                hr_lang_df = pd.concat([
                    pd.read_csv(os.path.join(hr_lang_dir, f'{split}.tsv'), sep='\t', quoting=3)
                    for split in ('train', 'dev', 'test', 'other', 'validated')
                ])
            else:
                loc = os.path.join(hr_lang_dir, f'{split}.tsv')
                hr_lang_df = pd.read_csv(loc, sep='\t', quoting=3)
            hr_lang_df.drop_duplicates(subset=['path'], inplace=True)
            hr_lang_df.rename(
                columns={
                    'path': 'audio_file',
                    'locale': 'language',
                    'sentence': 'transcription'
                }, inplace=True
            )
            hr_lang_df['votes'] = hr_lang_df['up_votes'] - hr_lang_df['down_votes']
            durations = pd.read_csv(
                os.path.join(hr_lang_dir, 'clip_durations.tsv'), sep='\t', quoting=3
            )
            durations.rename(
                columns = {
                    'clip': 'audio_file',
                    'duration[ms]': 'duration_ms'
                }, inplace=True
            )
            merged = pd.merge(
                hr_lang_df, durations
            )
            hr_lang_dfs.append(merged)

        # if 'msi' in HR_MAP[lang]:
        #     hr_lang_dfs.append()
        
        if hr_lang_dfs:
            lang_df = pd.concat(hr_lang_dfs)
        else:
            lang_df = None
            print(f'No auxiliary data found for {lang}.')

        if lang_df is not None:
            dfs.append(lang_df)
        else:
            continue

        all_len = len(lang_df)
        if log:
            print(f'Found {all_len} rows for {lang} using: {HR_MAP[lang]}.')

        if clean:
            lang_df = clean_df(lang_df)
            new_len = len(lang_df)
            if log and (all_len > new_len):
                print(f'Purged {all_len - new_len} rows. New length: {new_len}\n')
        elif log:
            print('\n')

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    if multilingual_drop_duplicates:
        df.drop_duplicates(subset=['audio_file'], inplace=True)
    
    if vote_upsampling:
        addl_dfs = []
        for vote_count in df['votes'].unique():
            addl_df = df[df['votes'] == vote_count]
            for _ in range(vote_count):
                addl_dfs.append(addl_df)
        final_df = pd.concat([df] + addl_dfs).reset_index(drop=True)
    else:
        final_df = df
        
    if limit_rows is not None:
        
        # Source - https://stackoverflow.com/a
        # Posted by Kris, modified by community. See post 'Timeline' for change history
        # Retrieved 2025-11-18, License - CC BY-SA 4.0
        final_df = final_df.sample(frac=1).reset_index(drop=True)[:limit_rows]

    if df_only:
        return final_df
    else:
        return SpeechDataset(split, langs, final_df)



def clean_df(lang_df: pd.DataFrame):

    lang_df['transcription'] = lang_df['transcription'].fillna('')
    lang_df = lang_df.loc[lang_df['transcription'].str.strip() != '']

    lang_df = lang_df.loc[lang_df['duration_ms'] > 0]

    return lang_df
