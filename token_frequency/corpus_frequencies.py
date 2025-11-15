import os

import numpy as np
from transformers import WhisperProcessor

from scripts.get_data import get_data, get_data_high_resource
from utils.lang_maps import ALL_TARGETS, HR_MAP

tokenizer = WhisperProcessor.from_pretrained('openai/whisper-large-v3').tokenizer

for lang in ALL_TARGETS:
    
    path_to_save = os.path.join('token_frequency', 'by_language', f'{lang}.npy')
    
    if not os.path.exists(path_to_save):
        
        print(f'Reading {lang}...')
    
        lang_transcripts = get_data(langs=lang, df_only=True)['transcription'].tolist()
        lang_transcripts += get_data(langs=lang, df_only=True)['transcription'].tolist()
        
        print('\tDone. Tokenizing...')
        
        seqs = tokenizer(
            lang_transcripts,
            add_special_tokens=False,
            return_attention_mask=False
        )['input_ids']
        
        print('\tDone. Counting...')
        
        counts = np.zeros(tokenizer.vocab_size)
        
        for seq in seqs:
            for token in seq:
                counts[token] += 1
                
        print('\tDone. Saving...')
                
        np.save(path_to_save, counts)
        
        print('\tDone.')
        
        del lang_transcripts, seqs, counts, path_to_save
    
    hr_langs = HR_MAP[lang]
    
    if not hr_langs:
        continue
    
    hr_path_to_save = os.path.join(
        'token_frequency',
        'by_language',
        f'{"_".join(hr_langs)}.npy'
    )
    
    if not os.path.exists(hr_path_to_save):
        
        hr_lang_transcripts = []
        for hr_lang in hr_langs:
            print(f'Reading {hr_lang}...')
            hr_lang_transcripts += get_data_high_resource(
                langs=lang,
                df_only=True
            )['transcription'].tolist()
        
        print('\tDone. Tokenizing...')
        
        try:
            seqs = tokenizer(
                hr_lang_transcripts,
                add_special_tokens=False,
                return_attention_mask=False
            )['input_ids']
        except:
            print(f'ERROR: {hr_path_to_save}')
            continue
        
        print('\tDone. Counting...')
        
        counts = np.zeros(tokenizer.vocab_size)
        
        for seq in seqs:
            for token in seq:
                counts[token] += 1
                
        print('\tDone. Saving...')
                
        np.save(hr_path_to_save, counts)
        
        print('\tDone.')
        
        del hr_lang_transcripts, seqs, counts