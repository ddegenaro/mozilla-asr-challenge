from scripts.get_data import LANGUAGES, get_data
import pandas as pd
from datasets import Dataset
import json
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
import librosa
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.whisper_data_collator import WhisperDataCollator
import torch
from jiwer import wer
import numpy as np
from utils.clean_transcript import clean
from peft import LoraConfig, get_peft_model


with open("config.json", "r") as f:
    config = json.load(f)
    f.close()

def evaluate(model, data, processor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to("cuda").eval()
    collator = WhisperDataCollator(processor=processor)
    test_dataloader = DataLoader(data, batch_size=16, collate_fn=collator)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=proxy_lang, task="transcribe")
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs = batch["input_features"].to(device)

            # Generate output token IDs
            predicted_ids = model.generate(
                inputs,
                forced_decoder_ids=forced_decoder_ids
            )
            transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            labs = [np.where(l != -100, l, processor.tokenizer.pad_token_id) for l in batch["labels"]] 
            labs = processor.batch_decode(labs, skip_special_tokens=True)
            predictions += transcriptions
            labels += labs
    wers= [wer(l, p) for (l, p) in zip(labels, predictions)]
    model.to("cpu")
    return predictions, labels, wers




if __name__ == "__main__":
    model_dir = f"output_{config['whisper_model'].split('/')[1]}"
    split = 'dev'
    overall_rows = []
    for lang in LANGUAGES:
        data = get_data(split=split, langs=None if lang == "all" else [lang])
        if config['lora']: 
            lora_config = LoraConfig(
                r=config["lora_rank"],
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "embed_tokens"], 
                )       
             # quantize    
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = WhisperForConditionalGeneration.from_pretrained(config["whisper_model"], quantization_config=bnb_config)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            model = WhisperForConditionalGeneration.from_pretrained(model_str)
        proxy_lang = config["proxy_langs"][lang]
        processor = WhisperProcessor.from_pretrained(config["whisper_model"], language=proxy_lang, task="transcribe")
        predictions, labels, wers = evaluate(model, data, processor)
        predictions = [clean(p) for p in predictions]
        labels = [clean(l) for l in labels]
        overall_rows.append([lang, np.mean(wers)])
        print(np.mean(wers))
        rows = []
        for i, p in enumerate(predictions):
            rows.append([p, labels[i], wers[i]])
        lang_df = pd.DataFrame(rows, columns=["prediction", "label", "wer"])
        lang_df.to_csv(f"results/{config['whisper_model'].split('/')[1]}/{lang}_eval.csv", index=False)

    df = pd.DataFrame(overall_rows, columns=["language", "wer"])
    df.to_csv(f"results/{config['whisper_model'].split('/')[1]}/summary.csv", index=False)
