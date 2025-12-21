import pandas as pd
from tqdm import tqdm
import json
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
from torch.utils.data import DataLoader
from utils.whisper_data_collator import WhisperDataCollator
from utils.lang_maps import ALL_TARGETS
import torch
from utils.clean_transcript import clean
from peft import PeftModel
import gc
from scripts.get_data import get_data
import librosa

def generate(model, data, processor, proxy_lang):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dtype = next(model.parameters()).dtype
    model.to(device).eval()
    collator = WhisperDataCollator(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=proxy_lang, task="transcribe")
    predictions = []
    filepaths = []
    with torch.no_grad():
        for filepath in tqdm(data):      
            audio = librosa.load(data[i]["audio_paths"][0], offset=0, mono=True, sr=16_000)[0]
            # chunk duration 30 seconds
            chunk_duration = 30
            chunk_samples = int(chunk_duration * 16_000)
            chunks = [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]
            inputs = processor(audio=chunks, sampling_rate=16_000, return_tensors='pt')
            input_features = inputs.input_features.to(model.device)
            input_features = input_features.to(dtype=input_dtype)      
            
           # Generate output token IDs
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=200
            )
            transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            predictions += transcriptions
            filepaths.append(filepath)
    model.to("cpu")
    return predictions, filepaths


def get_model(config, model_dir, lang):
    if config['lora']:  
        # quantize
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = WhisperForConditionalGeneration.from_pretrained(config["whisper_model"], quantization_config=bnb_config)
        model = PeftModel.from_pretrained(model, f"{model_dir}/{lang}")        
        model.print_trainable_parameters()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(f"{model_dir}/{lang}")
    return model 

def main(config):
    audio_folder = "/home/drd92/mozilla-asr-challenge/mdc_asr_shared_task_test_data/audios/"
    model_dir = "final_models"
    for lang in tqdm(ALL_TARGETS):                
        print(lang)
        proxy_lang = config["proxy_langs"][lang]
        processor = WhisperProcessor.from_pretrained(config["whisper_model"], language=proxy_lang, task="transcribe")
         
        print("loading data filepaths")
        
        data = pd.read_csv(f"mdc_asr_shared_task_test_data/small-model/{lang}.tsv" , delimiter="\t").audio_file.to_list()
        data = [audio_folder + d for d in data]
        print("loaded data")
        model = get_model(config, model_dir, lang)
        print("retrieved model")
        predictions, filepaths = generate(model, data, processor, proxy_lang)
        predictions = [clean(p) for p in predictions]
        rows = []
        for i, p in enumerate(predictions):
            rows.append([filepaths[i].split("/")[-1], p])
        lang_df = pd.DataFrame(rows, columns=["audio_file", "sentence"])
        lang_df.to_csv(f"test_whisper-large/{lang}.tsv", index=False, sep="\t")
        del model
        gc.collect()


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
        f.close()
    main(config)
