import gc
import json
import torch
import os
import evaluate
import pandas as pd
import numpy as np
import librosa
from typing import Optional
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from jiwer import wer
from scripts.get_data import LANGUAGES, get_data
from utils.whisper_data_collator import WhisperDataCollator
from utils.clean_transcript import clean
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import soundfile as sf
with open("config.json", "r") as f:
    config = json.load(f)
    f.close()
class WhisperTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_dtype = next(model.parameters()).dtype
        outputs = model(
            input_features=inputs["input_features"].to(dtype=input_dtype),
            labels=inputs["labels"]
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def train_whisper(language:str, ds:Dataset, lora:bool=False, proxy_lang:Optional[str]=None):
    model = WhisperForConditionalGeneration.from_pretrained(config["whisper_model"])
    if lora:        # TODO: quantize? lets start with no?
        lora_config = LoraConfig(
            r=config["lora_rank"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], # this is where we can freeze layers/not target them in the LoRA
        )       
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.config.forced_decoder_ids = None 
    processor = WhisperProcessor.from_pretrained(config["whisper_model"], language=proxy_lang, task="transcribe")
    def prepare_dataset(batch):
        try:
            # load and resample audio data from 48 to 16kHz
            audio_path = batch["audio_paths"]
            # loading audio with librosa rather than Datasets.cast_column because Google HPC doesnt have ffmpeg loaded as a module and 
            # torch & torchcodec are throwing an error because of that.
            audio, sr = librosa.load(audio_path, offset=0, duration=30, mono=True)
            audio = audio.astype(np.float32)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)

            # Trim to max length (shouldn't be necessary)
            if audio.shape[0] > 30*16000:
                audio = audio[:30*16000] 
            sampling_rate = 16000
            inputs = processor(
                audio=audio,
                sampling_rate=sampling_rate,
                return_tensors="np"
            )
            labels = processor.tokenizer(
                batch["transcription"],
                max_length=200, # max length is 448 for whisper, but trying to cut down this should be enough.
                truncation=True
                )
            return_obj = {
                "input_features": inputs.input_features[0].tolist(),
                "labels": labels["input_ids"]
            }
            del audio, inputs, labels
            gc.collect()
            return return_obj
        except Exception as e:
            print(e)
            print("FILE: " , batch["audio_paths"])
            print("_" * 80)
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer_pred = 100 * wer(label_str,pred_str)
        print("prediction:")
        print(pred_str[0])
        print("label:")
        print(label_str[0])
        print("wer", wer_pred)
        print("_________________________\n")
        return {"wer": wer_pred}

    proxy_lang_id = processor.tokenizer.get_decoder_prompt_ids(language=proxy_lang, task="transcribe")

    # Force the chosen token globally:
    model.config.forced_decoder_ids = [[1, proxy_lang_id]]
    print("preparing dev")
    dev_dataset = ds["validation"]
    dev_dataset = dev_dataset.map(prepare_dataset, remove_columns=["audio_paths", "transcription", "language", "duration", "votes"], batch_size=8, num_proc=4)
    data_collator = WhisperDataCollator(
        processor=processor,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"output_{config['whisper_model'].split('/')[1]}/{lang}", 
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        num_train_epochs=3*config["epochs"],
        gradient_checkpointing=False,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=448,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        gradient_accumulation_steps=2
    )
    print(f'training {lang}') 
    for i in range(3):
        print('preparing train')
        train_dataset = ds["train"]
        train_dataset = train_dataset.sort("votes", reverse=True)
        train_dataset = train_dataset.select(range(math.ceil(len(ds["train"]) * ((i + 1)/3))))
        print("len: ", len(train_dataset))
        train_dataset = train_dataset.map(prepare_dataset, remove_columns=["audio_paths", "transcription", "language", "duration", "votes"], batch_size=4, keep_in_memory=False, num_proc=2)
        trainer = WhisperTrainer(
            args=training_args,
            model=model,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
            tokenizer=processor.feature_extractor,
        )
        trainer.train()
        model = trainer.model
        del train_dataset
        gc.collect()
        torch.cuda.empty_cache()
    if config["lora"]:
        trainer.model.save_pretrained(f"output_{config['whisper_model'].split('/')[1]}/{lang}/final")
    else:
        trainer.save_model(f"output_{config['whisper_model'].split('/')[1]}/{lang}/final")
    del model
    del dev_dataset
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

def munge_data(data):
    audio_paths = data[:]["audios"]
    languages = data[:]["meta"]["language"].to_list() # this will likely be helpful later
    duration = data[:]["meta"]["duration_ms"].to_list() # will use this to remove empty 
    votes = data[:]["meta"]["votes"].to_list() # will use this for curriculum learning
    transcripts = data[:]["transcriptions"]
    transcripts = [clean(t) for t in transcripts]
    return {
        "audio_paths": audio_paths,
        "duration": duration, 
        "transcription": transcripts,
        "language": languages,
        "votes": votes
   }



if __name__ == "__main__":
    lang = config["language"]
    for lang in LANGUAGES:
        if not os.path.exists(f"output_{config['whisper_model'].split('/')[1]}/{lang}/final"):
            train_data = get_data(split='train', langs= None if lang == "all" else [lang])
            train = munge_data(train_data)
            print("train setup")
            dev_data = get_data(split='dev', langs=None if lang == "all" else [lang])
            dev = munge_data(dev_data)
            print("dev setup")

            train = pd.DataFrame(train)
            train = train.dropna()
            train = train[train['duration'] > 0]
            train = Dataset.from_pandas(train)
            dev = pd.DataFrame(dev)
            dev = dev.dropna()
            dev = dev[dev['duration'] > 0]
            dev = Dataset.from_pandas(dev)
            dataset = DatasetDict({
                "train": train,
                "validation": dev
            })
            train_whisper(lang, dataset, config["lora"], config["proxy_langs"][lang])
        else:
            print(f"skipping language {lang}, adapter already exists")
