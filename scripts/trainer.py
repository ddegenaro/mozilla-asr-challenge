from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import json
import torch
from utils.whisper_data_collator import WhisperDataCollator
import evaluate
from datasets import Dataset, DatasetDict, Audio
from peft import LoraConfig, get_peft_model
from scripts.get_data import LANGUAGES, get_data
import librosa
import pandas as pd

with open("config.json", "r") as f:
    config = json.load(f)
    f.close()

def train_whisper(language:str, ds:Dataset, lora:bool=False):
    model = WhisperForConditionalGeneration.from_pretrained(config["whisper_model"])
    if lora:
        # TODO: quantize? lets start with no?
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            # target_modules=[] # this is where we can freeze layers/not target them in the LoRA
        )       
        model = get_peft_model(model, lora_config)
    if language == "all":
        processor = WhisperProcessor.from_pretrained(config["whisper_model"])
    else:
        processor = WhisperProcessor.from_pretrained(config["whisper_model"], language=language)

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        batch.drop("audio")
        return batch
    train_dataset = ds["train"].map(prepare_dataset, num_proc=4)
    dev_dataset = ds["validation"].map(prepare_dataset, num_proc=4)

    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    wer = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{config['whisper_model']}_{language}", 
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1, 
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    trainer.model.save_pretrained(f"{config['whisper_model']}_{language}/final")

if __name__ == "__main__":
    # for language in LANGUAGES:
    lang = "all"
    train_data = get_data(split='train', langs= None if lang == "all" else [lang])
    train_audio_paths = train_data[:]["audios"]
    train_languages = train_data[:]["meta"]["language"].to_list()
    train_transcripts = train_data[:]["transcriptions"]
    train_audios = []
    for p in train_audio_paths:
        try: 
            y, sr = librosa.load(p)
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            train_audios.append(y)
        except Exception as e:
            print("could not load audio in file: ", p)
            train_audios.append(None)

    train = {"audio": train_audios,
             "transcription": train_transcripts,
             "language": train_languages
             }

    dev_data = get_data(split='train', langs=None if lang == "all" else [lang])
    dev_audio_paths = dev_data[:]["audios"]
    dev_audios = []
    for p in dev_audio_paths:
        try: 
            y, sr = librosa.load(p)
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            dev_audios.append(y)
        except Exception as e:
            print("could not load audio in file: ", p)
            dev_audios.append(None)
    dev_languages = dev_data[:]["meta"]["language"].to_list()
    dev_transcripts = dev_data[:]["transcriptions"]
    dev = {"audio": dev_audios,
            "transcription": dev_transcripts,
            "language": dev_languages
            }
    train = pd.DataFrame(train)
    train = train.dropna()
    train = Dataset.from_pandas(train)
    # train = train.cast_column("audio", Audio())
    dev = pd.DataFrame(dev)
    dev = Dataset.from_pandas(dev)
    dev = dev.dropna()
    # dev = dev.cast_column("audio", Audio())
    dataset = DatasetDict({
        "train": train,
        "validation": dev
    })
    train_whisper(lang, dataset, False)
