from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import json
import torch
from utils.whisper_data_collator import WhisperDataCollator
from utils.clean_transcript import clean
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
        sampling_rate = 16000
        inputs = processor(
            audio,
            sampling_rate=sampling_rate,
            text=batch["transcription"],
            padding="longest",
            return_tensors="pt"
        )
        return {
            "input_features": inputs.input_features[0],
            "labels": inputs.labels[0]
        }
    print('preparing train')
    train_dataset = ds["train"]
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=["audio", "transcription", "__index_level_0__"], num_proc=4)
    print("prepared train, preparing dev")
    dev_dataset = ds["validation"]
    dev_dataset = dev_dataset.cast_column("audio", Audio(sampling_rate=16000))
    dev_dataset = dev_dataset.map(prepare_dataset, remove_columns=["audio", "transcription", "__index_level_0__"], num_proc=4)
    print('collating')
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
    print('training')
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{config['whisper_model']}_{language}", 
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1, 
        learning_rate=1e-5,
        warmup_steps=200,
        max_steps=5000,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
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

def munge_data(data):
    audio_paths = data[:]["audios"]
    languages = data[:]["meta"]["language"].to_list()
    duration = data[:]["meta"]["duration_ms"].to_list()
    transcripts = data[:]["transcriptions"]
    transcripts = [clean(t) for t in transcripts]
    return {
        "audio": audio_paths,
        "duration": duration, 
        "transcription": transcripts,
        "language": languages
    }


if __name__ == "__main__":
    # for language in LANGUAGES:
    lang = "aln"
    train_data = get_data(split='train', langs= None if lang == "all" else [lang])
    train = munge_data(train_data)
    print("train setup")
    dev_data = get_data(split='train', langs=None if lang == "all" else [lang])
    dev = munge_data(dev_data)
    print("dev setup")

    train = pd.DataFrame(train)
    train = train.dropna()
    train = train[train['duration'] == 0]
    train = Dataset.from_pandas(train)
    dev = pd.DataFrame(dev)
    dev = dev.dropna()
    dev = dev[dev['duration'] == 0]
    dev = Dataset.from_pandas(dev)
    dataset = DatasetDict({
        "train": train,
        "validation": dev
    })
    train_whisper(lang, dataset, False)
