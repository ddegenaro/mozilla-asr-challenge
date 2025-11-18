import gc
import json
import torch
import os
from typing import Optional
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, BitsAndBytesConfig,EarlyStoppingCallback
from datasets import Dataset, IterableDatasetDict
from peft import LoraConfig, get_peft_model,  prepare_model_for_kbit_training
from jiwer import wer

from scripts.get_data import get_data, get_data_high_resource
from utils.whisper_data_collator import WhisperDataCollator
from utils.lang_maps import ALL_TARGETS, HR_MAP




with open("config.json", "r") as f:
    config = json.load(f)
    f.close()

def train_whisper(
    ds: Dataset,
    output_dir: str,
    lora: bool = False,
    proxy_lang: Optional[str] = None,
):
    
    if lora:
        lora_config = LoraConfig(
            r=config["lora_rank"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj","v_proj"],
        )   
        # quantize    
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            config["whisper_model"],
            quantization_config=bnb_config,
            dtype=torch.float16
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        if config['unfreeze_token_embeddings']:
            for name, param in model.named_parameters():
                if 'tokens' in name:
                    param.requires_grad = True
            
        model.print_trainable_parameters()
        
    else:
        model = WhisperForConditionalGeneration.from_pretrained(config["whisper_model"])

    processor = WhisperProcessor.from_pretrained(config["whisper_model"], language=proxy_lang, task="transcribe")
    
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

    # Force the chosen token globally:
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=proxy_lang,
        task="transcribe"
    )
    model.config.suppress_tokens = []
    model.config.lang_detection_threshold = 0.0

    print("preparing dev")
    data_collator = WhisperDataCollator(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir, 
        per_device_train_batch_size=config["batch_size"],
        learning_rate=5e-5,
        num_train_epochs=config["epochs"],
        gradient_checkpointing=False,
        fp16=True,
        eval_strategy="epoch",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=448,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_wer",
        greater_is_better=False,
        push_to_hub=False,
        gradient_accumulation_steps=1,
        report_to='wandb',
        run_name=str(output_dir.split('/')[-1]),
        project = 'mozilla-asr-challenge'
    )

    print(f'training {lang}')
    patience = 1 if config["lora"] else 5
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        compute_metrics=compute_metrics,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=0.0)]
    )
    
    trainer.train()
    
    if config["lora"]:
        trainer.model.save_pretrained(f"{output_dir}/final")
        if config['unfreeze_token_embeddings']:
            torch.save(
                trainer.model.base_model.model.model.decoder.embed_tokens.weight,
                os.path.join(output_dir, 'final', 'embeddings.pt')
            )
    else:
        trainer.save_model(f"{output_dir}/final")
    
    del model
    del ds
    del trainer

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if config["train_high_resource"]:
        # some of the languages will share HR adapters
        trained_hr_adapters = []
        for lang in ALL_TARGETS:
            hr_langs = HR_MAP[lang]
            if len(hr_langs) == 0:
                continue
            if "_".join(hr_langs) not in trained_hr_adapters:
                trained_hr_adapters.append("_".join(hr_langs))
                output_dir = f"output_{config['whisper_model'].split('/')[1]}/{'_'.join(hr_langs)}"
                if not os.path.exists(f"{output_dir}/final"): #todo change
                    train = get_data_high_resource(
                        split='train',
                        langs=[lang],
                        multilingual_drop_duplicates=False,
                        vote_upsampling=config['vote_upsampling']
                    )
                    dev = get_data_high_resource(
                        split='dev',
                        langs=[lang],
                        multilingual_drop_duplicates=False,
                        vote_upsampling=False
                    )
                    dataset = IterableDatasetDict({
                        "train": train,
                        "validation": dev
                    })
                    train_whisper(dataset, output_dir, config["lora"], config["proxy_langs"][lang])
                    

    else:
        for lang in ALL_TARGETS:
            output_dir = f"output_{config['whisper_model'].split('/')[1]}/{lang}"
            if not os.path.exists(f"{output_dir}/final"):
                train = get_data(
                    split='train',
                    langs=None if lang == "all" else [lang],
                    vote_upsampling=config['vote_upsampling']
                )
                dev = get_data(
                    split='dev',
                    langs=None if lang == "all" else [lang],
                    vote_upsampling=False
                )
                dataset = IterableDatasetDict({
                    "train": train,
                    "validation": dev
                })
                train_whisper(dataset, output_dir, config["lora"], config["proxy_langs"][lang])
            else:
                print(f"skipping language {lang}, adapter already exists")

