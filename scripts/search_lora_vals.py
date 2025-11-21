
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

class WhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        input_dtype = next(model.parameters()).dtype
        inputs["input_features"] = inputs["input_features"].to(dtype=input_dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def train_whisper(
    config,
    ds: Dataset,
    output_dir: str,
    proxy_lang: Optional[str] = None,
):
    
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=2*config["lora_rank"],
        lora_dropout=0.05,
        bias="none",
        target_modules=config["modules"],
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
        generation_max_length=200,
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
    patience = 3
    trainer = WhisperTrainer(
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
    eval_results = trainer.evaluate()
    del model
    del trainer
    return eval_results

def main(config):
    lang = "ukv"
    train = get_data(
        split='train',
        langs=[lang],
        vote_upsampling=config['vote_upsampling']
    )
    dev = get_data(
        split='dev',
        langs=[lang],
        vote_upsampling=False
    )
    dataset = IterableDatasetDict({
        "train": train,
        "validation": dev
    })
    lora_ranks = [8, 64, 128]
    unfreeze_token_embeddings = [True, False]
    target_modules = [["q_proj", "v_proj"], ["q_proj", "v_proj", "k_proj", "out_proj"],  ["q_proj", "v_proj", "k_proj", "out_proj", "fc_1", "fc_2"]]
    
    results = []

    i = 0
    for rank in lora_ranks:
        for ufe in unfreeze_token_embeddings:
            for target_module in target_modules:
                output_dir = f"lora_search_{lang}/_{i}"
                config["lora_rank"] = rank
                config["modules"] = target_module
                config["unfreeze_token_embeddings"] = ufe
                eval_results = train_whisper(config, dataset, output_dir, config["proxy_langs"][lang])
                r = {
                    "lora_rank": rank,
                    "unfreeze_token_embeddings": ufe,
                    "target_modules": target_module,
                    "results": eval_results
                }
                results.append(r)
                i += 1
    with open("LoRA_sweep_results.py", "w") as f:
        json.dump(results, f, indent=4)
        f.close()




if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
        f.close()
    main(config)
