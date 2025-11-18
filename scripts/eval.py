from scripts.get_data import get_data
import pandas as pd
import os
import json
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.whisper_data_collator import WhisperDataCollator
from utils.lang_maps import LANGUAGES, ALL_TARGETS, HR_MAP
import torch
from jiwer import wer
import numpy as np
from utils.clean_transcript import clean
from peft import PeftModel
from utils.task_vectors import TaskVector
from ax.service.ax_client import AxClient, ObjectiveProperties

with open("config_eval.json", "r") as f:
    config = json.load(f)
    f.close()

def evaluate(model, data, processor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dtype = next(model.parameters()).dtype
    model.to("cuda").eval()
    collator = WhisperDataCollator(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)
    test_dataloader = DataLoader(data, batch_size=16, collate_fn=collator)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=proxy_lang, task="transcribe")
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):            
            inputs = batch["input_features"].to(dtype=input_dtype).to(device)
            # Generate output token IDs
            predicted_ids = model.generate(
                inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=200
            )
            transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            labs = [np.where(l != -100, l, processor.tokenizer.pad_token_id) for l in batch["labels"]] 
            labs = processor.batch_decode(labs, skip_special_tokens=True)
            predictions += transcriptions
            labels += labs
    wers= [wer(l, p) for (l, p) in zip(labels, predictions)]
    model.to("cpu")
    return predictions, labels, wers


def get_model(model_dir, lang):
    if config['lora']:  
        # quantize
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = WhisperForConditionalGeneration.from_pretrained(config["whisper_model"], quantization_config=bnb_config)
        model = PeftModel.from_pretrained(model, f"{model_dir}/{lang}/final")
        
        if config['unfreeze_token_embeddings']:
            model.base_model.model.model.decoder.embed_tokens.weight = torch.load(
                os.path.join(model_dir, lang, 'final', 'embeddings.pt'),
                map_location=next(model.parameters()).device,
                weights_only=True
            )
        
        model.print_trainable_parameters()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(f"{model_dir}/{lang}/final")
    return model 

if __name__ == "__main__":
    model_dir = f"output_{config['whisper_model'].split('/')[1]}"
    split = 'dev'
    print("lora:", config["lora"])
    overall_rows = []
    for lang in ALL_TARGETS:
        data = get_data(
            split=split,
            langs=None if lang == "all" else [lang],
            vote_upsampling=False
        )
        proxy_lang = config["proxy_langs"][lang]
        processor = WhisperProcessor.from_pretrained(config["whisper_model"], language=proxy_lang, task="transcribe")

        # apply adapter or tv and get scaling coefficient
        hyperparameter_results = {}
        if config["run_merge_high_resource"] and len(HR_MAP[lang]) > 0:
            # set up hyperparameter search client
            ax_client = AxClient()
            ax_client.create_experiment(
                name="hyperparameter_search",
                parameters=[
                    {
                        "name": "scaling_coef",
                        "type": "range",
                        "bounds": [0.0, 1.0],  # Lower and upper bounds
                        "value_type": "float",
                        "log_scale": False,  # Sample on a log scale
                    },
                ],
                objectives={"wer": ObjectiveProperties(minimize=True)}
            )
            # get the task vector for the fully finetuned HR model, apply to LR model. Do a hyperparameter sweep for
            # scaling coefficient.
            hr_model_dir = f"{model_dir}/{'_'.join(HR_MAP[lang])}/final"
            
            if not config["lora"]:
                tv = TaskVector(
                            pretrained_model=WhisperForConditionalGeneration.from_pretrained(config["whisper_model"]), 
                            finetuned_model=WhisperForConditionalGeneration.from_pretrained(hr_model_dir)
                        )
            
            for i in range(config["hyperparameter_search_length"]): # set to 10
                parameters, trial_index = ax_client.get_next_trial()
                coef = parameters["scaling_coef"]
                model = get_model(model_dir, lang)
                if config["lora"]:
                    hr_adapter = model.load_adapter(hr_model_dir, adapter_name="hr_adapter")
                    lr_adapter =  model.load_adapter(f"{model_dir}/final", adapter_name="lr_adapter")
                    weighted_adapter = model.add_weighted_adapter(["lr_adapter", "hr_adapter"], 
                                                                  [1, coef], 
                                                                  adapter_name="merged", 
                                                                  combination_type="linear" # do we want to change this?
                                                                  )
                    model.set_adapter("merged")
                    _, _, wers = evaluate(model, data, processor)
                    avg_wer = np.mean(wers)
                    hyperparameter_results[coef] = avg_wer
                else:
                    model = tv.apply_to(model, scaling_coef=coef)
                    _, _, wers = evaluate(model, data, processor)
                    avg_wer = np.mean(wers)
                    hyperparameter_results[coef] = avg_wer
                ax_client.complete_trial(trial_index=trial_index, raw_data={"wer": avg_wer})
            best_lambda = min(hyperparameter_results, key=hyperparameter_results.get)
            if config["lora"]:
                model = get_model(model_dir, lang)
                hr_adapter = model.load_adapter(hr_model_dir, adapter_name="hr_adapter")
                lr_adapter =  model.load_adapter(f"{model_dir}/final", adapter_name="lr_adapter")
                weighted_adapter = model.add_weighted_adapter(["lr_adapter", "hr_adapter"], 
                                                                [1, best_lambda], 
                                                                adapter_name="merged", 
                                                                combination_type="linear" # do we want to change this?
                                                                )
                model.set_adapter("merged")
            else:
                model = get_model(model_dir, lang)
                model = tv.apply_to(model, scaling_coef=best_lambda)
        else:
            model = model = get_model(model_dir, lang)

        predictions, labels, wers = evaluate(model, data, processor)
        predictions = [clean(p) for p in predictions]
        labels = [clean(l) for l in labels]
        overall_rows.append([lang, np.mean(wers)])
        print(lang, np.mean(wers))
        rows = []
        for i, p in enumerate(predictions):
            rows.append([p, labels[i], wers[i]])
        lang_df = pd.DataFrame(rows, columns=["prediction", "label", "wer"])
        lang_df.to_csv(f"results/{config['whisper_model'].split('/')[1]}/{lang}_eval.csv", index=False)
        with open(f"results/{config['whisper_model'].split('/')[1]}/hyperparameters/{lang}.json", "w") as f:
            json.dump(hyperparameter_results, f, indent=4)
            f.close()

    df = pd.DataFrame(overall_rows, columns=["language", "wer"])
    df.to_csv(f"results/{config['whisper_model'].split('/')[1]}/summary.csv", index=False)
