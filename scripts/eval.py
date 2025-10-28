from scripts.get_data import LANGUAGES, get_data
from scripts.trainer import munge_data
import pandas as pd
from datasets import Dataset
import json
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.whisper_data_collator import WhisperDataCollator
import torch
from jiwer import wer


with open("config.json", "r") as f:
    config = json.load(f)
    f.close()

def evaluate(model, data, processor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio_path = batch["audio_paths"]
        # loading audio with soundfile rather than Datasets.cast_column because Google HPC doesnt have ffmpeg loaded as a module and 
        # torch & torchcodec are throwing an error because of that.
        with open(audio_path, "rb") as f:
            audio, sr = librosa.load(f)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            #if audio.ndim > 1:
                # Average across the channel axis to convert to mono
            #    audio = np.mean(audio, axis=1)
            f.close()
        sampling_rate = 16000
        inputs = processor(
            audio=audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        labels = processor.tokenizer(
            batch["transcription"],
            max_length=448,
            truncation=True
            )
        return {
            "input_features": inputs.input_features[0],
            "labels": labels["input_ids"]
        }
    data = data.map(prepare_dataset, remove_columns=["audio_paths", "language", "duration", "votes"], num_proc=4)
    
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
            predictions += transcriptions
            labels += batch["transcription"]
    lang_wer = wer(labels, predictions)
    model.to("cpu")
    return lang_wer




if __name__ == "__main__":
    model_dir = "output_whisper-tiny"
    split = 'dev'
    rows = []
    for lang in LANGUAGES:
        data = get_data(split=split, langs=None if lang == "all" else [lang])
        data = munge_data(data)
        data = pd.DataFrame(data)
        data = data.dropna()
        data = data[data['duration'] > 0]
        data = Dataset.from_pandas(data)
        model_str = f"{model_dir}/{lang}/final"
        model = WhisperForConditionalGeneration.from_pretrained(model_str)
        proxy_lang = config["proxy_langs"][lang]
        processor = WhisperProcessor.from_pretrained(model_str, language=proxy_lang, task="transcribe")
        avg_wer = evaluate(model, data, processor)
        rows.append([lang, avg_wer])
    df = pd.DataFrame(rows, columns=["language", "wer"])
    df.to_csv(f"{model_dir}_summary.csv", index=False)
