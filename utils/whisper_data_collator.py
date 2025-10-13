import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor

@dataclass
class WhisperDataCollator:
    """
        Written with help from ChatGPT
    """
    processor: WhisperProcessor
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract raw audio and text from features
        input_values = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]

        # Use processor to pad and convert inputs
        batch_inputs = self.processor.feature_extractor.pad(
            {"input_features": input_values},
            return_tensors=self.return_tensors
        )

        # Tokenize and pad labels
        batch_labels = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors=self.return_tensors
        )

        # Replace padding token IDs with -100 to ignore in loss
        batch_labels["input_ids"][batch_labels["input_ids"] == self.processor.tokenizer.pad_token_id] = -100

        # Final batch
        batch = {
            "input_features": batch_inputs["input_features"],  # (batch_size, 80, time)
            "labels": batch_labels["input_ids"]
        }

        return batch