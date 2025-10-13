from typing import Any, Dict, List, Union
import torch
from transformers import WhisperProcessor


class WhisperDataCollator:
    """
        I get by with a little help from my friends (chatGPT)
    """
    def __init__(
        self,
        processor: WhisperProcessor,
        feature_size: int = 80,
        max_length: int = 3000,
        padding_value: float = 0.0,
        label_pad_token_id: int = -100,
    ):
        self.processor = processor
        self.feature_size = feature_size  # 80 mel bins
        self.max_length = max_length      # 3000 time frames
        self.padding_value = padding_value
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [torch.tensor(f["input_features"]) for f in features]
        labels = [f["labels"] for f in features]

        # Pad/truncate input_features to (80, 3000)
        padded_inputs = []
        for feat in input_features:
            # feat shape: (80, N)
            n_frames = feat.shape[-1]
            if n_frames > self.max_length:
                feat = feat[:, :self.max_length]
            elif n_frames < self.max_length:
                pad_width = self.max_length - n_frames
                pad_tensor = torch.full((self.feature_size, pad_width), self.padding_value)
                feat = torch.cat((feat, pad_tensor), dim=-1)
            padded_inputs.append(feat)

        # Pad labels to the longest in batch, using -100
        max_label_len = max(len(label) for label in labels)
        padded_labels = []
        for label in labels:
            pad_len = max_label_len - len(label)
            padded = label + [self.label_pad_token_id] * pad_len
            padded_labels.append(torch.tensor(padded))

        return {
            "input_features": torch.stack(padded_inputs),
            "labels": torch.stack(padded_labels),
        }