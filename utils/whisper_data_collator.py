from dataclasses import dataclass
from typing import List, Dict, Union
from transformers import WhisperProcessor
import numpy as np
import torch.nn.functional as F
import torch

@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Union[List[float], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
            a lot of hackiness here to ensure that the data is in the right format because otherwise a padding error is thrown. 
            every time I try to pad using huggingface it throws an error
        """
        input = [torch.tensor(f["input_features"]) for f in features]


        # Pad the input features
        padded_inputs=[F.pad(t, (0, 3000-t.shape[1]), value=0.0) for t in input] # padding along the time (last) dimension to a fixed length of 3000
        input_features = torch.stack(padded_inputs)

        # Pad the labels
        label_sequences = [torch.tensor(f["labels"]) for f in features]
        max_lbl = max(len(l) for l in label_sequences)
        labels = torch.full((len(label_sequences), max_lbl), fill_value=-100, dtype=torch.long)
        # truncate
        for i, l in enumerate(label_sequences):
           labels[i, :len(l)] = l 
        return {
            "input_features": input_features,  
            "labels": labels,
        }
