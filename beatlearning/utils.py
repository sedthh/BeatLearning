import numpy as np
import pandas as pd

import torch

from typing import Union, List, Tuple

    
def tokenize(prediction: Union[list, np.ndarray], token_reduction: int = 1) -> Union[List[int], int]:
    """Returns a single integer from list of booleans: [1., 0., 1.] --> 5"""
    def _tokenize(prediction: np.ndarray) -> int:
        if token_reduction > 1:
            # joins every Nth element: [1, 1, 0, 0, 1, 0] -> [1, 0, 1] with token_reduction = 2
            prediction = prediction.reshape(-1, token_reduction).sum(axis=1).astype(bool)
        return int("".join([str(int(val)) for val in prediction]), 2)

    if isinstance(prediction, list):
        prediction = np.array(prediction)
    if len(prediction.shape) == 2:  # batch
        return np.array([_tokenize(pred) for pred in prediction])
    else:
        return _tokenize(prediction)

def get_hits_columns(mode: int = 0):
    """Get column names from data based on game mode"""
    if mode in (0, 1):  # osu and taiko
        direction = ["a"]
    elif mode in (2, 3):  # catch and mania
        direction = ["l", "d", "u", "r"]
    else:
        raise ValueError("Unknown OSU game mode.")
    results = []
    for d in direction:
        results += [f"hits_{d}", f"holds_{d}"]
    return results

def data_to_tensor(data: pd.DataFrame, 
                   audio_column: str = "audio_tokens_1",
                   hits_columns: List[str] = ["hits_a", "holds_a"],
                   meta_columns: List[str] = ["time", "difficulty"],
                   targets_columns: List[str] = ["tempo", "offset"],
                   weights_column: str = "weight",
                   device: str = "cpu") -> Tuple[torch.Tensor]:
    """Convert dataset into multiple tensors"""
    audio = []
    max_len = 0
    for tokens in data[audio_column]:
        max_len = max(max_len, len(tokens))
    for tokens in data[audio_column]:
        tokens = tokens.tolist()
        tokens += [0 for _ in range(max_len - len(tokens))]
        audio.append(tokens)
    audio = torch.Tensor(audio).type(torch.long).to(device)
    
    meta = torch.Tensor(data[meta_columns].values).to(device)
    weights = torch.Tensor(data[weights_column].values.astype(float))
    
    # get hits token inputs and y
    hits, y = [], []
    for _, row in data.iterrows():
        hits.append([list(row[f"{col}_token_inputs"]) for col in hits_columns])
        beats = []
        targets = [] 
        for col in hits_columns:
            beats.append([e for e in row[f"{col}_token_inputs"]])
            targets += [e for e in row[col]]
        hits.append(beats)
        targets += [row[col] for col in targets_columns]
        y.append(targets)
    hits = torch.Tensor(hits).type(torch.long).to(device)
    y = torch.Tensor(y).to(device)
    
    return audio, hits, meta, y, weights
