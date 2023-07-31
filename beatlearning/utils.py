import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from typing import Union, List, Tuple, Optional

    
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

def get_hits_columns(mode: Optional[int] = 0):
    """Get column names from data based on game mode"""
    if mode is None:
        return []
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
    
    # get hits token inputs and y
    if hits_columns:
        hits_tokens, targets = [], []
        for _, row in data.iterrows():
            #hits.append([list(row[f"{col}_token_inputs"]) for col in hits_columns])
            beats = []
            ys = [] 
            for col in hits_columns:
                beats.append([e for e in row[f"{col}_token_inputs"]])
                ys += [e for e in row[col]]
            hits_tokens.append(beats)
            ys += [row[col] for col in targets_columns]
            targets.append(ys)
    else:
        hits_tokens, targets = None, []
        for _, row in data.iterrows():
            targets.append([row[col] for col in targets_columns])
        
    if device== "cuda" or (hasattr(device, "type") and device.type=="cuda"):
        audio = torch.Tensor(audio).type(torch.long).pin_memory().to(device, non_blocking=True)
        hits_tokens = None if hits_tokens is None else torch.Tensor(hits_tokens).type(torch.long).pin_memory().to(device, non_blocking=True)
        meta_data = torch.Tensor(data[meta_columns].values).pin_memory().to(device, non_blocking=True)
        targets = torch.Tensor(targets).pin_memory().to(device, non_blocking=True)
        weights = torch.Tensor(data[weights_column].values.astype(float)).pin_memory().to(device, non_blocking=True)
    else:
        audio = torch.Tensor(audio).type(torch.long).to(device)
        hits_tokens = None if hits_tokens is None else torch.Tensor(hits_tokens).type(torch.long).to(device)
        meta_data = torch.Tensor(data[meta_columns].values).to(device)
        targets = torch.Tensor(targets).to(device)
        weights = torch.Tensor(data[weights_column].values.astype(float)).to(device)
        
    return audio, hits_tokens, meta_data, targets, weights

def generate_splits(folder: str, 
                    mode: Optional[int] = 0,  # OSU maps only
                    allow_missing_ids: bool = True,  # NOTE: this can result in data leakage
                    ratio: float=.8,  # train to test ratio 
                    seed: int = 1234567) -> Tuple[torch.Tensor]:
    extension = ".beat"
    np.random.seed(seed)
    
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(extension)]
    all_data = None
    for file in tqdm(files):
        data = pd.read_parquet(file)
        if allow_missing_ids:
            ids = [i for i in data["id"].unique() if i]
            if not ids:
                data["id"] = data["augmented"].apply(lambda x: f"{file}-{x}")
        if mode is not None:
            # NOTE: you might not want ALL data with mode = None
            data = data.loc[data["mode"] == mode]
        all_data = pd.concat([all_data, data], ignore_index=True)
    assert len(all_data), f"No converted OSZ files found. Make sure to extract and convert them to {extension} data format first!"
    ids = [i for i in all_data["id"].unique() if i]
    assert ids, "No BeatmapIDs to split!"
    train_ids = np.random.choice(ids, int(len(ids) * ratio), replace=False)
    test_ids = np.array([i for i in ids if i not in set(train_ids)])
    assert len(test_ids), f"Not enough BeatmapIDs to split data to {ratio * 100:.2f}% - {(1. - ratio) * 100:.2f}%"
    train = all_data.loc[all_data["id"].isin(train_ids)]
    test = all_data.loc[all_data["id"].isin(test_ids)]
    
    return train, test
