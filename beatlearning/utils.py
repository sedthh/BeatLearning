import os
import json
import tempfile
from copy import deepcopy
from shutil import copyfile
from zipfile import ZipFile
import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .tokenizers import BEaRTTokenizer


class RobotsDisallowException(Exception):
   "Raised when a Beatmap is not meant to be used for training purposes"
   pass

class RobotsGeneratedException(Exception):
   "Raised when a Beatmap is AI generated and therefor should not be used as training data"
   pass


class IntermediateBeatmapFormat:

    """
    Creates intermediate representation(s) from a beatmap file


    meta = {
        "audio": "path_to_audio_file.mp3",          # path to audio, required
        "difficulty": 0.5,                          # 0. is easyest - 1. is hardest, required
        "tracks": ["LEFT", "DOWN", "UP", "RIGHT"],  # len(meta["tracks"]) in (1, 2, 4), order matters, required
        ...
    }
    data = {  
            "TIME":     [0, 10, 20, 30],  # ms, quantized equidistant vlaues for aligning data
            "TEMPO":    [0.0, 120.0, 120.0, 120.0]  # bpm, for regression (0 = mask / unknown)

            "LEFT":     [0,  0,  1,  2],  # 0 - no event, 1 - press, 2 - hold
            "DOWN":     [0,  0,  0,  2],
            "UP":       [0,  0,  1,  0],
            "RIGHT":    [0,  0,  0,  0],

            "empty":    [1,  0,  0,  0],  # countdowns or breaks without any events, 0 or 1
            "spinner":  [0,  0,  0,  0],  # special hit event, 0 or 1 (non-reserved)
            ...
    }  # pandas DataFrame as parquet, only "TIME", "TEMPO" and columns defined in meta["tracks"] are requried
    """

    def __init__(self,
                 file_path: Optional[str] = None,
                 *,
                 data: Optional[pd.DataFrame] = None,
                 meta: Optional[Dict[str, Any]] = None):
        self.file_path = file_path
        if self.file_path is not None:
            with ZipFile(self.file_path, "r") as z:
                with z.open("meta.json", "r") as f:
                    self.meta = json.load(f)
                self.data = pd.read_parquet(z.open("data.pq"))
            assert isinstance(self.meta, dict) and self.meta, "No metadata provided!"
        elif data is not None:
            assert isinstance(meta, dict) and meta, "No metadata provided!"
            self.meta = deepcopy(meta)
            if not isinstance(data, pd.DataFrame):
                self.data = pd.DataFrame(data)
            else:
                self.data = data.copy()

        # check if both meta and data have sufficient information
        assert "audio" in self.meta and isinstance(self.meta["audio"], str), "Malformed or missing 'audio' metadata!"
        assert "difficulty" in self.meta and isinstance(self.meta["difficulty"], float), "Malformed or missing 'difficulty' metadata!"
        assert "tracks" in self.meta and isinstance(self.meta["tracks"], list), "Malformed or missing 'tracks' metadata!"
        assert "TIME" in self.data.columns, "Missing 'TIME' (ms) column from events DataFrame!"
        assert len(self.data["TIME"].diff().unique()) <= 2, "The values in the 'TIME' column must be quantized!"
        assert "TEMPO" in self.data.columns, "Missing 'TEMPO' (bpm) column from events DataFrame!"
        
        self.data = self.data.sort_values(by=["TIME"], ascending=True).reset_index(drop=True)
        self.data["TEMPO"] = np.where(np.isfinite(self.data["TEMPO"].values), self.data["TEMPO"].values, 0.0)
        for track in self.meta["tracks"]:
            assert track in self.data.columns, f"Missing '{track}' column from events DataFrame!"
            self.data[track] = self.data[track].astype(np.uint8)
    
    def __len__(self) -> int:
        return len(self.data)

    def save(self, output_path: str) -> None:
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            meta_path = os.path.join(temp_dir, "meta.json")
            data_path = os.path.join(temp_dir, "data.pq")
            
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False)
            self.data.to_parquet(data_path, engine="pyarrow")
            
            zip_file = os.path.join(temp_dir, "ibf.zip")
            with ZipFile(zip_file, "w") as zip_object:
                zip_object.write(meta_path, arcname="meta.json")
                zip_object.write(data_path, arcname="data.pq")
                
            copyfile(zip_file, output_path)
    

class BEaRTDataset(Dataset):

    """
    Generate a compact, cached representation from the intermediate beatmap representation, 
    then convert the compact representation into actual tensors for the model on the fly
    """

    def __init__(self, 
                 tokenizer: BEaRTTokenizer, 
                 augment: bool=False):
        self.tokenizer = tokenizer
        self.audio_cnt = 0
        self.audio_data = {}
        self.df_data = pd.DataFrame({"data": pd.Series(dtype=np.uint8),
                                     "tempo": pd.Series(dtype=float),
                                     "tracks": pd.Series(dtype=int),
                                     "audio_position": pd.Series(dtype=int),
                                     "audio_cnt": pd.Series(dtype=int)})
        self.segments = {tracks: self.tokenizer._generate_segment(tracks) for tracks in self.tokenizer.config.track_combinations}
        self.masks = {tracks: self.tokenizer._generate_mask(tracks) for tracks in self.tokenizer.config.track_combinations}
        self.generator = np.random.default_rng(self.tokenizer.config.random_seed)
        self.augment = augment

    def add(self,
            intermediate_files: List[IntermediateBeatmapFormat],
            audio_path: str,
            *,
            offsets: List[float] = [0.0],
            max_empty_interval: Optional[float] = 4.0,
            ignore_lead_in: bool = False) -> None:
        """Generates training data from intermediate file representations, that use the same audio file,
           skpping empty areas and applying optional offsets for augmentation purposes"""
        
        if not isinstance(intermediate_files, list) and isinstance(intermediate_files, IntermediateBeatmapFormat):
            intermediate_files = [intermediate_files]
        
        for offset in offsets:
            assert offset >= 0.0, "Offset must be non-negative!"
            results = []
            for intermediate_file in intermediate_files:
                assert "TIME" in intermediate_file.data.columns, "Missing 'TIME' column!"
                assert len(intermediate_file.data["TIME"].diff().unique()) <= 2, "Time information is not quantized!"
                assert intermediate_file.data["TIME"].diff().values[-1] == self.tokenizer.QUANTIZE, f"Time information is quantized differently from the tokenizer's settings: {self.tokenizer.QUANTIZE}"
    
                input_df = intermediate_file.data.copy()
                if offset > 0.0:
                    input_df = input_df.loc[input_df["TIME"] >= offset * 1000.]
                    input_df["TIME"] = input_df["TIME"] - int(offset * 1000)
                encoded_input_df = self.tokenizer.encode(input_df)
                empty_mask = self.tokenizer._group_empty(input_df)
                tempo_values = self.tokenizer._group_tempo(input_df)
                result = self._generate_training_data(encoded_input_df, 
                                                    difficulty=intermediate_file.meta["difficulty"],
                                                    tempo_values=tempo_values,
                                                    empty_mask=empty_mask,
                                                    max_empty_interval=max_empty_interval, 
                                                    ignore_lead_in=ignore_lead_in)
                results.append(result)

            self.df_data = pd.concat([self.df_data] + results, ignore_index=True)
            self.audio_data[self.audio_cnt] = self.tokenizer.audio_converter(audio_path, offset=offset)
            self.audio_cnt += 1

    def _generate_training_data(self,
                                encoded_input_df: pd.DataFrame,
                                difficulty: float = 0.5,  # 0. - 1.
                                tempo_values: Optional[np.ndarray] = None, 
                                empty_mask: Optional[np.ndarray] = None,
                                max_empty_interval: Optional[float] = 4.0,  # seconds
                                ignore_lead_in: bool = False) -> pd.DataFrame:
        """Generates input / output pairs as list of tokens based on settings"""
        use_tracks = [col for col in encoded_input_df.columns if col in self.tokenizer.config.tracks]
        cls = self.tokenizer.RESERVED_TOKENS[self.tokenizer._tokenize_difficulty(difficulty)]
        window = self.tokenizer.config.context_length // len(use_tracks) - 1
        assert self.tokenizer.config.audio_foresight % len(use_tracks) == 0, f"The value of 'audio_forsight' is not divisible by the number of tracks!"
        foresight = self.tokenizer.config.audio_foresight // len(use_tracks)
        empty_cnt, empty_first, empty_token = 0, ignore_lead_in, len(self.tokenizer.RESERVED_TOKENS)
        if max_empty_interval is not None:
            max_empty_interval = max_empty_interval * len(use_tracks) * 1000. / (self.tokenizer.QUANTIZE * self.tokenizer.config.groups)

        results = []
        for step in range(len(encoded_input_df)):
            if empty_mask is not None and empty_mask[step]:
                continue

            data = []
            for i, track in enumerate(use_tracks):
                data += [self.tokenizer.RESERVED_TOKENS["SEP"] if i else cls]
                data_chunk = encoded_input_df[track].values[:step + 1]
                if not len(data_chunk) or data_chunk[-1] == empty_token:
                    empty_cnt += 1
                else:
                    empty_cnt = 0
                    empty_first = False
                data_chunk = data_chunk[-window + foresight:].tolist()
                data += [self.tokenizer.RESERVED_TOKENS["PAD"] for _ in range(window - foresight - len(data_chunk))] + data_chunk + [self.tokenizer.RESERVED_TOKENS["FORESIGHT"] for _ in range(foresight)]

            if max_empty_interval is not None and empty_cnt > max_empty_interval:
                continue
            if empty_first and empty_cnt >= 0:
                continue

            results.append({
                "data": np.array(data).astype(np.uint8),
                "tempo": tempo_values[step] if tempo_values is not None else 0.0,
                "tracks": len(use_tracks),
                # get audio from the future, pad it from the left
                "audio_position": step + foresight + self.tokenizer.config.context_length,
                "audio_cnt": self.audio_cnt,
            }) 
        return pd.DataFrame(results)


    def __len__(self) -> int:
        return len(self.df_data)

    def __getitem__(self, idx) -> Dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df_data.loc[idx]
        output_data = row["data"].astype(np.int32)
        input_data = output_data.copy()

        if self.augment:
            # randomly mask multiple positions after the selected one if applicable
            mask_choices = self.generator.integers(0, len(self.masks[row["tracks"]]))
            for elem in self.masks[row["tracks"]][mask_choices:]:
                input_data[elem] = self.tokenizer.RESERVED_TOKENS["MASK"]
            # randomly dropout token positions and replace them with mask
            if self.tokenizer.config.dataset_dropout:
                token_mask = input_data >= len(self.tokenizer.RESERVED_TOKENS)
                random_mask = np.where(self.generator.uniform(size=len(input_data)) < self.tokenizer.config.dataset_dropout, True, False)
                input_data[np.logical_and(token_mask, random_mask)] = self.tokenizer.RESERVED_TOKENS["MASK"]
        else:
            # select one mask for eval
            mask_choices = idx % len(self.masks[row["tracks"]])
            for elem in self.masks[row["tracks"]][mask_choices:]:
                input_data[elem] = self.tokenizer.RESERVED_TOKENS["MASK"]

        audio = self.tokenizer._get_audio_chunk(self.audio_data[row["audio_cnt"]], 
                                                position=row["audio_position"],
                                                number_of_tracks=row["tracks"])
        return {
            "input_data": input_data,
            "segment_data": self.segments[row["tracks"]],
            "input_audio": audio,
            "output_data": output_data,
            "output_mask": self.masks[row["tracks"]][mask_choices],
            "tempo": max(0.0, row["tempo"]) if row["tempo"] < 400.0 else 0.0,  # a single outlier can kill training
        }
