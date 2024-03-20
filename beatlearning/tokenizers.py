import numpy as np
import pandas as pd
import librosa
from typing import List, Optional

from .configs import BEaRTConfig


class BEaRTTokenizer:

    """
        The tokenizer takes the intermediate beatmap output and turns it into LLM tokens.
        (QUANTIZE * groups) events will be grouped and converted into a single token,
        with each token capable of capturing up to 2 timed events within this interval
        (HOLD, HOLD and BEAT, BEAT and HOLD, BEAT, BEAT and BEAT) at QUANTIZE accuracy.
        A group = 10 is capable of capturing a CPS of 20, while group = 8 
        can capture events with a CPS of 25 (~ world record speed, but the length of
        how much of the beatmap the model sees at the time will be decreased).
    """

    RESERVED_TOKENS = {
        "DIFFICULTY_EASY": 0,
        "DIFFICULTY_NORMAL": 1,
        "DIFFICULTY_HARD": 2,
        "DIFFICULTY_INSANE": 3,
        "SEP": 4,
        "PAD": 5,
        "MASK": 6,
        "FORESIGHT": 7,
    }
    QUANTIZE = 10  # ms

    def __init__(self, config: BEaRTConfig):
        assert len(config.tracks) in config.track_combinations, "Unsupported number of tracks!"
        assert config.groups > 1, "Number of events grouped into a single token must be larger than 1!"
        assert config.mel_bands > 1, "Number of bands for the Mel Spectogram audio features must be larger than 1!"
        assert config.tempo_modifier >= 0.0, "Tempo modifier must be positive!"
        self.config = config

        # helper values
        self.distance = self.config.groups // 2
        self.audio_padding = [np.zeros((self.config.groups, self.config.mel_bands)).astype(np.float32) for _ in range(self.config.context_length)]
        self.arange = np.arange(self.config.groups)
        self._vocab_size = (self.config.groups + self.arange[1:(self.config.groups - self.distance) + 1].sum()) * 3 + len(self.RESERVED_TOKENS) + 2


    @property
    def vocab_size(self) -> int:
        """Returns the number of tokens required to represent beats based on tokenizer's settings"""
        return self._vocab_size  # pre-calculated
    
    def receptive_field(self, use_tracks: Optional[List[str]] = None) -> int:
        """Returns the maximum time interval of the beatmap the model can see"""
        track_len = len(self.config.tracks) if use_tracks is None else len(use_tracks)
        return int(np.ceil(((self.config.context_length - self.config.audio_foresight - track_len) / track_len)) * (self.config.groups * self.QUANTIZE))
    
    def _to_beat_token(self, values: np.ndarray) -> int:
        """Converts vector of QUANTIZEd beat events into a single token"""
        if isinstance(values, pd.Series):
            values = values.values
        elif isinstance(values, list):
            values = np.array(values)
        assert len(values.shape) == 1 and len(values) == self.config.groups, f"A NumPy vector with the length of {self.config.groups}x{self.QUANTIZE}ms events must be passed!"
        values = values.astype(int)
        assert np.all(np.isin(values, [0, 1, 2])), f"Only 0, 1 and 2 values are allowed! Got {values}"
        
        offset = 0
        if values.sum() == 0:
            # all 0s, no event
            t = 0
        elif values.sum() == 2 * self.config.groups:
            # all 2s, continued hold
            t = 1
        elif values[0] == 2:
            # ending continued hold + optional beat
            offset = 2 # 0, 1 already used
            first = self.arange[(values != 2)][0] - 1
            second = self.arange[(values == 1) & (self.arange >= first + self.distance)]
            if not len(second):
                # f"E{self.arange[first]}"
                t = self.arange[first]
            else:
                # f"E{self.arange[first]}B{self.arange[second]}"
                offset += self.config.groups
                second = second[0]
                t = self.arange[self.distance + 1 - first:self.distance + 1].sum() + self.arange[second] - (self.config.groups - self.distance) - self.arange[first]
        elif values[-1] == 2:
            # optional beat + starting hold (offset )                
            offset = 2 + self.config.groups + self.arange[1:(self.config.groups - self.distance) + 1].sum()
            second = self.arange[(values != 2)][-1] + 1
            first = self.arange[(values == 1) & (self.arange <= second - self.distance)]
            if not len(first):
                # f"S{self.arange[second]}"
                t = self.arange[second]
            else:
                # f"B{self.arange[first]}S{self.arange[second]}"
                offset += self.config.groups
                first = first[0]
                t = self.arange[self.distance + 1 - first:self.distance + 1].sum() + self.arange[second] - (self.config.groups - self.distance) - self.arange[first]
        else:
            # beat(s) only
            offset = 2 + self.config.groups * 2 + self.arange[1:(self.config.groups - self.distance) + 1].sum() * 2
            first = self.arange[(values == 1)]
            if len(first):
                first = first[0]
                second = self.arange[(values == 1) & (self.arange >= first + self.distance)]
                if not len(second):
                    #f"B{self.arange[first]}"
                    t = self.arange[first]
                else:
                    #f"B{self.arange[first]}B{self.arange[second]}"
                    offset += self.config.groups
                    second = second[0]
                    t = self.arange[self.distance + 1 - first:self.distance + 1].sum() + self.arange[second] - (self.config.groups - self.distance) - self.arange[first]
            else:
                # maybe it is a really short hold, turn it into a single beat instead of discarding
                if np.sum(values == 2) >= 2:
                    first = self.arange[(values == 2)][0]
                    t = self.arange[first]
                else:
                    # unknown / error
                    offset = 0
                    t = 0
           
        t += offset + len(self.RESERVED_TOKENS)
        assert t <= self.vocab_size, f"Unexpected error with token {t}"
        return t

    def _reverse_events(self, first:int, second: int, is_first_hold: bool, is_second_hold: bool) -> List[int]:
        """Helper function that creates a list of beat events based on the value of the first and second beat or hold events"""
        result = []
        for i in range(self.config.groups):
            if is_first_hold and i <= first:
                result.append(2)
            elif not is_first_hold and i == first:
                result.append(1)
            elif is_second_hold and i >= second:
                result.append(2)
            elif not is_second_hold and i == second:
                result.append(1)
            else:
                result.append(0)
        return result

    def _from_beat_token(self, token: int) -> List[str]:
        """Converts a single non-reserved beat token into a vector of QUANTIZEd beat events"""
        assert token < self.vocab_size, f"Received out of vocabulary token {token}!"
        t = token
        if t < len(self.RESERVED_TOKENS):
            raise ValueError("Reserved tokens can not be decoded into beat events!")
        t -= len(self.RESERVED_TOKENS)
        if t <= 1:
            return [0 if t == 0 else 2] * self.config.groups
        t -= 2
        
        if t < self.config.groups:
            # ending continued hold
            return self._reverse_events(t, -1, is_first_hold=True, is_second_hold=False)
        t -= self.config.groups
        if t < self.arange[1:(self.config.groups - self.distance) + 1].sum():
            #ending continued hold + beat
            for first in range(self.config.groups - self.distance):
                for second in range(first + self.distance, self.config.groups):
                    if t == self.arange[self.distance + 1 - first:self.distance + 1].sum() + self.arange[second] - (self.config.groups - self.distance) - self.arange[first]:
                        return self._reverse_events(first, second, is_first_hold=True, is_second_hold=False) 
        t -= self.arange[1:(self.config.groups - self.distance) + 1].sum()
    
        if t < self.config.groups:
            # hold started
            return self._reverse_events(-1, t, is_first_hold=False, is_second_hold=True)
        t -= self.config.groups
        if t < self.arange[1:(self.config.groups - self.distance) + 1].sum():
            # hold started + beat
            for first in range(self.config.groups - self.distance):
                for second in range(first + self.distance, self.config.groups):
                    if t == self.arange[self.distance + 1 - first:self.distance + 1].sum() + self.arange[second] - (self.config.groups - self.distance) - self.arange[first]:
                        return self._reverse_events(first, second, is_first_hold:=False, is_second_hold=True) 
        t -= self.arange[1:(self.config.groups - self.distance) + 1].sum()
    
        if t < self.config.groups:
            # single beat
            return self._reverse_events(t, -1, is_first_hold=False, is_second_hold=False)
        t -= self.config.groups
        if t < self.arange[1:(self.config.groups - self.distance) + 1].sum():
            # beat + beat
            for first in range(self.config.groups - self.distance):
                for second in range(first + self.distance, self.config.groups):
                    if t == self.arange[self.distance + 1 - first:self.distance + 1].sum() + self.arange[second] - (self.config.groups - self.distance) - self.arange[first]:
                        return self._reverse_events(first, second, is_first_hold=False, is_second_hold=False) 
        # unknown
        return [0] * self.config.groups
        
    def encode(self, 
               input_df: pd.DataFrame,
               *,
               use_tracks: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert QUANTIZEd beat events into beat tokens for each track (does not add special tokens or convert non-track columns)"""
        if use_tracks is None:
            use_tracks = [col for col in input_df.columns if col in self.config.tracks]
        assert len(use_tracks) in (1, 2, 4), "Unsupported number of tracks!"
        results = {col: [] for col in use_tracks}
        for part_df in [input_df[i:i + self.config.groups] for i in range(0,  len(input_df), self.config.groups)]:
            for col in use_tracks:
                values = np.pad(part_df[col].values, (0, self.config.groups - len(part_df[col].values)), 'constant', constant_values=0)
                results[col].append(self._to_beat_token(values))
        return pd.DataFrame(results)

    def decode(self, 
               output_df: pd.DataFrame,
               *,
               offset: float = 0.0,
               use_tracks: Optional[List[str]] = None) -> pd.DataFrame:
        """Decodes DataFrame of tokens into a DataFrame of beat events for the intermediate representation"""
        if not isinstance(output_df, pd.DataFrame):
            output_df = pd.DataFrame(output_df)
        if use_tracks is None:
            use_tracks = [col for col in output_df.columns if col in self.config.tracks]
        results = {col: [] for col in use_tracks + ["TEMPO"]}
        for _, row in output_df.iterrows():
            for col in use_tracks:
                results[col] += self._from_beat_token(row[col])
            results["TEMPO"] += [row["TEMPO"] if "TEMPO" in row else 0.0] * self.config.groups
        results = pd.DataFrame(results)
        results["TIME"] = np.arange(len(results)) * self.QUANTIZE + int(offset * 1000)
        return results

    def audio_converter(self, 
                        audio_file: str, 
                        *,
                        offset: float=0.0,  # for augmentation purposes only
                        ) -> List[np.ndarray]:
        """Returns a list of padded [(groups, mel_bands)] arrays containing parts of the Mel Spectogram based on the audio file"""
        assert offset >= 0.0, "Offset value can not be negative!"
        mono_audio_data, sr = librosa.load(audio_file, sr=44100, offset=offset, mono=True)
        hop_length = int(sr / 1000 * self.QUANTIZE)
        S = librosa.feature.melspectrogram(y=mono_audio_data, sr=sr, n_mels=self.config.mel_bands, hop_length=hop_length, fmax=sr // 2)
        S_dB = 1. + (librosa.power_to_db(S, ref=np.max).T / 80.0)
        if S_dB.shape[0] % self.config.groups:
            S_dB = np.vstack((S_dB, np.zeros((self.config.groups - S_dB.shape[0] % self.config.groups, self.config.mel_bands))))
        if self.config.mel_gradient:
            S_dB = np.gradient(S_dB, axis=1)
        S_dB = np.clip(S_dB, -1.0, 1.0)
        results = []
        for i in range(0, S_dB.shape[0], self.config.groups):
            results.append(S_dB[i:i + self.config.groups, :].astype(np.float32))
        # add padding on both sides in case there are beatmap definitions without music
        return self.audio_padding + results + self.audio_padding
    
    def _tokenize_difficulty(self, difficulty: float) -> int:
        """Convert difficulty value from 0. - 1. into a single CLS token"""
        difficulty_tokens = {value: key for key, value in self.RESERVED_TOKENS.items() if key.lower().startswith("difficulty")}
        cls = min(len(difficulty_tokens) - 1, int(len(difficulty_tokens) * difficulty))
        return difficulty_tokens[cls]
    
    def _group_empty(self, input_df: pd.DataFrame) -> np.ndarray:
        """Group empty time intervals based on tokenizer's settings so data generation can skip these areas"""
        if "empty" in input_df.columns:
            results = []
            for i in range(0, len(input_df), self.config.groups):
                check = input_df["empty"].values[i:i + self.config.groups]
                results.append(np.sum(check) >= len(check))
            return np.array(results)
        else:
            return None
        
    def _group_tempo(self, input_df: pd.DataFrame) -> np.ndarray:
        """Create a vector of BPMs, where the last value can be used as an additional regression task for the input / output pairs"""
        if "TEMPO" in input_df.columns:
            results = []
            for i in range(0, len(input_df), self.config.groups):
                check = [val for val in input_df["TEMPO"].values[i:i + self.config.groups] if val > 0.0]
                if len(check):
                    results.append(check[-1])
                else:
                    results.append(0.0)
            return np.array(results)
        else:
            return None
    
    def _generate_pad(self, col: List[int], window: int, mask: bool=False) -> List[int]:
        result = [self.RESERVED_TOKENS["PAD"] for _ in range(max(0, window - len(col)))] + col[-window:]
        if mask:
            return result[1:] + [self.RESERVED_TOKENS["MASK"]]
        else:
            return result
        
    def _generate_mask(self, number_of_tracks: int)  -> List[int]:
        track_length = self.config.context_length // number_of_tracks
        return np.array([track_length - self.config.audio_foresight - 1 for _ in range(number_of_tracks)]).astype(np.int32)

    def _generate_segment(self, number_of_tracks: int)  -> List[int]:
        segment = []
        for track_token in range(number_of_tracks):
            segment += [track_token for _ in range(self.config.context_length // number_of_tracks)]
        return np.array(segment).astype(np.int32)
    
    def _get_audio_chunk(self, 
                         audio_features: List[np.ndarray],  
                         position: int,
                         number_of_tracks: int) -> List[np.ndarray]:
        window = self.config.context_length // number_of_tracks - 1
        audio_slice = np.vstack(audio_features[max(0, position-window):position+1]).astype(np.float32)
        if number_of_tracks > 1:
            audio_slice = np.tile(audio_slice, (number_of_tracks, 1))
        return audio_slice.ravel()
    
        