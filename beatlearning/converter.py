import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from zipfile import ZipFile
import tempfile
from copy import deepcopy
import re

import torch, torchaudio
from transformers import EncodecModel, AutoProcessor
from transformers.utils import logging as hf_logging 

from typing import List, Tuple, Any, Optional, Union, Dict
import logging
#logging.basicConfig(format='%(asctime)s %(levelname)s > %(message)s', datefmt='%H:%M:%S', level=logging.WARNING)

from .utils import tokenize


class BeatConverter:
    """Converts beatmaps to features"""
    
    FILE_TYPES = []  # list of allowed file types without comma
    TRACKS = ["a"]  # name of input / output tracks
    ENCODEC = "facebook/encodec_24khz"  # Meta's 24khz model https://huggingface.co/facebook/encodec_24khz
    
    def __init__(self,
                 input_folder: Optional[str] = None,
                 output_folder: Optional[str] = None,
                 *,
                 audio_bins: int = 12,  # number of bins for each audio_chunks_length time interval (will provide x2 outputs per track)
                 output_tokens_n: int = 32,  # block size of output tokens for beat representations
                 output_token_reduction: int = 2,  # reduces token vocabulary, by combining outputs (12 bins by 2 --> 6)  
                 audio_chunks_length: int = 1000,  # length of encodec chunks in ms
                 **kwargs) -> None:
        self._iter_index = 0
        assert np.all([check not in ("time", "tempo", "meter", "silence") for check in self.TRACKS]), "Tracks can not have the same name as reserved columns!"
        
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.input_folder_list = self._get_files(self.input_folder, self.FILE_TYPES, recurring=True) if (self.FILE_TYPES and self.input_folder is not None) else []
        self.input_folder_list = sorted(self.input_folder_list)
        
        self.audio_bins = max(1, int(audio_bins))
        self.output_tokens_n = max(1, int(output_tokens_n))
        self.output_token_reduction = max(1, int(output_token_reduction))
        assert self.output_token_reduction <= self.audio_bins and not self.audio_bins % self.output_token_reduction, "Token reduction must be proportional to the number of bins!"
        
        self.audio_chunks_length = float(audio_chunks_length)
        
        # load HF models while supressing warnings
        hf_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity(40)
        self.encodec_model = EncodecModel.from_pretrained(self.ENCODEC)
        self.encodec_processor = AutoProcessor.from_pretrained(self.ENCODEC)
        hf_logging.set_verbosity(hf_verbosity)
        
        self._min_bpm = int(np.ceil((1000. / (self.audio_chunks_length / 2.)) * 60))
        self._max_bpm = int(np.floor((1000. / (self.audio_chunks_length / self.audio_bins)) * 60))
        info = f"Found {len(self)} matching files in '{self.input_folder}'\n" if self.input_folder is not None else ""
        logging.info(f"{info}BPM values allowed: {self._min_bpm} - {self._max_bpm}\n"
                     f"Maximum window length for beats: {self.output_tokens_n * self.audio_chunks_length / 1000.}s\n")
    
        
    @property
    def vocab_size(self) -> int:
        return 2**int(self.audio_bins // self.output_token_reduction) + 1
    
    def __len__(self) -> int:
        return len(self.input_folder_list)
    
    def __iter__(self):
        self._iter_index = 0
        return self
    
    def __next__(self) -> Any:
        """Yields the extracted list of features for each matching file"""
        try:
            result = self.input_folder_list[self._iter_index]
        except IndexError:
            raise StopIteration
        self._iter_index += 1
        return result
    
    def _get_files(self, subfolder: str, types: Union[str, List[str]], recurring: bool = False) -> List[str]:
        files = []
        if not isinstance(types, (list, tuple)):
            types = [types]
        for file in os.listdir(subfolder):
            file = os.path.join(subfolder, file)
            if os.path.isfile(file):
                for file_type in types:
                    if file.lower().endswith(f".{file_type.lower()}"):
                        files.append(file)
            elif recurring:
                files += self._get_files(file, types, recurring=True)
        return files    
    
    def encode(self, 
               file: str,
               audio_offset: int = 0  # for data augmentation, in ms
               ) -> np.ndarray:
        """Returns list of token encoded versions of consecutive audio chunks"""
        audio, sr = torchaudio.load(file)
        sr = int(sr * self.audio_chunks_length / 1000.)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, axis=0).reshape(1, -1)
        audio = audio.squeeze(0)
        audio_offset = int(audio_offset / 1000. * sr)
        length = int(np.ceil((len(audio) - audio_offset) / sr))
        results_1, results_2 = [], []
        for i in tqdm(range(length)):
            chunk = audio[max(0, int(i * sr + audio_offset)):int((i + 1) * sr + audio_offset)]
            if len(chunk) < sr:
                break
            inputs = self.encodec_processor(raw_audio=chunk, sampling_rate=self.encodec_processor.sampling_rate, return_tensors="pt")
            audio_codes = self.encodec_model(inputs["input_values"], inputs["padding_mask"]).audio_codes
            results_1.append([int(val) for val in audio_codes[0][0][0].numpy()])  # first quantized channel
            results_2.append([int(val) for val in audio_codes[0][0][1].numpy()])  # residuals 
        return results_1, results_2
        
    def convert(self, 
                data: pd.DataFrame, 
                audio: np.ndarray, 
                meta: dict = {},
                audio_offset: int = 0,  # NOTE: both data and audio should have the same offset value
                deviation: float = 0.05,  # offset fix deviation %
                ) -> Optional[pd.DataFrame]:
        """Convert cached intermediary song information into training data"""
        for col in ("time", "tempo", "meter"):
            if col not in data.columns:
                logging.error(f"Can not convert data: missing column '{col}' from beatmap DataFrame.")
                return None
        
        data = data.sort_values(by=["time"], ascending=True)
        data.set_index("time", drop=False, inplace=True)
          
        bin_length = self.audio_chunks_length / self.audio_bins
        if data.loc[data["tempo"] != 0]["tempo"].min() < bin_length:
            logging.error(f"Can not convert data: some beat lengths are too short ({data.loc[data['tempo'] != 0]['tempo'].min()} < {bin_length}), therefor can not be represented by the output. Try increasing the number of audio_bins ({self.audio_bins}).")
            return None
        
        # add holds as events
        fix = []
        check = self.TRACKS + ["silence"]
        last_row = {}
        for index, row in data.iterrows():
            if last_row:
                if np.any([last_row[f"{col}e"] > 0 for col in check]):
                    if last_row["tempo"]:
                        while last_row["time"] + last_row["tempo"] + 1 < row["time"]:
                            last_row["time"] += last_row["tempo"]
                            fix.append(last_row)
            else:
                last_row = {key: value for key, value in row.items()}
            last_row = {f"{col}e": max(last_row[f"{col}e"], row[f"{col}e"]) for col in check}
            last_row = {f"{col}e": 0 if last_row[f"{col}e"] < row["time"] else last_row[f"{col}e"] for col in check}
            last_row.update({key: value for key, value in row.items() if key not in last_row})    
            fix.append({key: value for key, value in last_row.items()})
        if fix:
            ending = np.max([fix[-1][f"{col}e"] for col in check])
            if last_row["tempo"]:
                while last_row["time"] + 1 < ending:
                    last_row["time"] += last_row["tempo"]
                    last_row.update({f"{col}e": 0 if last_row[f"{col}e"] < last_row["time"] else last_row[f"{col}e"] for col in check})
                    fix.append({key: value for key, value in last_row.items()})
        
        if not len(fix):
            return None
        fix = pd.DataFrame(fix)
        for col in check:
            fix[f"{col}h"] = np.where(fix[f"{col}e"] > 0, True, False)
        fix.drop(columns=[f"{col}e" for col in check] + ["silenceh"], inplace=True)
        fix.rename(columns={"silenced": "silence"}, inplace=True)
        fix.set_index("time", drop=False, inplace=True)
        
        # check for changes in beat timing
        fix["diff"] = fix["time"].diff()
        check_fix = fix.loc[fix["diff"] > 0]
        if len(check_fix.loc[check_fix["diff"] < check_fix["tempo"].min() / 2. * .95]) > 2:
            if len(check_fix.loc[check_fix["diff"] < bin_length]) > 2:
                logging.warning(f"Some beats can not be represented as they are shorter than the bin length settings ({check_fix['diff'].min()} < {bin_length})")
            else:
                logging.warning(f"The beatmap is actually faster than twice its BPM settings ({check_fix['diff'].min()} < {check_fix['tempo'].min()})")
        if check_fix["tempo"].max() > self.audio_chunks_length:
            logging.warning(f"Some tempo definitions are longer than the length of audio chunks ({check_fix['diff'].max()} > {self.audio_chunks_length})")
        
        for info in ("tempo", "meter", "diff"):
            fix[f"{info}_change"] = fix[info].shift(1, fill_value=fix[info].values[0]) != fix[info]
            
        # generate chunks from boolean beatmap representation
        results = []
        tempo, offset, weight = 0., 0., 0
        proposed_offset = np.nan  # use estimated offset if possible instead of very first beat
        output_fifo = {}
        last_event = fix["time"].values[-1] + (fix["tempo"].values[-1] / 1000) * 12
        for position, (audio_tokens_1, audio_tokens_2) in enumerate(zip(audio[0], audio[1])):
            output = {}
            relative = position * self.audio_chunks_length
            select = fix.loc[relative:relative + self.audio_chunks_length - 1.]
            
            # is it just silence / end of song?
            if relative > last_event:
                #logging.warning("Audio is longer than the beatmap definition, truncating training data.")
                break
            
            # check for beat declarations
            if len(select):
                tempo = select["tempo"].values[0]
                offset = select["time"].values[0] - relative
                if offset + tempo >= self.audio_chunks_length * 2:
                    output_fifo = {}
                    proposed_offset = np.nan
                    weight = 0  # workaround to trigger weight = 1 next time
                    continue
                elif tempo > 0.:
                    while offset >= tempo:  # bin_length
                        offset -= tempo
                    if offset < 0.:
                        offset += tempo
                    # see if offset isn't for a half-beat
                    found = False
                    if proposed_offset is not None:
                        div_min, div_max = proposed_offset * (1. - deviation), proposed_offset * (1. + deviation)
                        if offset < div_min or offset > div_max:
                            # offset and proposed offset are different
                            # use proposed offset if there is a beat at that time
                            for check in select["time"].values:
                                check = check - relative
                                while check > div_min:
                                    if check >= div_min and check <= div_max:
                                        found = True
                                        break
                                    check -= tempo
                                if found:
                                    break
                    if found:
                        offset, proposed_offset = proposed_offset, offset
                    else:
                        proposed_offset = offset
                    # guess next offset for full beat
                    while proposed_offset <= self.audio_chunks_length:
                        proposed_offset += tempo
                    while proposed_offset >= self.audio_chunks_length:
                        proposed_offset -= self.audio_chunks_length    
                     
                if len(select) <= select["silence"].sum():
                    # entire chunk is silence, skip
                    output_fifo = {}
                    proposed_offset = np.nan
                    weight = 0  # workaround to trigger weight = 1 next time
                    continue
                
                for track, prefix in zip(("hits", "holds"), ("d", "h")):
                    output[track] = {col: [] for col in self.TRACKS}
                    
                for part in range(self.audio_bins):
                    select_bin = select.loc[relative + part * bin_length: relative + (part + 1) * bin_length - 1.]    
                    for track, prefix in zip(("hits", "holds"), ("d", "h")):
                        for col in self.TRACKS:
                            output[track][col].append((1 if select_bin[f"{col}{prefix}"].sum() > 0 else 0) if len(select_bin) else 0)
                
                if select["tempo_change"].sum() or select["diff"].min() < bin_length or weight == 0:  # change in tempo or start of new chunk
                    weight = 1
                else:
                    if tempo > 0.:
                        weight = max(1, np.sum(np.arange(int(np.ceil(self.audio_chunks_length / tempo)) - 1) + 1))
                    else:
                        weight = 1
            else:
                for track in ("hits", "holds"):
                    output[track] = {col: [0 for _ in range(self.audio_bins)] for col in self.TRACKS}
                if weight == 0:  # in case the empty part is at the start of a new chunk
                    weight = 1
            
            if tempo > 0. and offset + tempo < self.audio_chunks_length * 2:
                result = {
                    "audio_tokens_1": audio_tokens_1,
                    "audio_tokens_2": audio_tokens_2,
                    
                    "tempo": tempo / 1000.,  # ms to s
                    "offset": offset / 1000.,  # relative, ms to s
                    "timestamp": relative + audio_offset,  # for debugging only
                    "time": min(1., position / max(1, (len(audio[0]) - 1))),  # relative time based on audio length 0. - 1.
                    "weight": weight,
                }
                for track in ("hits", "holds"):
                    if track not in output_fifo:
                        output_fifo[track] = {}
                    for col in self.TRACKS:
                        result[f"{track}_{col}"] = [elem for elem in output[track][col]]
                        output_token = tokenize(output[track][col], self.output_token_reduction) + 1  # 0 is empty padding so add +1 to all possible values
                        result[f"{track}_{col}_token_output"] = output_token
                        if col not in output_fifo[track]:
                            output_fifo[track][col] = [0 for _ in range(self.output_tokens_n)]  # start with silence padding
                        output_fifo[track][col].append(output_token)
                        if len(output_fifo[track][col]) > self.output_tokens_n + 1:  # + 1 because ending is removed
                            output_fifo[track][col].pop(0)
                        result[f"{track}_{col}_token_inputs"] = [elem for elem in output_fifo[track][col][:-1]]  # do not include output in input
                        
                results.append(result)
        if not len(results):
            return None
        
        results = pd.DataFrame(results)
        for key, value in meta.items():
            if key in results.columns:
                logging.warning(f"Meta column {key} was already declared in converted DataFrame results.")
            results[key] = value
        
        return results
    
    def extract(self, file: str) -> dict:
        """Generate intermediary cached data out of song files"""
        raise NotImplementedError


class OsuBeatConverter(BeatConverter):
    
    FILE_TYPES = ["osz"]  # .osz is a zip packed OSU file
    TRACKS = ["l", "d", "u", "r", "a"]  # left, down, up, right, any (merged)
    
    def __init__(self, *args, **kwargs):
         super().__init__(*args, **kwargs)
    
    def osu_parser(self,
                   file:str,
                   audio_offset: int = 0  # for data augmentation, in ms
                   ) -> Tuple[Optional[dict], Optional[pd.DataFrame]]:
        
        with open(file, "r", encoding="utf-8") as f:
            contents = f.read()
        
        # convert file contents to dict
        beatmap = OsuBeatConverter._osu_to_dict(contents)
        for check in ("General", "Difficulty", "Metadata", "TimingPoints", "HitObjects"):
            if check not in beatmap or not len(beatmap[check]):
                logging.error(f"{check} settings were not found in '{file}'")
                return None, None
        
        # get useful metadata from beatmap dict
        meta = {
            "audio": beatmap["General"]["AudioFilename"] if "AudioFilename" in beatmap["General"] else "audio.mp3",
            "preview": int(beatmap["General"]["PreviewTime"]) if "PreviewTime" in beatmap["General"] else 0,
            "mode": int(beatmap["General"]["Mode"]) if "Mode" in beatmap["General"] else 0,
            "sample": beatmap["General"]["SampleSet"] if "SampleSet" in beatmap["General"] else "Normal",
            "countdown": int(beatmap["General"]["Countdown"]) if "Countdown" in beatmap["General"] else 0,
            "level": beatmap["Difficulty"]["OverallDifficulty"],
            
            # other OSU related for generating statistics
            "difficulty": deepcopy(beatmap["Difficulty"]),
            "info": deepcopy(beatmap["Metadata"]),
        }
        
        # generate important events from multiple sources
        events = {}
        directions = [col for col in self.TRACKS if col != "a"]
        
        # get timings
        for tp in beatmap["TimingPoints"]:
            try:
                # time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
                parts = [elem.strip() for elem in tp.split(",")]
                time = int(float(parts[0]))
                if time not in events:
                    events[time] = []
                events[time].append({
                    "type": "timing",
                    "tempo": float(parts[1]),
                    "meter": int(parts[2]),
                    "uninherited": parts[-2] == "1", # uninherited is True (beatLength instead of velocity)
                })
            except Exception as e:
                logging.error(f"Timing events in '{file}' are malformed: {e}, {parts}")
                return None, None
        
        # get silent intervals
        if "AudioLeadIn" in beatmap["General"]:
            leadin = int(beatmap["General"]["AudioLeadIn"])
            if leadin > 0:
                if 0 not in events:
                    events[0] = []
                events[0].append({
                    "type": "silence",
                    "end": leadin,
                })
        for ep in beatmap["Events"]:
            try:
                parts = [elem.strip() for elem in ep.split(",")]
                if parts[0] in (2, "2", "Break", "break"):
                    time = int(float(parts[1]))
                    if time not in events:
                        events[time] = []
                    events[time].append({
                        "type": "silence",
                        "end": int(float(parts[2])),
                    })
            except Exception as e:
                logging.warning(f"Silence events in '{file}' are malformed: {e}")
        
        # add hit events
        for ho in beatmap["HitObjects"]:
            parts = [elem.strip() for elem in ho.split(",")]
            time = int(float(parts[2]))
            if time not in events:
                events[time] = []
                
            object_type = int(parts[3])
            if object_type & 1 == 1:
                object_name = "circle"
            elif object_type & 2 == 2:
                object_name = "slider"
            elif object_type & 8 == 8:
                object_name = "spinner"
            elif object_type & 128 == 128:
                object_name = "hold"
            else:
                logging.warning(f"Unknown object type in '{file}': {object_type}")
                object_name = "circle"
                
            # circle-0 : x,y,time,type,hitSound,objectParams,hitSample
            # sliders-1: x,y,time,type,hitSound,curveType|curvePoints,slides,length,edgeSounds,edgeSets,hitSample
            # spinner-3: x,y,time,type,hitSound,endTime,hitSample
            # hold-7:    x,y,time,type,hitSound,endTime:hitSample
            hit = {
                "x": int(parts[0]),
                "y": int(parts[1]),
                "type": object_name,
                "repeat": int(parts[6]) if object_name == "slider" else 0,
                "length": float(parts[7]) if object_name == "slider" else 0,
                "end": int(float(parts[5].split(":")[0])) if object_name in ("spinner", "hold") else 0,
            }
            pos = min(3, max(0, int(hit["x"] * 4. // 512)))
            hit["pos"] = directions[pos]
            events[time].append(hit)
        
        hit_df = []
        tempo, meter, velocity = 0., 0, 1.
        for time in sorted(events.keys()):
            row = {"time": time - audio_offset, "tempo": tempo, "meter": meter}
            for d in directions + ["silence"]:
                row[f"{d}d"] = False
                row[f"{d}e"] = 0
            for elem in events[time]:
                if elem["type"] == "silence":
                    row["silenced"] = True
                    row["silencee"] = elem["end"] - audio_offset
                elif elem["type"] == "timing":
                    meter = elem["meter"]
                    row["meter"] = elem["meter"]
                    if elem["uninherited"]:
                        tempo = elem["tempo"]
                        row["tempo"] = elem["tempo"]
                        velocity = 1.
                    else:
                        velocity = np.abs(100. / elem["tempo"]) if elem["tempo"] else 1.
                else:
                    row[f"{elem['pos']}d"] = True
                    if elem["type"] in ("circle", "spinner", "hold"):
                        row[f"{elem['pos']}e"] = elem["end"] - audio_offset
                    elif elem["type"] == "slider":
                        # oh boy
                        px_per_beat = meta["difficulty"]["SliderMultiplier"] * 100. * velocity
                        beats_number    = (elem["length"] * elem["repeat"]) / px_per_beat
                        duration = np.ceil(beats_number * tempo)
                        elem["end"]  = time + duration - audio_offset
                    else:
                        logging.warning(f"Unknown parsed object name in '{elem['type']}' at {time}")
                        continue
            hit_df.append(row)     
        
        hit_df = pd.DataFrame(hit_df)
        hit_df["ad"] = hit_df[[f"{col}d" for col in directions]].sum(axis=1).astype(bool)
        hit_df["ae"] = hit_df[[f"{col}e" for col in directions]].max(axis=1)
        
        return meta, hit_df
    
    @staticmethod
    def _osu_to_dict(contents: str) -> dict:
        # return ini-like osu file contents as a single dict
        
        def cast(v: str) -> Union[str, int, float]:
            v = v.strip()
            if v.isnumeric():
                return int(v)
            try:
                v = float(v)
            except ValueError:
                pass
            return v
        
        beatmap = {}
        key = None
        for i, line in enumerate(contents.splitlines()):
            line = line.strip()
            if not line:
                continue
            if key is None and "Version" not in beatmap:
                version = line.split("format")[-1].strip()
                try:
                    version = int(version[1:]) if version else -1
                    if version > 14:
                        logging.warning(f"The beatmap version is v{version}, the latest supported version is v14! This could lead to parsing errors.")
                    elif version <= 0:
                        logging.warning(f"The beatmap version is unknown! This could lead to parsing errors.")
                except ValueError:
                    logging.warning(f"The beatmap version was not found! This could lead to parsing errors.")
                    version = -1 
                beatmap["Version"] = version
            else:
                header = re.findall(r'^\[(\w+)\]\s*$', line)
                if header:
                    key = header[0]
                elif key is not None:
                    if key in ("Events", "TimingPoints", "HitObjects"):
                        if key not in beatmap:
                            beatmap[key] = []
                        beatmap[key].append(line)  # parsed by other function
                    else:
                        if line.startswith("//"):
                            continue
                        if key not in beatmap:
                            beatmap[key] = {}
                        try:
                            k, v = line.split(":")
                        except ValueError:
                            k, v = line.strip(), None
                            if k in beatmap[key]:
                                continue
                        beatmap[key][k] = cast(v)
                else:
                    logging.warning(f"The beatmap definition is malformed: {line}")          
        return beatmap
        
    
    def extract(self, 
                file:str,
                audio_offset: int = 0  # for data augmentation, in ms
                ) -> Dict[str, Dict]:
        # unzip .osz file and save its meta data as json, beatmap as DataFarme and audio (unaffected by settings)
        tmpfile = ZipFile(file)
        logging.info(f"Extracting beatmaps from '{file}'")
        with tempfile.TemporaryDirectory() as tempdir:
            tmpfile.extractall(tempdir)
            
            # look for all osu difficulties (beatmap files) and modes
            beatmaps = {}
            audio = set()
            for beatmap in self._get_files(tempdir, "osu"):
                readable = os.path.basename(beatmap)
                meta, hit_df = self.osu_parser(beatmap, audio_offset)
                if meta is None:
                    logging.error(f"Failed to parse OSU beatmap for '{readable}'")
                    continue
                elif hit_df is None or not len(hit_df):
                    logging.error(f"Could not generate HitObject DataFrame for '{readable}'")
                    continue
                beatmaps[readable] = {
                    "meta": deepcopy(meta),
                    "data": hit_df.copy(),
                }
                mode = ("osu", "taiko", "catch", "mania")[beatmaps[readable]["meta"]["mode"]]    
                audio.add(beatmaps[readable]["meta"]["audio"])
                logging.info(f"Parsed '{readable}' for {len(beatmaps[readable]['data'])} {mode} events.")    
        
            # encode all audio files
            audio_tokens = {}
            if not audio:
                logging.error(f"Error! No audio files were assigned in '{file}'")
            else:
                for i, song in enumerate(audio):
                    logging.info(f"Encoding '{song}' ({i+1} / {len(audio)}) with offset {audio_offset}ms. This may take a while...")
                    c1, c2 = self.encode(os.path.join(tempdir, song), audio_offset)  
                    audio_tokens[song] = [c1, c2]
            
        if self.output_folder is not None:
            path = os.path.join(self.output_folder, os.path.splitext(os.path.basename(file))[0])
            logging.info(f"Saving extracted results to '{path}'")
            os.makedirs(path, exist_ok=True)
            for readable in beatmaps:
                output = os.path.join(path, os.path.splitext(readable)[0]) 
                with open(f"{output}.json", "w", encoding="utf-8") as f:
                    json.dump(beatmaps[readable]["meta"], f, ensure_ascii=False)
                beatmaps[readable]["data"].to_parquet(f"{output}.pq", index=False)
            
            for song in audio_tokens:
                output = os.path.join(path, os.path.splitext(song)[0]) 
                with open(f"{output}.enc", "w", encoding="utf-8") as f:
                    json.dump(audio_tokens[song], f, ensure_ascii=False)
        
        return {
            "augmentation": {"audio_offset": audio_offset},
            "beatmaps": beatmaps, 
            "audio": audio_tokens
        }

