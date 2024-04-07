import os
import re
import tempfile
from copy import deepcopy
from shutil import copyfile
from zipfile import ZipFile
import logging

import numpy as np
import pandas as pd

from typing import Any, List, Tuple, Union, Optional

from .utils import IntermediateBeatmapFormat, RobotsDisallowException


class BeatmapConverter:

    QUANTIZE = 10  # ms
    FILE_TYPES = []  # file types to parse from within the beatmap file
    TRACKS = []

    def __init__(self, *, random_seed: int = 69420):
        self.random_seed = random_seed
        self.seed()

    def convert(self, input_file: str, output_folder: str, audio_path: Optional[str]) -> None:
        os.makedirs(output_folder, exist_ok=True)
        zip_file = ZipFile(input_file)
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_file.extractall(temp_dir)
            for file in self._get_relevant_files(temp_dir):
                filename = os.path.basename(file)
                if filename.lower().endswith("mp3"):
                    copyfile(file, os.path.join(output_folder, filename))
                else:
                    try:
                        with open(file, "r", encoding="utf-8") as f:
                            contents = f.read()
                        meta, data = self.parse(contents)
                        ibf = IntermediateBeatmapFormat(meta=meta, data=data)
                        ibf.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.ibf"))
                    except Exception as e:
                        logging.error(f"Failed to parse {filename}: {e}")

    def generate(self, input_file: str, output_file: str, meta: dict = {}) -> None:
        raise NotImplementedError

    def snap(self, time: Any, is_second=False) -> int:
        time = float(time)
        if is_second:
            time *= 1000.
        return int(time // self.QUANTIZE * self.QUANTIZE)
    
    def seed(self) -> None:
        np.random.seed(self.random_seed)

    def _get_relevant_files(self, folder: str, recursive: bool = False) -> List[str]:
        files = []
        for file in os.listdir(folder):
            file = os.path.join(folder, file)
            if os.path.isfile(file):
                for file_type in self.FILE_TYPES:
                    if file.lower().endswith(f".{file_type.lower()}"):
                        files.append(file)
                        break
            elif recursive:
                files += self._get_relevant_files(file, recursive=True)
        return files


class OsuBeatmapConverter(BeatmapConverter):

    FILE_TYPES = ["osu", "mp3"]  # relevant file types in beatmap file
    TRACKS = ["LEFT", "UP", "DOWN", "RIGHT"]
    BEATMAP_DEFAULTS = {
        "osu_file": "beatmap.osu",
        "audio": "audio.mp3", 
        "lead_in": 0, 
        "mode": 0,
        "title": "Untitled",
        "artist": "Unknown Artist",
        "source": "",
        "tags": "gnerative ai generated beatlearning",
        "difficulty_name": "medium",
        "hp_drain_rate": 5,
        "overall_difficulty": 4,
        "approach_rate": 5, 
        "bg": None,
    }

    def __init__(self, 
                 *,
                 spinner_to_hold: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.spinner_to_hold = spinner_to_hold

    @staticmethod
    def _osu_to_dict(contents: str) -> dict:
        # parse the ini-like osu file contents as a single dict
        
        def cast(v: str) -> Union[str, int, float]:
            if v is None:
                return None
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
                except ValueError:
                    version = -1 
                if version > 14:
                    logging.warning(f"The beatmap version is v{version}, the latest supported version is v14! This could lead to parsing errors.")
                elif version <= 0:
                    logging.warning(f"The beatmap version is unknown! This could lead to parsing errors.")
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
    
    def parse(self, contents: str) -> dict:
        beatmap = self._osu_to_dict(contents)
        for check in ("General", "Difficulty", "Metadata", "TimingPoints", "HitObjects"):
            if check not in beatmap or not len(beatmap[check]):
                raise ValueError("OSU beatmap file is missing important meta information!")
        
        # do not parse the beatmap if the creator prohibits it
        if "robots" in beatmap["Metadata"] and beatmap["Metadata"]["robots"].lower().strip() == "disallow":
            raise RobotsDisallowException("The creator has denied access to this beatmap!")

        # get useful metadata from beatmap dict
        meta = {
            "audio": beatmap["General"]["AudioFilename"] if "AudioFilename" in beatmap["General"] else "audio.mp3",
            "difficulty": min(1., beatmap["Difficulty"]["OverallDifficulty"] / 7.),
            # rest of the metadata
            "preview": int(beatmap["General"]["PreviewTime"]) if "PreviewTime" in beatmap["General"] else 0,
            "game_mode": int(beatmap["General"]["Mode"]) if "Mode" in beatmap["General"] else 0,
            "sample": beatmap["General"]["SampleSet"] if "SampleSet" in beatmap["General"] else "Normal",
            "countdown": int(beatmap["General"]["Countdown"]) if "Countdown" in beatmap["General"] else 0,
            "osu_difficulty": deepcopy(beatmap["Difficulty"]),   
            "info": deepcopy(beatmap["Metadata"]),
        }
        meta["game_type"] = ("osu", "taiko", "catch", "mania")[meta["game_mode"]]
        if meta["game_type"] in ("osu", "catch"):
            meta["tracks"] = ["LEFT"]
        elif meta["game_type"] == "taiko":
            meta["tracks"] = ["LEFT", "RIGHT"]
        else:
            meta["tracks"] = ["LEFT", "DOWN", "UP", "RIGHT"]
        
        # generate important events from multiple sources
        events = {}
        for tp in beatmap["TimingPoints"]:
            try:
                # time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
                parts = [elem.strip() for elem in tp.split(",")]
                time = self.snap(parts[0])
                if time not in events:
                    events[time] = []
                events[time].append({
                    "type": "timing",
                    "tempo": 60000 / float(parts[1]),  # convert beat duration to BPM
                    "meter": int(parts[2]),
                    "uninherited": parts[-2] == "1",  # uninherited is True (beatLength instead of velocity)
                    "kai": int(parts[-1]) & 1 == 1,  # effects bit 0 is on
                })
            except Exception:
                pass
        
        # get silent intervals
        if "AudioLeadIn" in beatmap["General"]:
            leadin = int(beatmap["General"]["AudioLeadIn"])
            if leadin > 0:
                if 0 not in events:
                    events[0] = []
                events[0].append({
                    "type": "empty",
                    "end": self.snap(leadin),
                })
        for ep in beatmap["Events"]:
            try:
                parts = [elem.strip() for elem in ep.split(",")]
                if parts[0] in (2, "2", "Break", "break"):
                    time = self.snap(parts[1])
                    if time not in events:
                        events[time] = []
                    events[time].append({
                        "type": "empty",
                        "end": self.snap(parts[2]),
                    })
            except Exception:
                pass
        
        # add hit events based on https://github.com/Syps/aisu_circles/
        for ho in beatmap["HitObjects"]:
            parts = [elem.strip() for elem in ho.split(",")]
            time = self.snap(parts[2])
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
                object_name = "circle"  # unknown
                
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
                "end": self.snap(parts[5].split(":")[0]) if object_name in ("spinner", "hold") else 0,
                "new_combo": int(object_type & 4 == 4),
            }
            if (int(parts[4]) & 2 == 2) or (int(parts[4]) & 8 == 8):
                # Hit circles with whistle or clap hitsounds become kats, and other hit circles become dons
                hit["kats"] = 1
            else:
                hit["kats"] = 0
            hit["pos"] = min(3, max(0, int(hit["x"] * 4. // 512)))
            if meta["game_type"] in ("osu", "catch"):
                hit["track"] = "LEFT"  # always left
            elif meta["game_type"] == "taiko":
                hit["track"] = "RIGHT" if hit["kats"] else "LEFT"  # don, katsu / left, right
            else:
                hit["track"] = meta["tracks"][hit["pos"]]  # left, down, up, right based on position

            events[time].append(hit)
        
        # turn found events into DF
        hit_df, on_hold = [], {}
        tempo, meter, velocity, kai = 0., 0, 1., 0
        additional = ["EMPTY", "NEW_COMBO"] if self.spinner_to_hold else ["SPINNER", "EMPTY", "NEW_COMBO"]
        for time in range(0, max(events.keys()) + self.QUANTIZE, self.QUANTIZE):
            row = {col.upper(): 0 for col in meta["tracks"] + additional}
            if time in events:
                for elem in events[time]:
                    if elem["type"] == "empty":
                        on_hold["EMPTY"] = elem["end"]
                    elif elem["type"] == "timing":
                        meter = elem["meter"]
                        kai = int(elem["kai"])
                        if elem["uninherited"]:
                            tempo = elem["tempo"]
                            velocity = 1.
                        else:
                            velocity = np.abs(100. / elem["tempo"]) if elem["tempo"] else 1.
                    else:
                        row["NEW_COMBO"] = elem["new_combo"]
                        if elem["type"] == "circle":
                            row[elem["track"]] = 1
                        elif elem["type"] == "spinner":
                            if self.spinner_to_hold:
                                for track in meta["tracks"]:
                                    on_hold[track] = elem["end"]
                            else:
                                on_hold["SPINNER"] = elem["end"]
                        elif elem["type"] == "hold":
                            on_hold[elem["track"]] = elem["end"]
                        elif elem["type"] == "slider":
                            # I'm just gonna believe you on this one https://github.com/Syps/aisu_circles/
                            px_per_beat = meta["osu_difficulty"]["SliderMultiplier"] * 100. * velocity
                            beats_number  = (elem["length"] * elem["repeat"]) / px_per_beat
                            on_hold[elem["track"]] = time + np.ceil(beats_number * tempo)
            for key, value in on_hold.items():
                if value >= time:
                    row[key] = 2 if key in self.TRACKS else 1
            hit_df.append({**{"TIME": time, "TEMPO": tempo, "METER": meter, "VELOCITY": velocity, "KAI": kai}, 
                           **row})
        
        return meta, pd.DataFrame(hit_df).to_dict("list")

    @staticmethod
    def get_x_y_angle(last_x: float = 256., last_y: float = 192., last_angle: float = 0.0, d: int = 12) -> Tuple[float, float, float]:
        # 640 x 334 VS 512 x 384 grid
        rot = np.random.choice([-2., -1., 1., 2.]) 
        bound = np.random.choice([3, 4, 6])
        while True:
            x = last_x + (d * np.cos(last_angle))
            y = last_y + (d * np.sin(last_angle))
            if x <= bound or x >= 512 - bound or y <= bound or y >= 384 - bound:
                last_angle += rot * 30. * np.pi / 180.
                d += 1
            else:
                break
        return x, y, last_angle + np.random.normal(0, .2)
        
    def generate(self, 
                 input_file: str, 
                 output_file: str, 
                 *,
                 meta: dict = {}) -> None:
        ibf = IntermediateBeatmapFormat(input_file)

        results = ["[HitObjects]"]
        x, y, angle = np.random.randint(32, 512 - 32), np.random.randint(32, 384 - 32), np.random.rand() * 2. - 1.
        last_x, last_y = x, y
        rolling_indicies = [0]
        is_new, new_events, slider_happened = True, 0, False
        holds = {col: {} for col in ibf.meta["tracks"]}
        for index, row in ibf.data.iterrows():
            for check in ibf.meta["tracks"]:
                if row[check] == 2:
                    if holds[check]:
                        continue
                    holds[check] = {"TIME": row["TIME"], "x": x, "y": y, "angle": angle}
                    slider_happened = False
                elif row[check] == 1:
                    if np.abs(last_x - x) > 120 or np.abs(last_y - y) > 120:
                        is_new = True
                    elif new_events % 24 == 0:
                        is_new = True
                    elif len(rolling_indicies) > 12:
                        if np.median(rolling_indicies[-24:]) // 10 != (index - rolling_indicies[-1]) // 10:
                            is_new = True
                        else:
                            is_new = False
                        rolling_indicies = [rolling_indicies[-1]]
                    else:
                        is_new = False
                    if is_new:
                        _, _, angle = self.get_x_y_angle(x, y, angle)
                        new_events = 0
                    last_x, last_y = x, y
                    rolling_indicies.append(index - rolling_indicies[-1])
                    new_events += 1
                    slider_happened = False
                    hit = f"{int(x)},{int(y)},{int(row['TIME'])},{21 if is_new else 1},{2 if is_new else 0}"  # 0,1:0:0:0:
                    results.append(hit)
                else:
                    if holds[check]:
                        x, y, angle = holds[check]['x'], holds[check]['y'], holds[check]['angle']
                        # length = ms * (SliderMultiplier * 100 * SV) / beatlength
                        length = (row['TIME'] - holds[check]['TIME']) * 100 / 360
                        hit = f"{int(x)},{int(y)},{int(holds[check]['TIME'])},2,0,B"
                        for _ in range(int(length / 12) + 1):
                            x, y, angle = self.get_x_y_angle(x, y, angle, d=12)
                            hit = f"{hit}|{int(x)}:{int(y)}"
                        holds[check] = {}
                        is_new = False
                        slider_happened = True
                        last_x, last_y = x, y
                        rolling_indicies.append(index - rolling_indicies[-1])
                        hit = f"{hit},1,{int(length / 12 * 11)}"
                        results.append(hit)
                    else:
                        if index % 12 == 0 and not slider_happened:
                            x, y, angle = self.get_x_y_angle(x, y, angle)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            header_meta = {**self.BEATMAP_DEFAULTS, **ibf.meta, **meta}
            ous_file = os.path.join(temp_dir, header_meta["osu_file"])
            abspath = os.path.abspath(os.path.dirname(__file__))
            
            if header_meta["bg"] is None:
                bg, header_meta["bg"] = os.path.join(abspath, "static/BG.png"), "BG.png"
            else:
                bg, header_meta["bg"] = header_meta["bg"], os.path.basename(header_meta["bg"])
            if os.path.exists(header_meta["audio"]):
                audio = header_meta["audio"]
            else:
                audio = os.path.join(os.path.dirname(input_file), header_meta["audio"])
                assert os.path.exists(audio), "Could not resolve audio file path from metadata!"
            header_meta["audio"] = os.path.basename(header_meta["audio"])

            with open(os.path.join(abspath, "static/osu_header.txt"), "r") as f:
                contents = f.read().format(**header_meta) + "\n".join(results)
            with open(ous_file, "w", encoding="utf-8") as f:
                f.write(contents)
        
            zip_file = os.path.join(temp_dir, "beatmap.zip")
            with ZipFile(zip_file, "w") as zip_object:
                zip_object.write(ous_file, arcname=header_meta["osu_file"])
                zip_object.write(audio, arcname=header_meta["audio"])
                zip_object.write(bg, arcname=header_meta["bg"])

            copyfile(zip_file, output_file)
