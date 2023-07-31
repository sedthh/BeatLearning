from dataclasses import dataclass
from typing import Optional


@dataclass
class BasicModelConfig:
    seed: int = 42069  # Achieved Comedy
    
    audio_block_size: int = 150  # EnCodec 24khz
    audio_vocab_size: int = 1024  # EnCodec 24khz
    audio_layer: int = 4
    audio_head: int = 2
    audio_embd: int = 64
    audio_ff: int = 512
    audio_dropout: float = 0.1
    audio_activation: str = "gelu"
    
    hits_block_size: int = 32 
    hits_vocab_size: int = 65  # group every 2nd in 12 bins / 1 sec + empty -> 2**6+1
    hits_layer: int = 2
    hits_head: int = 1
    hits_embd: int = 16
    hits_ff: int = 128
    hits_dropout: float = 0.1
    hits_activation: str = "gelu"
    hits_mask: bool = True  # mask self self-attention part of the decoder
        
    meta_features: int = 2  # more input values (time, difficulty)
    meta_fc: int = 2  # number of neurons in layer after meta inputs
    meta_dropout: float = 0.  # could be useful with more fatures
    meta_activation: str = "gelu"
    meta_enable: bool = True  # enable metadata loading and projection to decoder
        
    output_fc: int = 64  # number of neurons in first fully connected layer after decoder
    output_dropout: float = 0.  # dropout right before logits
    output_activation: str = "gelu"
    output_bins: int = 12  # audio_bins (number of logits = output_bins * output_tracks + 2)
    output_tracks: int = 2  # nubmer of tracks, including holds
    output_token_reduction: int = 2  # group output tokens to speed up training

@dataclass    
class AudioModelConfig(BasicModelConfig):
    meta_enable: bool = False  # disable meta data input
    output_tracks: int = 0  # disable hit classification output
    output_dropout: float = 0.  # dropout right before logits


@dataclass
class BasicTrainConfig:
    device: str = "cpu"
    experiment: str = "model"  # prefix for the generated model
    epochs: int = 100
    batch_size: int = 16
    early_stopping: int = 20
    checkpoints: Optional[str] = "models"  # save state dicts to folder, use None to disable
    mode: Optional[int] = 1  # 1 = OSU data
    lr: float = 3e-4
    wd: float = 3e-5  # weight decay
    clip_norm: float = 1.
    freeze_encoder: bool = False  # unfreeze audio encoder (+ meta projection)
    
@dataclass    
class AudioTrainConfig(BasicTrainConfig):
    experiment: str = "audio-pretrain"  # prefix for the generated model
    mode: Optional[int] = None  # all data
    freeze_encoder: bool = False  # unfreeze audio encoder (+ meta projection)
