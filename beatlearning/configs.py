from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BEaRTConfig:
    groups: int
    context_length: int  # for beat event tokens (aka max_position_embeddings in BERT)
    audio_foresight: int  # padded area (from the right) with audio information only
    embedding_size: int
    attention_layers: int
    attention_heads: int
    attention_feedforwad: int  # embedding_size * 4
    mel_bands: int
    mel_gradient: bool = False
    attention_dropout: float = 0.1
    embedding_drouput: float = 0.1
    attention_activation: str = "gelu"
    tracks: List = field(default_factory=lambda: ["LEFT", "UP", "DOWN", "RIGHT"])
    track_combinations: List = field(default_factory=lambda: [1, 2, 4])
    tempo_modifier: float = 100.  # match regression head's range with tempo
    dataset_dropout: Optional[float] = 0.2  # randomly mask beat tokens throughout training
    dataset_mel_scaling: Optional[Tuple[float, float]] = (0.0, 1.1)
    dataset_mel_noise: Optional[float] = 0.05
    random_seed: int = 69420


@dataclass
class QuaverBEaRT(BEaRTConfig):
    groups: int = 10
    context_length: int = 512
    audio_foresight: int = 64
    mel_bands: int = 32
    embedding_size: int = 32
    attention_layers: int = 8
    attention_heads: int = 4
    attention_feedforwad: int = 16 * 4


@dataclass
class CrotchetBEaRT(BEaRTConfig):
    groups: int = 10
    context_length: int = 1024
    audio_foresight: int = 128
    mel_bands: int = 32
    embedding_size: int = 32
    attention_layers: int = 8
    attention_heads: int = 4
    attention_feedforwad: int = 16 * 4
    

@dataclass
class TrainingConfig:
    use_cuda_if_available: bool = True
    batch_size_train: int = 1024
    batch_size_test: int = 2048
    num_epochs: int = 100
    early_stopping_rounds: Optional[int] = 10
    warmp_up_rounds: Optional[int] = None
    learning_rate: float = 3e-4
    learning_rate_decay: Optional[float] = 0.98
    weight_decay: float = 1e-5
    weight_decay_embeddings: bool = True
    num_workers: int = 0  # number of DataLoader workers
    grad_norm: Optional[float] = 1.0
    log_frequency: int = 100
    model_directory: Optional[str] = "models/"
    tensorboard_directory: Optional[str] = "runs/"
    regression_loss_weight: float = 0.1
    classification_loss_weight: float = 0.9
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    check_for_sudden_drops_in_the_loss_function: bool = True
