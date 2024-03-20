import numpy as np
from tqdm import tqdm

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import BEaRTTokenizer
from .utils import IntermediateBeatmapFormat


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_size: int, context_length: int = 512):
        super().__init__()

        position = torch.arange(context_length).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-np.log(10000.0) / embedding_size))
        
        pe = torch.zeros(1, context_length, embedding_size).float()
        pe.requires_grad = False
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape: batch_size, context_length, embedding_size
        return self.pe[:x.size(0)]

   
class BEaRT(nn.Module):

    def __init__(self, tokenizer: BEaRTTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

        self.input_embedding        = nn.Embedding(self.tokenizer.vocab_size, self.tokenizer.config.embedding_size)
        self.positional_embedding   = PositionalEncoding(self.tokenizer.config.embedding_size, self.tokenizer.config.context_length)
        self.segment_embedding      = nn.Embedding(len(self.tokenizer.config.tracks), self.tokenizer.config.embedding_size)
        self.audio_embedding        = nn.Linear(self.tokenizer.config.groups * self.tokenizer.config.mel_bands, self.tokenizer.config.embedding_size, bias=False)
        self.dropout_embedding      = nn.Dropout(self.tokenizer.config.embedding_drouput)

        self.transformer_layer      = nn.TransformerEncoderLayer(self.tokenizer.config.embedding_size, 
                                                                 self.tokenizer.config.attention_heads, 
                                                                 self.tokenizer.config.attention_feedforwad, 
                                                                 dropout=self.tokenizer.config.attention_dropout,
                                                                 activation=self.tokenizer.config.attention_activation,
                                                                 batch_first=True)
        self.transformers           = nn.TransformerEncoder(self.transformer_layer, self.tokenizer.config.attention_layers)

        self.regression_head        = nn.Linear(self.tokenizer.config.embedding_size, 1, bias=False)
        self.regression_loss        = nn.MSELoss()
        self.classification_head    = nn.Linear(self.tokenizer.config.embedding_size, self.tokenizer.vocab_size, bias=False)
        self.classification_loss    = nn.CrossEntropyLoss()
        self._fix_weights()

    @property
    def num_params(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    def _fix_weights(self) -> None:
        torch.manual_seed(self.tokenizer.config.random_seed)
        # based on https://github.com/karpathy/nanoGPT/
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

        self.apply(_init_weights)
        std = 0.02 / np.sqrt(2 * self.tokenizer.config.attention_layers)
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(param, mean=0.0, std=std)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        
    def forward(self, 
                input_data: torch.Tensor,
                segment_data: torch.Tensor,
                input_audio: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # generate embedding (batch first)
        
        x = self.input_embedding(input_data)
        x += self.positional_embedding(x)
        x += self.segment_embedding(segment_data)
        x += self.audio_embedding(input_audio.view(-1, self.tokenizer.config.context_length, self.tokenizer.config.groups * self.tokenizer.config.mel_bands))
        x = self.dropout_embedding(x)

        # encoder (allow all padding due to possible audio overlaps, batch first)
        x = self.transformers(x, mask=None)

        # regression task at position 0 (aka CLS token)
        regression = self.regression_head(x[:, 0, :])
        # classification task (apply to all so it's easier to calculate mask indicies, but really it should be x[:, 1:, :])
        classification = self.classification_head(x)

        return regression, classification
           
    def hearing_loss(self,
                     regression: torch.Tensor,
                     classification: torch.Tensor,
                     output_data: torch.Tensor,
                     output_mask: torch.Tensor,
                     tempo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # weigth by number of tracks, divide tempo by 100. so it is closer in range to the regression head's output        
        loss_regression = self.regression_loss(tempo[tempo > 0.0] / self.tokenizer.config.tempo_modifier, 
                                               regression.squeeze(-1)[tempo > 0.0]) 
        loss_regression = torch.where(torch.isfinite(loss_regression), loss_regression, torch.zeros_like(loss_regression))

        output_onehot = F.one_hot(output_data.long(), num_classes=self.tokenizer.vocab_size).float()
        mask_helper = torch.arange(len(output_mask))
        loss_classification = self.classification_loss(classification[mask_helper, output_mask], output_onehot[mask_helper, output_mask])

        return loss_regression, loss_classification
    
    def sample(self, 
               input_data: torch.Tensor,
               segment_data: torch.Tensor,
               input_audio: torch.Tensor,
               output_mask: torch.Tensor,
               *,
               results_only: bool=False,
               temperature: float = 1.0,
               top_k: Optional[int] = None,
               top_p: Optional[float] = None,
               logit_bias: Dict[int, float] = {},
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            assert temperature > 0.0, "Temperature must be a positive float!"
            regression, classification = self(input_data, segment_data, input_audio)
            tempo = torch.maximum(torch.zeros_like(regression), regression.detach() * self.tokenizer.config.tempo_modifier)
            prediction = classification.detach()[torch.arange(len(output_mask)), output_mask] / temperature
            
            # alter logits based on token id & ensure the model never predicts reserved tokens
            prediction[:, :len(self.tokenizer.RESERVED_TOKENS)] = -float('Inf')
            for token, value in logit_bias.items():
                prediction[:, int(token)] += value

            # both top_p and top_k can be used
            if top_p is not None:
                assert top_p > 0.0 and top_p <= 1.0, "The value of top_p must be positive float between (0. - 1.]"
                # via https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
                sorted_values, sorted_indices = torch.sort(prediction, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_values, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs >= top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(prediction, dtype=sorted_indices_to_remove.dtype).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                prediction[indices_to_remove] = -float('Inf')
            if top_k is not None:
                assert top_k > 0, "The value of top_k must be positive integer!"
                top_values, _ = torch.topk(prediction, min(top_k, self.tokenizer.vocab_size - len(self.tokenizer.RESERVED_TOKENS)))
                prediction[prediction < top_values[:, [-1]]] = -float('Inf')
            
            assert not torch.all(torch.isinf(prediction)), "No values left to be selected!"
            if top_k is None and top_p is None:
                result = torch.argmax(prediction, axis=-1)
            else:
                result = torch.multinomial(F.softmax(prediction, dim=-1), num_samples=1)

            if results_only:
                return tempo, result
            else:
                output_data = input_data.clone().detach()
                output_data[torch.arange(len(output_mask)), output_mask] = result.squeeze()
                return tempo, output_data

    def generate(self, 
                 audio_file: str,
                 audio_start: float = 0.0,
                 audio_end: Optional[float] = None,
                 use_tracks: List[str] = ["LEFT"],
                 difficulty: float = 0.5, 
                 *,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 logit_bias: Dict[int, float] = {},
                 random_seed: Optional[int] = None,
                 ) -> torch.Tensor:
        if random_seed is not None:
            torch.manual_seed(random_seed)
        assert np.all([col in self.tokenizer.config.tracks for col in use_tracks]), "Unknown track type!"
        device = next(self.parameters()).device

        audio_features = self.tokenizer.audio_converter(audio_file)
        audio_features_len = len(audio_features) - 2 * self.tokenizer.config.context_length
        audio_start_ = int(audio_start * self.tokenizer.QUANTIZE)
        audio_end_ = int(audio_end * self.tokenizer.QUANTIZE) if audio_end is not None else None

        cls = self.tokenizer.RESERVED_TOKENS[self.tokenizer._tokenize_difficulty(difficulty)]
        window = self.tokenizer.config.context_length // len(use_tracks) - 1
        assert self.tokenizer.config.audio_foresight % len(use_tracks) == 0, f"The value of 'audio_forsight' is not divisible by the number of tracks!"
        foresight = self.tokenizer.config.audio_foresight // len(use_tracks)
        
        segment_data = torch.Tensor(self.tokenizer._generate_segment(len(use_tracks))).unsqueeze(0).long().to(device)

        result = {col: [] for col in use_tracks + ["TEMPO"]}
        for step in tqdm(range(audio_features_len), 
                         total=audio_end_ - audio_start_ if audio_end is not None else audio_features_len - audio_start_):
            if step < audio_start_:
                continue
            elif audio_end_ is not None and step >= audio_end_:
                break
            input_audio = self.tokenizer._get_audio_chunk(audio_features, 
                                                          position=step + foresight + self.tokenizer.config.context_length,
                                                          number_of_tracks=len(use_tracks))
            input_audio = torch.Tensor(input_audio.astype(np.float32)).unsqueeze(0).float().to(device)
            tempo = []
            for i, col in enumerate(use_tracks):
                input_data = []
                for j, track in enumerate(use_tracks):
                    input_data += [self.tokenizer.RESERVED_TOKENS["SEP"] if j else cls]
                    input_data += self.tokenizer._generate_pad(result[track], window, mask=i <= j)
                output_mask = [(window + 1) * (i + 1) - 1]
                t, p = self.sample(torch.Tensor(np.array(input_data).astype(np.int32)).unsqueeze(0).long().to(device), 
                                      segment_data, 
                                      input_audio,
                                      torch.Tensor(np.array(output_mask).astype(np.int32)).long().to(device),
                                      results_only = True,
                                      temperature = temperature,
                                      top_k = top_k,
                                      top_p = top_p,
                                      logit_bias = logit_bias)
                result[col].append(int(p.numpy(force=True).ravel()[0]))
                tempo.append(t.numpy(force=True))
            result["TEMPO"].append(np.nanmean(tempo))
        
        data = self.tokenizer.decode(result, offset=audio_start, use_tracks=use_tracks)
        meta = {
            "audio": audio_file,
            "difficulty": difficulty,
            "tracks": use_tracks,
        }
        return IntermediateBeatmapFormat(data=data, meta=meta)