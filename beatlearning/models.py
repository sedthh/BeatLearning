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
               temperature: float = 1.0,
               top_k: int = 1,  # top_k for predictions, not the same as top_k for generate
               logit_bias: Dict[int, float] = {},
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            assert temperature > 0.0, "Temperature must be a positive float!"
            assert top_k > 0, "The value of top_k must be positive integer!"

            regression, classification = self(input_data, segment_data, input_audio)
            tempo = torch.maximum(torch.zeros_like(regression), regression.detach() * self.tokenizer.config.tempo_modifier)
            prediction = classification.detach()[torch.arange(len(output_mask)), output_mask] / temperature
            
            # alter logits based on token id & ensure the model never predicts reserved tokens
            prediction[:, :len(self.tokenizer.RESERVED_TOKENS)] = -float('Inf')
            for token, value in logit_bias.items():
                prediction[:, int(token)] += value

            top_values, top_indicies = torch.topk(prediction, top_k)
            return tempo, top_values, top_indicies

    def generate(self, 
                 audio_file: str,
                 audio_start: float = 0.0,
                 audio_end: Optional[float] = None,
                 use_tracks: List[str] = ["LEFT"],
                 difficulty: float = 0.5, 
                 *,
                 temperature: float = 0.5,
                 beams: List[int] = [],
                 max_beam_width: Optional[int] = None,
                 top_k: int = 1,  # not the same as top_k for sample()
                 logit_bias: Dict[int, float] = {},
                 random_seed: Optional[int] = None,
                 ) -> torch.Tensor:
        if random_seed is not None:
            torch.manual_seed(random_seed)
        assert np.all([col in self.tokenizer.config.tracks for col in use_tracks]), "Unknown track type!"
        assert top_k > 0, "The value of top_k must be positive integer!"
        device = next(self.parameters()).device

        audio_features = self.tokenizer.audio_converter(audio_file)
        audio_features_len = len(audio_features) - 2 * self.tokenizer.config.context_length
        audio_start_ = int(audio_start * self.tokenizer.QUANTIZE)
        audio_end_ = int(audio_end * self.tokenizer.QUANTIZE) if audio_end is not None else None

        cls = self.tokenizer.RESERVED_TOKENS[self.tokenizer._tokenize_difficulty(difficulty)]
        window = self.tokenizer.config.context_length // len(use_tracks) - 1
        assert self.tokenizer.config.audio_foresight % len(use_tracks) == 0, f"The value of 'audio_forsight' is not divisible by the number of tracks!"
        foresight = self.tokenizer.config.audio_foresight // len(use_tracks)
        if not beams:
            beams = [max(2, top_k)] * len(use_tracks)
        assert not len(beams) % len(use_tracks), "Search beam length must be an integer multiple of the number of tracks!"

        with torch.no_grad():
            segment_data = torch.Tensor(self.tokenizer._generate_segment(len(use_tracks))).unsqueeze(0).long().to(device)
            positions = self.tokenizer._generate_mask(len(use_tracks))
            result = {col: [] for col in use_tracks + ["TEMPO"]}
            for step in tqdm(range(0, audio_features_len - (len(beams) * 2), len(beams))):
                if step < audio_start_:
                    continue
                elif audio_end_ is not None and step >= audio_end_:
                    break
                
                # O(NM) beam search over masked slices of tokens cycling from left to right
                tempos = []
                input_data = []
                for i, track in enumerate(use_tracks):
                    input_data += [self.tokenizer.RESERVED_TOKENS["SEP"] if i else cls]
                    input_data += [self.tokenizer.RESERVED_TOKENS["PAD"] for _ in range(max(0, window - foresight - len(result[track])))] 
                    input_data += result[track][-(window - foresight):]
                    input_data += [self.tokenizer.RESERVED_TOKENS["FORESIGHT"] for _ in range(foresight)]

                input_data = torch.Tensor(np.array(input_data).astype(np.int32)).unsqueeze(0).long().to(device)
                output_mask = torch.Tensor([positions[0]]).long().to(device)
                scores = torch.ones(1).float().to(device) 
                empty_audio = True
                for b, beam in enumerate(beams):
                    input_audio = self.tokenizer._get_audio_chunk(audio_features, 
                                                                  position=step + b + foresight + self.tokenizer.config.context_length,
                                                                  number_of_tracks=len(use_tracks))
                    if empty_audio:
                        empty_audio = np.all(input_audio == 0.0)
                    input_audio = torch.Tensor(input_audio.astype(np.float32)).unsqueeze(0).float().to(device)

                    # correct missing masks while doing beam search
                    if not torch.sum(input_data == self.tokenizer.RESERVED_TOKENS["MASK"]):
                        stacks = []
                        for position in positions:
                            stacks += [input_data[:, position + foresight - window:position + foresight - window + 1], 
                                       input_data[:, position + foresight - window + 2:position + 1], 
                                       torch.ones_like(input_data[:, position:position + 1]).to(device) * self.tokenizer.RESERVED_TOKENS["MASK"],
                                       input_data[:, position + 1:position + foresight + 1]]
                        input_data = torch.hstack(stacks)

                    # sample beam number of top candidates
                    tempo, top_values, top_indicies = self.sample(input_data,
                                                                    segment_data, 
                                                                    input_audio,
                                                                    output_mask,
                                                                    temperature = temperature,
                                                                    top_k = beam,
                                                                    logit_bias = logit_bias)
                    top_values = F.softmax(top_values, dim=-1)

                    # expand dimensions so every element of the next beam can be sampled as a single batch
                    tempos.append(torch.mean(tempo).numpy(force=True))
                    input_data = torch.repeat_interleave(input_data, beam, axis=0)
                    output_mask = torch.Tensor([positions[(b + 1) % len(positions)]] * output_mask.shape[0] * beam).long().to(device)
                    scores = torch.repeat_interleave(scores, beam, axis=0)
                    scores *= top_values.ravel()

                    # add predictions to the correct masked position
                    stacks = []
                    for position in positions:
                        if positions[b % len(positions)] == position:
                            stacks += [input_data[:, position + foresight - window:position], 
                                       top_indicies.ravel().view(-1, 1),
                                       input_data[:, position + 1:position + foresight + 1]]
                        else:
                            stacks += [input_data[:, position + foresight - window:position + foresight + 1]]
                    input_data = torch.hstack(stacks)

                    # maximize number of concurent beams
                    if max_beam_width is not None and input_data.shape[0] > max_beam_width:
                        _, remaining_indicies = torch.topk(scores, max_beam_width)
                        input_data = input_data[remaining_indicies]
                        output_mask = output_mask[remaining_indicies]
                        scores = scores[remaining_indicies]

                # select best candidates out of all (remaining) beams
                scores = scores.ravel()
                top_scores, _ = torch.topk(scores, top_k + 1)
                scores[scores < top_scores[-1]] = -float('Inf')  # exclude worse scores
                if result["TEMPO"] and not empty_audio:
                    scores[scores == top_scores[0]] = -float('Inf')  # top result is almost always all empty due to softmax multiplications
                choice = torch.multinomial(F.softmax(scores / temperature, dim=-1), num_samples=1)
                generated = input_data[choice].numpy(force=True).ravel()

                for position, track in zip(positions, use_tracks):
                    result[track] += generated[position - (len(beams) // len(use_tracks)) + 1: position + 1].tolist()
                result["TEMPO"] += tempos
                
        data = self.tokenizer.decode(result, offset=audio_start, use_tracks=use_tracks)
        assert np.sum(data[use_tracks].values), "No hit events have been generated! Consider tweaking your top_k and / or temperature values!"
        meta = {
            "audio": audio_file,
            "difficulty": difficulty,
            "tracks": use_tracks,
        }
        return IntermediateBeatmapFormat(data=data, meta=meta)