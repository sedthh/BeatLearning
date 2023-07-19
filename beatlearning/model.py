import numpy as np

import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Tuple
import logging
#logging.basicConfig(format='%(asctime)s %(levelname)s > %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

from .utils import tokenize


@dataclass
class Config:
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
    hits_vocab_size: int = 4097  # 12 bins / 1 sec + empty
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
        
    output_fc: int = 64  # number of neurons in first fully connected layer after decoder
    output_dropout: float = 0.  # dropout right before logits
    output_activation: str = "gelu"
    output_bins: int = 12 # audio_bins (number of logits = output_bins * output_tracks + 2)
    output_tracks: int = 2  # nubmer of tracks, including holds
    

class PositionalEncoding(nn.Module):
    # TODO: probably not the best representation for audio tokens
    # NOTE: order is batch first
    
    def __init__(self, n_embd: int, block_size: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-np.log(10000.0) / n_embd))
        pe = torch.zeros(1, block_size, n_embd)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: shape is (batch_size, block_size, nembd) unlike in the Pytorch example
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class OsuTransformerOuendan(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.model_type = "Transformer"
        self.config = config
        
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.manual_seed(self.config.seed)
            torch.backends.cudnn.deterministic = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logging.info(f"Torch device found: {device.type}")

        ##### MODEL 
        ### audio token encoder (unmasked)
        self.audio_token_embedding = nn.Embedding(self.config.audio_vocab_size, self.config.audio_embd)
        self.audio_position_embedding = PositionalEncoding(self.config.audio_embd, self.config.audio_block_size)
        self.audio_encoder_layers = nn.TransformerEncoderLayer(self.config.audio_embd, 
                                                               self.config.audio_head, 
                                                               self.config.audio_ff, 
                                                               self.config.audio_dropout,
                                                               activation=self.config.audio_activation,
                                                               batch_first=True)
        self.audio_transformer_encoder = nn.TransformerEncoder(self.audio_encoder_layers, self.config.audio_layer)
        self.audio_projection = nn.Linear(self.config.audio_block_size * self.config.audio_embd + self.config.meta_features, 
                                        self.config.hits_block_size * self.config.hits_embd, bias=False)
        
        ### meta features
        self.meta_fc = nn.Linear(self.config.meta_features, self.config.meta_fc, bias=True)
        assert self.config.meta_activation in ("relu", "gelu"), "Only relu and gelu activations are supported."
        self.meta_activation = nn.ReLU() if self.config.meta_activation == "relu" else nn.GELU()
        self.meta_dropout = nn.Dropout(self.config.meta_dropout)
        
        ### hits token decoder (unmasked) for each track
        self.hits_token_embedding_list = nn.ModuleList()
        self.hits_position_embedding = PositionalEncoding(self.config.hits_embd, self.config.hits_block_size)
        self.hits_transformer_decoder_list = nn.ModuleList()
        self.hits_generate_mask = nn.Transformer().generate_square_subsequent_mask
        for i in range(self.config.output_tracks):
            self.hits_token_embedding_list.append(nn.Embedding(self.config.hits_vocab_size, self.config.hits_embd))
            if i:  # tie embedding weights
                self.hits_token_embedding_list[-1].weight = self.hits_token_embedding_list[0].weight
            hits_decoder_layers = nn.TransformerDecoderLayer(self.config.hits_embd, 
                                                             self.config.hits_head, 
                                                             self.config.hits_ff,
                                                             self.config.hits_dropout,
                                                             activation=self.config.hits_activation,
                                                             batch_first=True)
            self.hits_transformer_decoder_list.append(nn.TransformerDecoder(hits_decoder_layers, self.config.hits_layer))
        
        ### output
        self.output_layer_norm = nn.LayerNorm(self.config.hits_block_size * self.config.hits_embd)
        self.output_fc = nn.Linear(self.config.hits_block_size * self.config.hits_embd, self.config.output_fc, bias=True)
        assert self.config.output_activation in ("relu", "gelu"), "Only relu and gelu activations are supported."
        self.output_activation = nn.ReLU() if self.config.output_activation == "relu" else nn.GELU()
        self.output_dropout = nn.Dropout(self.config.output_dropout)
        self.output_head = nn.Linear(self.config.output_fc, self.config.output_bins * self.config.output_tracks + 2, bias=False)
        self.output_sigmoid = nn.Sigmoid()  # will only be used for generation, BCE takes logits
        
        self.losses = {
            "bce": nn.BCEWithLogitsLoss(reduction="mean"),
            "mse": nn.MSELoss(reduction="none"),  # reduced manually after applying weights
        }
        self.apply(self._init_weights)
        #####
        
        # report number of parameters via NanoGPT
        logging.info(f"Number of parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.03)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
    def forward(self,
                audio_tokens: torch.Tensor, 
                hits_tokens: Optional[torch.Tensor] = None,
                meta_data: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = audio_tokens.device
        
        # audio
        batch_size, at = audio_tokens.size()
        assert at <= self.config.audio_block_size, f"Cannot forward audio token sequence of length {at}: audio block size is {self.config.audio_block_size}"
        audio_emb = self.audio_token_embedding(audio_tokens)
        audio_emb = self.audio_position_embedding(audio_emb)
        audio_attention = self.audio_transformer_encoder(audio_emb, None)
        
        # meta
        if meta_data is None:
            meta_data = torch.zeros((batch_size, self.config.meta_features), device=device, requires_grad=False)
        meta_output = self.meta_dropout(self.meta_activation(self.meta_fc(meta_data)))
        
        # join audio and meta then convert it to decoder input size by linear projection
        encoder_output = self.audio_projection(torch.concat([audio_attention.view(batch_size, -1), meta_output], axis=1))
        encoder_output = encoder_output.view(batch_size, self.config.hits_block_size, self.config.hits_embd)
        
        # hits (decoder outputs for each track are summed)
        if self.config.output_tracks:
            if hits_tokens is None:
                hits_tokens = torch.zeros((batch_size, self.config.output_tracks, self.config.hits_block_size), device=device, dtype=audio_tokens.dtype, requires_grad=False)
            _, tracks, ht = hits_tokens.size()
            assert ht <= self.config.hits_block_size, f"Cannot forward hits sequence of token length {ht}: block size is {self.config.hits_block_size}"
            assert tracks <= self.config.output_tracks, f"Cannot forward hits sequence of track length {tracks}: expected number of tracks is {self.config.output_tracks}"
            
            if self.config.hits_mask:
                tgt_mask = self.hits_generate_mask(self.config.hits_block_size, device=device)
            else:
                tgt_mask = None
            
            for i in range(self.config.output_tracks):
                hits_emb = self.hits_token_embedding_list[i](hits_tokens[:, i, :].view(batch_size, -1))
                hits_emb = self.hits_position_embedding(hits_emb)
                if i:
                    hits_attention += self.hits_transformer_decoder_list[i](hits_emb, encoder_output, tgt_mask=tgt_mask)
                else:
                    hits_attention = self.hits_transformer_decoder_list[i](hits_emb, encoder_output, tgt_mask=tgt_mask)
        
        # output
        if self.config.output_tracks:
            all_output = hits_attention.view(batch_size, -1)
            all_output += encoder_output.view(batch_size, -1)
            all_output = self.output_layer_norm(all_output)
        else:
            all_output = encoder_output.view(batch_size, -1)
        all_output = self.output_dropout(self.output_activation(self.output_fc(all_output)))
        
        # prediction
        logits = self.output_head(all_output)
        
        if targets is not None:  
            # predict beats as multiclass
            if self.config.output_tracks:
                loss_beats = self.losses["bce"](logits[:, :-2], targets[:, :-2])  #/ float(self.config.output_tracks)
            else:
                loss_beats = 0
            # apply weights to regression loss only
            loss_metronome = self.losses["mse"](logits[:, -2:], targets[:, -2:])
            if weights is None:
                loss_metronome = torch.mean(loss_metronome)
            else:
                loss_metronome = torch.mean(loss_metronome * weights.view(-1, 1))
            loss = loss_beats + loss_metronome
        else:
            loss = None
        
        return torch.cat([self.output_sigmoid(logits[:, :-2]), logits[:, -2:]], axis=1), loss

    @torch.no_grad()
    def generate(self, 
                 audio_tokens_list: torch.Tensor, 
                 hits_tokens: Optional[torch.Tensor] = None,
                 meta_data: Optional[torch.Tensor] = None,
                 hit_threshold: float = .8,
                 hit_random: float = .4):
        device = audio_tokens_list.device
        if hits_tokens is None:
            hits_tokens = [[0 for _hits in range(self.config.hits_block_size)] for _tracks in range(self.config.output_tracks)]
        for audio_tokens in audio_tokens_list:
            results, _ = self(audio_tokens=audio_tokens.view(1, -1), 
                              hits_tokens=torch.Tensor([hits_tokens]).type(torch.long).to(device),
                              meta_data=meta_data if meta_data is not None else None)
            results = results.squeeze()
            all_hits, offset, tempo = results[:-2], float(results[-2].item()), float(results[-1].item())
            allowed_hits_list, token_list = [], []
            for i in range(self.config.output_tracks):
                hits = all_hits[i * self.config.output_bins:(i + 1) * self.config.output_bins]
                allowed_hits = [] 
                for hit in hits:
                    if hit > hit_threshold:
                        allowed_hits.append(1)
                    elif hit > hit_random:
                        r = (hit - hit_random) / (hit_threshold - hit_random)
                        if np.random.random() <= r:
                            allowed_hits.append(1)
                        else:
                            allowed_hits.append(0)
                    else:
                        allowed_hits.append(0)
                hits_tokens[i].pop(0)
                token = tokenize(allowed_hits) + 1
                hits_tokens[i].append(token)
                allowed_hits_list.append(allowed_hits)
                token_list.append(token)
            yield allowed_hits_list, token_list, offset, tempo
