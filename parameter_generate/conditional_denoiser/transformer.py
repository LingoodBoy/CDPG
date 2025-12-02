# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from einops import rearrange
import sys

from conditional_denoiser.encoder import CrossAttention_Encoder
from conditional_denoiser.utils import generate_original_PE, generate_regular_PE
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2 
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0, "dimension must be even"
        half_dim = dim // 2

        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):

        x = rearrange(x, 'b -> b 1')

        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi

        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)

        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class TransformerDenoiser(nn.Module):

    def __init__(self,
                 d_input: int,         
                 d_model: int,          
                 d_output: int,         
                 d_kgEmb: int,          
                 d_timeEmb: int,       
                 q: int,             
                 v: int,            
                 h: int,            
                 N: int,            
                 attention_size: int = None,   
                 layernum: int = 0,          
                 dropout: float = 0.3,       
                 chunk_mode: str = 'chunk',    
                 pe: str = None,               
                 pe_period: int = None,       
                 learned_sinusoidal_cond: bool = False, 
                 random_fourier_features: bool = False,  
                 learned_sinusoidal_dim: int = 16,     
                 ):
        super().__init__()

        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum
        self.self_condition = False

        self.layers_encoding = nn.ModuleList([
            CrossAttention_Encoder(d_model,
                                  q, v, h,                     
                                  attention_size=attention_size,
                                  dropout=dropout,
                                  chunk_mode=chunk_mode)
            for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'The position encoding type is not supported')

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.kgEmb_linear = nn.Linear(self.kgEmb_dim, d_model)
        self.timeEmb_linear = nn.Linear(self.timeEmb_dim, d_model)

        self.last_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        x2 = x.permute(0, 2, 1)                                 # [batch, seq, channels]

        kgEmb = self.kgEmb_linear(kgEmb)                        # [batch, d_model]
        kgEmb = kgEmb.unsqueeze(2)                              # [batch, d_model, 1]

        timeEmb = self.timeEmb_linear(timeEmb)                  # [batch, d_model]
        timeEmb = timeEmb.unsqueeze(2)                          # [batch, d_model, 1]

        kgtEmb = torch.cat((kgEmb, timeEmb), 2)

        step = self.step_mlp(t)                                 # [batch, d_model]
        step = step.unsqueeze(1)                                # [batch, 1, d_model]
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)  # [batch, seq, d_model]

        encoding = self._embedding(x2)                          # [batch, seq, d_model]
        encoding.add_(step_emb)                               

        K = self.layernum

        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        for layer in self.layers_encoding:
            encoding = layer(encoding, kgtEmb)

        output = self._linear(encoding)
        return output.permute(0, 2, 1)
