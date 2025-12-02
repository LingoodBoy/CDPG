import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from conditional_denoiser.multiHeadAttention import MultiHeadAttention, MultiHeadAttentionChunk, MultiHeadAttentionWindow, MultiHeadCrossAttentionTest
from conditional_denoiser.positionwiseFeedForward import PositionwiseFeedForward


class CrossAttention_Encoder(nn.Module):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk'):
        """Initialize the CrossAttention_Encoder block"""
        super().__init__()

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(d_model, q, v, h, attention_size=attention_size)

        self._crossAttention = MultiHeadCrossAttentionTest(d_model)

        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)
        self._layerNorm3 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, kgtEmb: torch.Tensor) -> torch.Tensor:
        # Block 1, Self attention
        residual = x
        x = self._layerNorm1(x)
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dopout(x)
        x = x + residual

        # Block 2, Cross attention
        residual = x
        x = self._layerNorm2(x)
        x = self._crossAttention(x, kgtEmb)
        x = self._dopout(x)
        x = x + residual

        # Block 3, Feed forward
        residual = x
        x = self._layerNorm3(x)
        x = self._feedForward(x)
        x = self._dopout(x)
        x = x + residual

        return x
