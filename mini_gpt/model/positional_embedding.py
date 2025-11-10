import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, block_size: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, block_size, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
