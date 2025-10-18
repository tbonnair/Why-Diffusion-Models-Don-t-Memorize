import torch
from torch import nn
import numpy as np

class DiscreteTimeResidualBlock(nn.Module):
    """Generic block to learn a nonlinear function f(x, t), where
    t is discrete and x is continuous."""

    def __init__(self, d_model: int, maxlen: int = 512):
        super().__init__()
        self.d_model = d_model
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.norm(x + self.lin2(self.act(self.lin1(x))))


class BasicDiscreteTimeModel(nn.Module):
    def __init__(self, d: int, d_model: int = 128, n_layers: int = 3):
        super().__init__()
        self.d = d
        self.d_model = d_model
        self.n_layers = n_layers
        self.lin_in = nn.Linear(d, d_model)
        self.lin_out = nn.Linear(d_model, d)
        self.blocks = nn.ParameterList(
            [DiscreteTimeResidualBlock(d_model=d_model) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.lin_in(x)
        for block in self.blocks:
            x = block(x)
        return self.lin_out(x)
    
    
    
# From https://github.com/Jmkernes/Diffusion/blob/main/diffusion/ddpm/models.py
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model=128,
        maxlen=512,
        min_freq=1e-4,
        device='cpu',
        dtype=torch.float32,
    ):
        super().__init__()
        pos_enc = self._get_pos_enc(d_model=d_model, maxlen=maxlen, min_freq=min_freq)
        self.register_buffer(
            "pos_enc", torch.tensor(pos_enc, dtype=dtype, device=device)
        )

    def _get_pos_enc(self, d_model, maxlen, min_freq):
        position = np.arange(maxlen)
        freqs = min_freq ** (2 * (np.arange(d_model) // 2) / d_model)
        pos_enc = position[:, None] * freqs[None]
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        return pos_enc

    def forward(self, x):
        return self.pos_enc[x]


class SimpleResidualBlock(nn.Module):
    def __init__(self, d_model=128, maxlen=10000):
        super().__init__()
        self.d_model = d_model
        self.emb = PositionalEncoding(d_model=d_model, maxlen=maxlen)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x, t):
        out = x + self.emb(t)       # Add the positional encoding
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return self.norm(x + out)

class SimpleTimeModel(nn.Module):
    def __init__(self, d, d_model=128, n_blocks=3):
        super().__init__()
        self.d = d
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.in_layer = nn.Linear(d, d_model)
        self.out_layer = nn.Linear(d_model, d)
        self.blocks = nn.ParameterList(
            [SimpleResidualBlock(d_model=d_model) for _ in range(n_blocks)]
        )

    def forward(self, x, t):
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x, t)
        return self.out_layer(x)