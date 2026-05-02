import torch.nn as nn


class SambaBlock(nn.Module):
    """Placeholder temporal MLP block (NOT full Samba/Mamba implementation).

This class only provides a trainable temporal transform hook for stage-1
integration. It intentionally does not include Samba's mamba/selective-scan core.
"""

    def __init__(self, dim: int, hidden_mult: int = 2, dropout: float = 0.0, **kwargs):
        super().__init__()
        hidden = dim * hidden_mult
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))
