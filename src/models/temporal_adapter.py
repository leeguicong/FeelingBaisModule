import torch.nn as nn
from src.models.samba_block import SambaBlock


class IdentityTemporalAdapter(nn.Module):
    def forward(self, z, masks_enc=None):
        return z


class SambaTemporalAdapter(nn.Module):
    def __init__(self, dim: int, depth: int, block_cfg: dict):
        super().__init__()
        self.blocks = nn.ModuleList([SambaBlock(dim=dim, **block_cfg) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, z, masks_enc=None):
        x = z
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
