from src.models.samba_block import SambaBlock


class IdentityTemporalAdapter:
    def __call__(self, z, masks_enc=None):
        return z


class SambaTemporalAdapter:
    def __init__(self, dim: int, depth: int, block_cfg: dict):
        self.blocks = [SambaBlock(dim=dim, **block_cfg) for _ in range(depth)]

    def __call__(self, z, masks_enc=None):
        x = z
        for blk in self.blocks:
            x = blk(x)
        return x
