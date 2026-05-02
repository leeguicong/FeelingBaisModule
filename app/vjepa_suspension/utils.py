from src.models.video_encoder import vit_large
from src.models.jepa_predictor import vit_predictor
from src.models.temporal_adapter import IdentityTemporalAdapter, SambaTemporalAdapter


def build_video_encoder(cfg):
    model_cfg = cfg.get("model", {})
    embed_dim = model_cfg.get("pred_embed_dim", 384)
    return vit_large(embed_dim=embed_dim)


def build_jepa_predictor(cfg):
    model_cfg = cfg.get("model", {})
    embed_dim = model_cfg.get("pred_embed_dim", 384)
    depth = model_cfg.get("pred_depth", 2)
    return vit_predictor(embed_dim=embed_dim, depth=depth)


def init_video_model(cfg):
    encoder = build_video_encoder(cfg)
    predictor = build_jepa_predictor(cfg)
    model_cfg = cfg.get("model", {})
    if model_cfg.get("temporal_adapter", "identity") == "samba":
        temporal_adapter = SambaTemporalAdapter(
            dim=model_cfg.get("pred_embed_dim", 384),
            depth=model_cfg.get("temporal_depth", 2),
            block_cfg=model_cfg.get("temporal_block_cfg", {}),
        )
    else:
        temporal_adapter = IdentityTemporalAdapter()
    return encoder, predictor, temporal_adapter
