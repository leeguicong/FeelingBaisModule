from app.vjepa_suspension.utils import init_video_model
from src.datasets.video_dataset import build_dummy_video_batch


class JEPAWithTemporalAdapter:
    def __init__(self, encoder, predictor, temporal_adapter=None):
        self.encoder = encoder
        self.predictor = predictor
        self.temporal_adapter = temporal_adapter

    def forward_context(self, clips, masks_enc, masks_pred):
        z = self.encoder(clips, masks_enc)
        if self.temporal_adapter is not None:
            z = self.temporal_adapter(z, masks_enc=masks_enc)
        out = self.predictor(z, masks_enc, masks_pred)
        return out


def main(cfg):
    encoder, predictor, temporal_adapter = init_video_model(cfg)
    model = JEPAWithTemporalAdapter(encoder, predictor, temporal_adapter)

    clips, masks_enc, masks_pred = build_dummy_video_batch(
        batch_size=2,
        num_frames=cfg.get("data", {}).get("frames", 16),
        image_size=cfg.get("data", {}).get("image_size", 256),
        patch_size=cfg.get("data", {}).get("patch_size", 16),
        embed_dim=cfg.get("model", {}).get("pred_embed_dim", 384),
    )

    out = model.forward_context(clips, masks_enc, masks_pred)

    print("[vjepa_suspension] entered training entrypoint")
    print(f"forward ok: output shape={out.shape}")
