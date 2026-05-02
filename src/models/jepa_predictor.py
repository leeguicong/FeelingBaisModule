from src.models.video_encoder import DummyTensor


class VITPredictor:
    def __init__(self, embed_dim=384, depth=2):
        self.embed_dim = embed_dim
        self.depth = depth

    def __call__(self, z, masks_enc=None, masks_pred=None):
        return DummyTensor(z.shape)


def vit_predictor(embed_dim=384, depth=2):
    return VITPredictor(embed_dim=embed_dim, depth=depth)
