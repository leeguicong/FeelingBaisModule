
class DummyTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __mul__(self, other):
        return DummyTensor(self.shape)


class VideoEncoder:
    def __init__(self, embed_dim=384):
        self.embed_dim = embed_dim

    def __call__(self, clips, masks_enc=None):
        b, _, t, h, w = clips.shape
        token_count = t * (h // 16) * (w // 16)
        x = DummyTensor((b, token_count, self.embed_dim))
        if masks_enc is not None:
            x = x * masks_enc
        return x


def vit_large(embed_dim=384):
    return VideoEncoder(embed_dim=embed_dim)
