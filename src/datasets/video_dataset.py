
class DummyTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


def build_dummy_video_batch(batch_size, num_frames, image_size, patch_size, embed_dim):
    clips = DummyTensor((batch_size, 3, num_frames, image_size, image_size))
    num_tokens = num_frames * (image_size // patch_size) * (image_size // patch_size)
    masks_enc = DummyTensor((batch_size, num_tokens))
    masks_pred = DummyTensor((batch_size, num_tokens))
    return clips, masks_enc, masks_pred
