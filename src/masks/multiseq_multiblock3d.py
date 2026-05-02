from src.datasets.video_dataset import DummyTensor


def build_masks(batch_size: int, num_tokens: int):
    return DummyTensor((batch_size, num_tokens)), DummyTensor((batch_size, num_tokens))
