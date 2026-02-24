import torch
from ugbio_srsnv.deep_srsnv.cnn_model import CNNReadClassifier


def test_cnn_forward_shape() -> None:
    model = CNNReadClassifier(base_vocab_size=8, t0_vocab_size=12, numeric_channels=12)
    batch_size = 4
    length = 300
    out = model(
        read_base_idx=torch.randint(0, 8, (batch_size, length)),
        ref_base_idx=torch.randint(0, 8, (batch_size, length)),
        t0_idx=torch.randint(0, 12, (batch_size, length)),
        x_num=torch.randn(batch_size, 12, length),
        mask=torch.ones(batch_size, length),
    )
    assert out.shape == (batch_size,)
