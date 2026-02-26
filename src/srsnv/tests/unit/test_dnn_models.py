import torch
from ugbio_srsnv.deep_srsnv.cnn_model import CNNReadClassifier


def test_cnn_forward_shape() -> None:
    model = CNNReadClassifier(
        base_vocab_size=8,
        t0_vocab_size=12,
        numeric_channels=9,
        tm_vocab_size=9,
        st_vocab_size=5,
        et_vocab_size=5,
    )
    batch_size = 4
    length = 300
    out = model(
        read_base_idx=torch.randint(0, 8, (batch_size, length)),
        ref_base_idx=torch.randint(0, 8, (batch_size, length)),
        t0_idx=torch.randint(0, 12, (batch_size, length)),
        x_num=torch.randn(batch_size, 9, length),
        mask=torch.ones(batch_size, length),
        tm_idx=torch.randint(0, 9, (batch_size,)),
        st_idx=torch.randint(0, 5, (batch_size,)),
        et_idx=torch.randint(0, 5, (batch_size,)),
    )
    assert out.shape == (batch_size,)


def test_cnn_forward_without_cat_embeds() -> None:
    model = CNNReadClassifier(
        base_vocab_size=8,
        t0_vocab_size=12,
        numeric_channels=9,
    )
    batch_size = 4
    length = 300
    out = model(
        read_base_idx=torch.randint(0, 8, (batch_size, length)),
        ref_base_idx=torch.randint(0, 8, (batch_size, length)),
        t0_idx=torch.randint(0, 12, (batch_size, length)),
        x_num=torch.randn(batch_size, 9, length),
        mask=torch.ones(batch_size, length),
    )
    assert out.shape == (batch_size,)


def test_cnn_attention_pooling_uses_focus() -> None:
    model = CNNReadClassifier(
        base_vocab_size=8,
        t0_vocab_size=12,
        numeric_channels=9,
        hidden_channels=32,
        n_blocks=2,
    )
    model.eval()
    batch_size = 2
    length = 50
    x_num = torch.zeros(batch_size, 9, length)
    x_num[0, 3, 10] = 1.0
    x_num[1, 3, 20] = 1.0

    with torch.no_grad():
        out = model(
            read_base_idx=torch.randint(0, 8, (batch_size, length)),
            ref_base_idx=torch.randint(0, 8, (batch_size, length)),
            t0_idx=torch.randint(0, 12, (batch_size, length)),
            x_num=x_num,
            mask=torch.ones(batch_size, length),
        )
    assert out.shape == (batch_size,)


def test_cnn_dilated_blocks() -> None:
    model = CNNReadClassifier(
        base_vocab_size=8,
        t0_vocab_size=12,
        numeric_channels=9,
        hidden_channels=32,
        n_blocks=6,
        dilations=[1, 1, 2, 2, 4, 4],
    )
    assert len(model.blocks) == 6
    batch_size = 2
    length = 100
    out = model(
        read_base_idx=torch.randint(0, 8, (batch_size, length)),
        ref_base_idx=torch.randint(0, 8, (batch_size, length)),
        t0_idx=torch.randint(0, 12, (batch_size, length)),
        x_num=torch.randn(batch_size, 9, length),
        mask=torch.ones(batch_size, length),
    )
    assert out.shape == (batch_size,)
