"""Data pipeline: BPE round-trip + compression, text packing, image/audio front ends."""

import torch

from tessera.data import (
    ByteBPETokenizer,
    MultimodalEmbedder,
    PackedTextDataset,
    PatchEmbed,
    log_mel_spectrogram,
    mel_filterbank,
    patchify,
    stream_windows,
)

# -- tokenizer ---------------------------------------------------------------
_CORPUS = [
    "the quick brown fox jumps over the lazy dog. " * 4,
    "knowledge distillation distills knowledge into a smaller student. " * 4,
    "tessera tiles attention into blocks for speed. " * 4,
]


def test_bpe_roundtrip_is_lossless():
    tok = ByteBPETokenizer().train(_CORPUS, vocab_size=400)
    for text in ["hello world", "tessera 蒸馏 🚀", "the lazy dog"]:
        assert tok.decode(tok.encode(text)) == text


def test_bpe_actually_compresses():
    tok = ByteBPETokenizer().train(_CORPUS, vocab_size=400)
    text = _CORPUS[0]
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tok.encode(text))
    assert n_tokens < n_bytes  # merges should shorten frequent substrings
    assert tok.vocab_size <= 400


def test_bpe_save_load(tmp_path):
    tok = ByteBPETokenizer().train(_CORPUS, vocab_size=350)
    tok.add_special("<eos>")
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = ByteBPETokenizer.load(path)
    s = "the quick brown fox"
    assert tok2.encode(s) == tok.encode(s)


# -- text packing ------------------------------------------------------------
def test_packed_dataset_shapes_and_shift():
    toks = [list(range(1, 21)), list(range(21, 41))]
    ds = PackedTextDataset(toks, seq_len=8, eos_id=0)
    assert len(ds) >= 1
    x, y = ds[0]
    assert x.shape == (8,) and y.shape == (8,)
    # target is input shifted by one position in the packed stream
    assert torch.equal(x[1:], y[:-1])


def test_stream_windows_yields_full_windows():
    windows = list(stream_windows(range(100), seq_len=10))
    assert all(x.shape == (10,) and y.shape == (10,) for x, y in windows)
    assert len(windows) == 100 // 11  # needs seq_len+1 tokens per window


# -- image -------------------------------------------------------------------
def test_patchify_and_embed():
    img = torch.randn(2, 3, 32, 32)
    patches = patchify(img, patch=16)
    assert patches.shape == (2, 4, 3 * 16 * 16)  # (32/16)^2 = 4 patches
    embed = PatchEmbed(in_channels=3, patch=16, dim=64)
    out = embed(img)
    assert out.shape == (2, 4, 64)


# -- audio -------------------------------------------------------------------
def test_mel_filterbank_properties():
    fb = mel_filterbank(n_mels=20, n_fft=400, sample_rate=16000)
    assert fb.shape == (20, 201)
    assert (fb >= 0).all()
    assert fb.max() <= 1.0 + 1e-6


def test_log_mel_spectrogram_shape():
    wav = torch.randn(16000)  # 1 second @ 16 kHz
    mel = log_mel_spectrogram(wav, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)
    assert mel.shape[1] == 80
    assert mel.shape[0] > 1  # several frames


# -- fusion ------------------------------------------------------------------
def test_multimodal_embedder_concatenates_modalities():
    emb = MultimodalEmbedder(vocab_size=256, dim=48, in_channels=3, patch=16, n_mels=80)
    text_ids = torch.randint(0, 256, (1, 5))
    image = torch.randn(1, 3, 32, 32)  # 4 patches
    log_mel = log_mel_spectrogram(torch.randn(8000), n_mels=80)  # ~ frames
    out = emb(text_ids=text_ids, image=image, log_mel=log_mel)
    expected_len = 5 + 4 + log_mel.shape[0]
    assert out.shape == (1, expected_len, 48)
