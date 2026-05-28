"""Image + audio front ends that turn pixels/waveforms into token-like sequences.

Distilling a multimodal teacher needs every modality reduced to a sequence of `dim`-vectors
the transformer can consume:
  * images  -> ViT-style non-overlapping patches -> linear projection,
  * audio   -> log-mel spectrogram frames -> linear projection.
The mel filterbank and STFT are implemented in plain torch (no torchaudio dependency).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- image
def patchify(image: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, num_patches, C*patch*patch), row-major over patches."""
    b, c, h, w = image.shape
    if h % patch or w % patch:
        raise ValueError(f"image {h}x{w} not divisible by patch {patch}")
    x = image.unfold(2, patch, patch).unfold(3, patch, patch)  # B,C,H/p,W/p,p,p
    x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, (h // patch) * (w // patch), c * patch * patch)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, patch: int, dim: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(in_channels * patch * patch, dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(patchify(image, self.patch))


# --------------------------------------------------------------------------- audio
def _hz_to_mel(f: float) -> float:
    return 2595.0 * torch.log10(torch.tensor(1.0 + f / 700.0)).item()


def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sample_rate: int, fmin: float = 0.0,
                   fmax: float | None = None) -> torch.Tensor:
    """Triangular mel filterbank, shape (n_mels, n_fft//2 + 1), peaks normalized to 1."""
    fmax = fmax or sample_rate / 2
    n_freqs = n_fft // 2 + 1
    mel_pts = torch.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = torch.floor((n_fft + 1) * hz_pts / sample_rate).long().clamp(max=n_freqs - 1)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        for k in range(left, center):
            if center > left:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb


def log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
) -> torch.Tensor:
    """1-D waveform (T,) -> log-mel features (n_frames, n_mels)."""
    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True
    )
    power = stft.abs().pow(2)  # (n_freqs, n_frames)
    fb = mel_filterbank(n_mels, n_fft, sample_rate).to(waveform.device)
    mel = fb @ power  # (n_mels, n_frames)
    return torch.log(mel.clamp_min(1e-10)).transpose(0, 1)  # (n_frames, n_mels)


class AudioFrontend(nn.Module):
    def __init__(self, n_mels: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(n_mels, dim)

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        return self.proj(log_mel)


# --------------------------------------------------------------------------- fusion
class MultimodalEmbedder(nn.Module):
    """Project text ids, image patches, and audio frames into one (B, seq, dim) stream.

    Each modality gets a learned type embedding so the transformer can tell them apart —
    the standard interleaved-token recipe for multimodal distillation.
    """

    def __init__(self, vocab_size: int, dim: int, in_channels: int = 3,
                 patch: int = 16, n_mels: int = 80):
        super().__init__()
        self.dim = dim
        self.text = nn.Embedding(vocab_size, dim)
        self.image = PatchEmbed(in_channels, patch, dim)
        self.audio = AudioFrontend(n_mels, dim)
        self.type_emb = nn.Embedding(3, dim)  # 0=text, 1=image, 2=audio

    def forward(self, text_ids=None, image=None, log_mel=None) -> torch.Tensor:
        parts = []
        if text_ids is not None:
            parts.append(self.text(text_ids) + self.type_emb.weight[0])
        if image is not None:
            parts.append(self.image(image) + self.type_emb.weight[1])
        if log_mel is not None:
            frames = self.audio(log_mel)
            if frames.dim() == 2:
                frames = frames.unsqueeze(0)
            parts.append(frames + self.type_emb.weight[2])
        if not parts:
            raise ValueError("at least one modality must be provided")
        return torch.cat(parts, dim=1)
