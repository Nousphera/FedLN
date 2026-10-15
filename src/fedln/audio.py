"""Speech Commands log-Mel frontend, implemented entirely in PyTorch."""

from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import functional as F


@lru_cache(maxsize=8)
@torch.inference_mode(False)
def mel_filterbank(*, device=None, dtype=torch.float32) -> Tensor:
    """TensorFlow-compatible HTK triangles, with the DC bin excluded."""
    frequencies = torch.linspace(0., 8000., 513, device=device, dtype=dtype)
    to_mel = lambda hz: 1127. * torch.log(1. + hz / 700.)
    edges = torch.linspace(to_mel(frequencies.new_tensor(60.)).item(),
                           to_mel(frequencies.new_tensor(7800.)).item(),
                           66, device=device, dtype=dtype)
    frequencies = to_mel(frequencies).unsqueeze(1)
    lower = (frequencies - edges[:-2]) / (edges[1:-1] - edges[:-2])
    upper = (edges[2:] - frequencies) / (edges[2:] - edges[1:-1])
    weights = torch.minimum(lower, upper).clamp_min(0.)
    weights[0] = 0.
    return weights


def audio_spectrogram(waveform: Tensor, training: bool = False) -> Tensor:
    """Normalized mono 16-kHz PCM to [1, time, 64] log-Mel magnitudes."""
    if waveform.ndim != 1 or not waveform.numel():
        raise ValueError("Expected a nonempty mono waveform")
    if not waveform.is_floating_point():
        raise ValueError("Waveform must contain floating-point PCM samples")
    waveform = waveform.to(torch.float32)
    if not torch.isfinite(waveform).all():
        raise ValueError("Waveform contains nonfinite samples")
    padding = max(16000 - waveform.numel(), 0)
    waveform = F.pad(waveform, (padding // 2, padding - padding // 2))
    waveform = waveform / waveform.square().sum().clamp_min(1e-9).sqrt()
    if training:
        start = int(torch.randint(waveform.numel() - 15690 + 1, ()).item())
        waveform = waveform[start:start + 15690]
    frames = waveform.unfold(0, 400, 160)
    window = torch.hann_window(400, periodic=True, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.fft.rfft(frames * window, n=1024).abs()
    mel = spectrum @ mel_filterbank(device=waveform.device, dtype=waveform.dtype)
    return mel.clamp(1e-5, 1e8).log().unsqueeze(0)
