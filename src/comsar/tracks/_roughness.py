"""comsar/tracks/_roughness.py -- Wavelet-based roughness analysis.

Ported from the "Wavelet" C# application (R. Bader): instead of an FFT, the
spectrum of each analysis frame is estimated with a Gaussian-windowed
single-frequency correlation (a Gabor / Morlet wavelet), evaluated on a fine,
*continuous* frequency grid. Spectral peaks are located with sub-grid
(parabolic) precision, which yields far more exact partial frequencies than
FFT bin spacing.

From the resulting per-frame list of partials (frequency, amplitude) two
roughness measures are computed:

* **Helmholtz-Bader** -- pair contribution peaks at a beating distance of
  ``HELMHOLTZ_MAX`` Hz and vanishes beyond ``HELMHOLTZ_LIMIT`` Hz.
* **Sethares** -- the Plomp-Levelt / Sethares sensory-dissonance curve.

``WaveletRoughness.extract`` returns a :class:`RoughnessResult` holding both a
feature table (one roughness value per frame, indexed by ``time_s``) and the
partials as a long-format table ``[time_s, frequency, amplitude]`` (a variable
number of rows per frame).

Windowing is specified in time units (milliseconds + overlap fraction) exactly
like :class:`comsar.TimbreTrack`, so any sample rate works and files of equal
duration produce the same number of frames.
"""
from datetime import datetime
from timeit import default_timer as timer
from typing import Optional

import numpy as np
import pandas as pd

from apollon.audio import AudioFile

import comsar
from .utilities import TrackMeta, SourceMeta


# --- default parameters ------------------------------------------------------
WINDOW_MS_DEFAULT = 370.0     # analysis window length in milliseconds
OVERLAP_DEFAULT = 0.5         # overlap as a fraction of the window (0 < o < 1)
FMIN_DEFAULT = 50.0           # lowest analysed frequency (Hz)
FMAX_DEFAULT = 5000.0         # highest analysed frequency (Hz)
THRESHOLD_DEFAULT = 0.05      # keep partials with amplitude >= 5% of the peak
FREQ_STEP_DEFAULT = 2.0       # frequency grid spacing before peak refinement

# roughness model constants (from the C# implementation)
HELMHOLTZ_LIMIT = 200.0       # Hz, ignore wider pairs
HELMHOLTZ_MAX = 33.0          # Hz, distance of maximum roughness
SETHARES_LIMIT = 1000.0       # Hz, ignore wider pairs

_FREQ_CHUNK = 512             # frequencies processed per matmul (bounds memory)


def gauss_window(n: int) -> np.ndarray:
    """Gaussian analysis window of length ``n`` (as in the C# code)."""
    i = np.arange(n, dtype=np.float64)
    k = (n / 2.0 - i) ** 2 / (n ** 2 / 10.0)
    return np.exp(-k)


def helmholtz_bader_roughness(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Helmholtz-Bader roughness of a set of partials.

    ``R = sum_{i<j, d<LIMIT} A_i A_j * d * exp(-d/MAX) / (MAX * e^-1)`` with
    ``d = |f_i - f_j|``. The pair term is maximal at ``d = HELMHOLTZ_MAX``.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    amps = np.asarray(amps, dtype=np.float64)
    if freqs.size < 2:
        return 0.0
    iu, ju = np.triu_indices(freqs.size, k=1)
    d = np.abs(freqs[iu] - freqs[ju])
    w = amps[iu] * amps[ju]
    m = d < HELMHOLTZ_LIMIT
    d, w = d[m], w[m]
    contrib = w * d * np.exp(-d / HELMHOLTZ_MAX) / (HELMHOLTZ_MAX * np.exp(-1.0))
    return float(contrib.sum())


def sethares_roughness(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Sethares / Plomp-Levelt sensory-dissonance roughness of a set of partials.

    ``R = sum_{i<j, d<LIMIT} A_i A_j * (exp(-3.5 s d) - exp(-5.75 s d))`` with
    ``d = |f_i - f_j|`` and ``s = 0.24 / (0.021 * min(f_i, f_j) + 19)``.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    amps = np.asarray(amps, dtype=np.float64)
    if freqs.size < 2:
        return 0.0
    iu, ju = np.triu_indices(freqs.size, k=1)
    d = np.abs(freqs[iu] - freqs[ju])
    w = amps[iu] * amps[ju]
    f_low = np.minimum(freqs[iu], freqs[ju])
    m = d < SETHARES_LIMIT
    d, w, f_low = d[m], w[m], f_low[m]
    s = 0.24 / (0.021 * f_low + 19.0)
    contrib = w * (np.exp(-3.5 * s * d) - np.exp(-5.75 * s * d))
    return float(contrib.sum())


class RoughnessResult:
    """Result of :meth:`WaveletRoughness.extract`.

    Attributes:
        features:  ``pandas.DataFrame`` indexed by ``time_s`` with the columns
                   ``RoughnessHelmholtzBader`` and ``RoughnessSethares``.
        partials:  long-format ``pandas.DataFrame`` with columns
                   ``[time_s, frequency, amplitude]`` -- one row per detected
                   partial, so the number of rows per frame varies.
    """
    def __init__(self, meta: TrackMeta, params: dict,
                 features: pd.DataFrame, partials: pd.DataFrame) -> None:
        self._meta = meta
        self.params = params
        self.features = features
        self.partials = partials

    def partials_by_frame(self) -> "list[pd.DataFrame]":
        """Return the partials grouped into one DataFrame per analysis frame."""
        groups = dict(tuple(self.partials.groupby("time_s")))
        return [groups.get(t, self.partials.iloc[:0]) for t in self.features.index]

    def to_csv(self, path) -> None:
        """Write the roughness feature table to ``path`` (CSV)."""
        self.features.to_csv(path)

    def partials_to_csv(self, path) -> None:
        """Write the long-format partials table to ``path`` (CSV)."""
        self.partials.to_csv(path, index=False)

    def to_dict(self) -> dict:
        return {"meta": self._meta.to_dict(), "params": self.params,
                "features": self.features.to_dict(orient="list"),
                "partials": self.partials.to_dict(orient="list")}


class WaveletRoughness:
    """Wavelet (Gabor) roughness analysis.

    Args:
        window_ms:    Analysis window length in milliseconds.
        overlap:      Overlap between consecutive windows as a fraction of the
                      window length (0 < overlap < 1).
        f_min:        Lowest analysed frequency in Hz.
        f_max:        Highest analysed frequency in Hz (clipped to Nyquist).
        threshold:    Keep only partials whose amplitude is at least this
                      fraction (0 < t < 1) of the strongest partial in the
                      whole file. Fewer/larger -> fewer partials. This is the
                      parameter that controls how many frequencies are found.
        freq_step:    Spacing of the frequency analysis grid in Hz before
                      parabolic peak refinement (smaller = finer/slower).
    """
    def __init__(self,
                 window_ms: float = WINDOW_MS_DEFAULT,
                 overlap: float = OVERLAP_DEFAULT,
                 f_min: float = FMIN_DEFAULT,
                 f_max: float = FMAX_DEFAULT,
                 threshold: float = THRESHOLD_DEFAULT,
                 freq_step: float = FREQ_STEP_DEFAULT) -> None:
        if not 0.0 < float(overlap) < 1.0:
            raise ValueError('``overlap`` must be a fraction with '
                             f'0 < overlap < 1, got {overlap}.')
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError('``threshold`` must be a fraction with '
                             f'0 < threshold < 1, got {threshold}.')
        self.window_ms = float(window_ms)
        self.overlap = float(overlap)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.threshold = float(threshold)
        self.freq_step = float(freq_step)
        self.feature_names = ('RoughnessHelmholtzBader', 'RoughnessSethares')
        self.verbose = False

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def _frames(self, signal: np.ndarray, fps: int):
        """Split ``signal`` into (window, hop) frames; return (frames, n_frames).

        ``frames`` has shape (n_perseg, n_frames). The frame count matches
        ``TimbreTrack`` (ceil(duration_ms / hop_ms)); frames are zero-padded at
        the signal borders.
        """
        n_perseg = max(2, int(round(self.window_ms * fps / 1000.0)))
        hop = max(1, n_perseg - int(round(n_perseg * self.overlap)))
        duration_ms = signal.shape[0] / fps * 1000.0
        hop_ms = self.window_ms * (1.0 - self.overlap)
        n_frames = int(np.ceil(duration_ms / hop_ms - 1e-6))
        # centre the first frame on t=0 by padding half a window at the front
        pad = n_perseg // 2
        padded = np.concatenate([np.zeros(pad), signal,
                                 np.zeros(n_perseg + hop * n_frames)])
        frames = np.empty((n_perseg, n_frames))
        for k in range(n_frames):
            frames[:, k] = padded[k * hop: k * hop + n_perseg]
        return frames, n_perseg, hop

    def _gabor_amplitudes(self, frames: np.ndarray, fps: int, freqs: np.ndarray):
        """Gaussian-windowed single-frequency amplitudes for all frames.

        Returns an array of shape (n_freqs, n_frames).
        """
        n_perseg = frames.shape[0]
        w = gauss_window(n_perseg)
        wframes = frames * w[:, None]                 # (n_perseg, n_frames)
        t = np.arange(n_perseg, dtype=np.float64) / fps
        amp = np.empty((freqs.size, frames.shape[1]))
        two_pi = 2.0 * np.pi
        for s in range(0, freqs.size, _FREQ_CHUNK):
            fchunk = freqs[s:s + _FREQ_CHUNK]
            ang = two_pi * fchunk[:, None] * t[None, :]     # (chunk, n_perseg)
            cos_b = np.cos(ang).astype(np.float32)
            sin_b = np.sin(ang).astype(np.float32)
            c = cos_b @ wframes.astype(np.float32)          # (chunk, n_frames)
            sn = sin_b @ wframes.astype(np.float32)
            amp[s:s + fchunk.size] = np.sqrt(c * c + sn * sn) / n_perseg
        return amp

    @staticmethod
    def _peaks(col: np.ndarray, freqs: np.ndarray, thr: float):
        """Find local maxima > ``thr`` in ``col`` and refine them parabolically.

        Returns (peak_freqs, peak_amps).
        """
        if col.size < 3:
            return np.empty(0), np.empty(0)
        left, mid, right = col[:-2], col[1:-1], col[2:]
        idx = np.where((mid > left) & (mid >= right) & (mid > thr))[0] + 1
        if idx.size == 0:
            return np.empty(0), np.empty(0)
        a0, a1, a2 = col[idx - 1], col[idx], col[idx + 1]
        denom = (a0 - 2 * a1 + a2)
        delta = np.where(denom != 0, 0.5 * (a0 - a2) / denom, 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        step = freqs[1] - freqs[0]
        pfreq = freqs[idx] + delta * step
        pamp = a1 - 0.25 * (a0 - a2) * delta
        return pfreq, pamp

    def extract(self, path) -> RoughnessResult:
        """Run the wavelet roughness analysis on the audio file ``path``.

        Any sample rate is accepted. Returns a :class:`RoughnessResult`.
        """
        snd = AudioFile(path)
        fps = snd.fps
        data = snd.data.squeeze()
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = np.asarray(data, dtype=np.float64)

        f_max = min(self.f_max, fps / 2.0 - 1.0)
        freqs = np.arange(self.f_min, f_max + self.freq_step, self.freq_step)

        frames, n_perseg, hop = self._frames(data, fps)
        n_frames = frames.shape[1]

        if self.verbose:
            print(f'wavelet roughness: {n_frames} frames x {freqs.size} freqs '
                  f'({self.f_min:.0f}-{f_max:.0f} Hz) ...', end=' ')
        pace = timer()
        amp = self._gabor_amplitudes(frames, fps, freqs)      # (n_freqs, n_frames)
        ref = amp.max() if amp.size else 1.0
        if ref <= 0:
            ref = 1.0
        amp_norm = np.clip(amp / ref, 0.0, 1.0)

        hop_s = hop / fps
        rgh = np.zeros((n_frames, 2))
        p_time, p_freq, p_amp = [], [], []
        for k in range(n_frames):
            pf, pa = self._peaks(amp_norm[:, k], freqs, self.threshold)
            if pf.size:
                rgh[k, 0] = helmholtz_bader_roughness(pf, pa)
                rgh[k, 1] = sethares_roughness(pf, pa)
                t = round(k * hop_s, 6)
                p_time.append(np.full(pf.size, t))
                p_freq.append(pf)
                p_amp.append(pa)
        if self.verbose:
            print(f'{timer() - pace:.4} s.')

        index = pd.Index(np.round(np.arange(n_frames) * hop_s, 6), name='time_s')
        features = pd.DataFrame(rgh, columns=self.feature_names, index=index)
        if p_time:
            partials = pd.DataFrame({
                'time_s': np.concatenate(p_time),
                'frequency': np.concatenate(p_freq),
                'amplitude': np.concatenate(p_amp)})
        else:
            partials = pd.DataFrame(columns=['time_s', 'frequency', 'amplitude'])

        file_meta = SourceMeta(*snd.file_name.split('.'), snd.hash)
        meta = TrackMeta(comsar.__version__, datetime.utcnow(), file_meta)
        params = {'window_ms': self.window_ms, 'overlap': self.overlap,
                  'f_min': self.f_min, 'f_max': f_max,
                  'threshold': self.threshold, 'freq_step': self.freq_step}
        snd.close()
        return RoughnessResult(meta, params, features, partials)
