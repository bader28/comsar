"""comsar/tracks/timbre.py -- TimbreTack implementation
License: BSD-3-Clasuse
Copyright (C) 2020, Michael Blaß, michael.blass@uni-hamburg.de
"""
from datetime import datetime
from timeit import default_timer as timer
from typing import Optional

import numpy as np
import pandas as pd

from apollon.audio import AudioFile
from apollon.segment import Segmentation
from apollon.signal import container, features
from apollon.signal.spectral import StftSegments

import comsar
from . utilities import (TrackMeta, TrackResult, SourceMeta,
                         TimbreTrackParams, TimbreTrackCorrGramParams)

# Legacy sample-based default (2**15 samples at 44.1 kHz = the historical
# behaviour). Only used when an explicit ``stft_params`` object is passed.
STFT_DEFAULT = container.StftParams(fps=44100, window='hamming', n_fft=None,
                                    n_perseg=2**15, n_overlap=2**14,
                                    extend=True, pad=True)

CORR_DIM_DEFAULT = container.CorrDimParams(delay=14, m_dim=80, n_bins=1000,
                                           scaling_size=10)

CORR_GRAM_DEFAULT = container.CorrGramParams(wlen=2**10, n_delay=2**8,
                                             total=True)

# Time-based defaults: equivalent to the historical 2**15-sample window with
# 50% overlap at 44.1 kHz, but expressed in milliseconds so that any sample
# rate yields the same time resolution.
WINDOW_MS_DEFAULT = 743.0     # ~ 2**15 samples at 44.1 kHz
OVERLAP_DEFAULT = 0.5

# Upper frequency limit of the roughness estimate. Clipped to the Nyquist
# frequency of the analysed file, so low sample rates work, too.
ROUGHNESS_FRQ_MAX = 15000.0


class TimbreTrack:
    """High-level interface for timbre feature extraction.

    The analysis windowing is specified in *time* units (milliseconds and an
    overlap fraction), not in samples. The window and hop sizes are converted
    to samples using the sample rate of each analysed file, so audio files
    with *different sample rates* are directly comparable: equal durations
    produce (up to rounding) equal numbers of analysis frames, and the
    returned feature table carries the frame time in seconds as its index.
    """
    def __init__(self,
                 window_ms: float = WINDOW_MS_DEFAULT,
                 overlap: float = OVERLAP_DEFAULT,
                 corr_dim_params: Optional[container.CorrDimParams] = None,
                 stft_params: Optional[container.StftParams] = None,
                 window: str = 'hamming',
                 n_fft: Optional[int] = None,
                 wavelet_roughness: bool = True,
                 roughness_params: Optional[dict] = None,
                 ) -> None:
        """
        Args:
            window_ms:         Length of the analysis window in milliseconds.
            overlap:           Overlap between consecutive windows as a
                               fraction of the window length (0 < overlap < 1).
            corr_dim_params:   Parameter set for correlation dimension.
            stft_params:       Legacy sample-based parameters. If given,
                               ``window_ms``/``overlap`` are ignored, and the
                               audio file must match ``stft_params.fps``
                               exactly (historical behaviour).
            window:            Window function for the STFT.
            n_fft:             FFT length; ``None`` uses the window size.
            wavelet_roughness: If ``True`` (default), two extra columns
                               ``RoughnessHelmholtzBader`` and
                               ``RoughnessSethares`` from the wavelet
                               :class:`WaveletRoughness` analysis are appended
                               to the feature table.
            roughness_params:  Optional dict of extra keyword arguments for
                               :class:`WaveletRoughness` (``f_min``, ``f_max``,
                               ``threshold``, ``freq_step``); ``window_ms`` and
                               ``overlap`` are taken from this track.
        """
        if stft_params is None and not 0.0 < float(overlap) < 1.0:
            raise ValueError('``overlap`` must be a fraction with '
                             f'0 < overlap < 1, got {overlap}.')

        self.window_ms = float(window_ms)
        self.overlap = float(overlap)
        self.window = window
        self.n_fft = n_fft
        self.corr_dim = corr_dim_params or CORR_DIM_DEFAULT
        self._fixed = stft_params      # legacy mode if not None
        self.wavelet_roughness = wavelet_roughness
        self.roughness_params = dict(roughness_params or {})

        if self._fixed is not None:
            self.params = TimbreTrackParams(self._fixed, self.corr_dim)
        else:
            self.params = None         # set per file in ``extract``

        self.feature_names = ('SpectralCentroid', 'SpectralSpread',
                              'SpectralFlux', 'Roughness', 'Sharpness',
                              'SPL', 'CorrelationDimension')

        self.funcs = [features.spectral_centroid,
                      features.spectral_spread,
                      features.spectral_flux,
                      features.roughness_helmholtz,
                      features.sharpness,
                      features.spl,
                      features.cdim]

        self.pace = np.zeros(self.n_features)
        self.verbose = False

    @property
    def n_features(self) -> int:
        """Number of features.

        Returns:
            Number of audio features.
        """
        return len(self.feature_names)

    def _window_params(self, fps: int) -> container.StftParams:
        """Convert the time-based window setup to samples for rate ``fps``."""
        n_perseg = max(2, int(round(self.window_ms * fps / 1000.0)))
        n_overlap = int(round(n_perseg * self.overlap))
        n_overlap = min(max(n_overlap, 1), n_perseg - 1)
        return container.StftParams(fps=fps, window=self.window,
                                    n_fft=self.n_fft,
                                    n_perseg=n_perseg, n_overlap=n_overlap,
                                    extend=True, pad=True)

    def extract(self, path) -> pd.DataFrame:
        """Run TimbreTrack on audio file.

        Any sample rate is accepted; the analysis windows are computed from
        ``window_ms``/``overlap`` for the file's own sample rate. The index of
        the returned feature table is the frame time in seconds (``time_s``).

        Args:
            path:   Path to audio file.

        Returns:
           Extracted features.
        """
        snd = AudioFile(path)
        fps = snd.fps

        if self._fixed is not None:
            if fps != self._fixed.fps:
                snd.close()
                raise ValueError(f'Sample rate of {snd!s} differs from init.')
            stft_params = self._fixed
        else:
            stft_params = self._window_params(fps)
        self.params = TimbreTrackParams(stft_params, self.corr_dim)

        cutter = Segmentation(stft_params.n_perseg, stft_params.n_overlap,
                              stft_params.extend, stft_params.pad)
        stft = StftSegments(fps, stft_params.window, stft_params.n_fft)

        segs = cutter.transform(snd.data.squeeze())
        sxx = stft.transform(segs)

        # clip the roughness limit to the Nyquist frequency, so files with
        # sample rates below 2 * ROUGHNESS_FRQ_MAX are analysable
        frq_max = min(ROUGHNESS_FRQ_MAX, fps / 2.0)

        args = [(sxx.frqs, sxx.power),
                (sxx.frqs, sxx.power),
                (sxx.abs,),
                (sxx.d_frq, sxx.abs, frq_max),
                (sxx.frqs, sxx.abs),
                (segs.data,),
                (segs.data,)]

        kwargs = [{}, {}, {}, {}, {}, {}, self.corr_dim.to_dict()]

        out = np.zeros((segs.n_segs, self.n_features))
        for i, (fun, arg, kwarg) in enumerate(zip(self.funcs, args, kwargs)):
            out[:, i] = self._worker(i, fun, arg, kwarg)

        file_meta = SourceMeta(*snd.file_name.split('.'), snd.hash)
        track_meta = TrackMeta(comsar.__version__, datetime.utcnow(),
                               file_meta)
        out = pd.DataFrame(data=out, columns=self.feature_names)
        # frame time in seconds: with ``extend=True`` the first frame is
        # centred on t=0, so frame i sits at i * hop
        hop = stft_params.n_perseg - stft_params.n_overlap
        if self._fixed is None:
            # Rounding the hop to whole samples plus border extension/padding
            # can yield an extra frame at some sample rates. Trim to the
            # rate-independent count ceil(duration_ms / hop_ms), so files
            # of equal duration produce the same number of frames at any
            # sample rate.
            duration_ms = snd.data.squeeze().shape[0] / fps * 1000.0
            hop_ms = self.window_ms * (1.0 - self.overlap)
            n_expected = int(np.ceil(duration_ms / hop_ms - 1e-6))
            if len(out) > n_expected:
                out = out.iloc[:n_expected]
        out.index = np.round(np.arange(len(out)) * (hop / fps), 6)
        out.index.name = 'time_s'

        # Optionally append the two wavelet roughness columns (Helmholtz-Bader,
        # Sethares) computed on the same signal with matching windowing.
        if self.wavelet_roughness:
            from ._roughness import WaveletRoughness
            win_ms = self.window_ms if self._fixed is None else \
                stft_params.n_perseg / fps * 1000.0
            ovl = self.overlap if self._fixed is None else \
                stft_params.n_overlap / stft_params.n_perseg
            wr = WaveletRoughness(window_ms=win_ms, overlap=ovl,
                                  **self.roughness_params)
            wr.verbose = self.verbose
            sig = snd.data.squeeze()
            if sig.ndim > 1:
                sig = sig.mean(axis=1)
            rgh, partials, _, _, _ = wr._compute(sig, fps, want_partials=True)
            hb, seth = rgh[:, 0], rgh[:, 1]
            n = len(out)
            hb = np.resize(hb, n) if len(hb) >= n else \
                np.concatenate([hb, np.zeros(n - len(hb))])
            seth = np.resize(seth, n) if len(seth) >= n else \
                np.concatenate([seth, np.zeros(n - len(seth))])
            out['RoughnessHelmholtzBader'] = hb
            out['RoughnessSethares'] = seth

        snd.close()
        result = TrackResult(track_meta, self.params, out)
        # Attach the wavelet partials so the player can show the partial-gram
        # via ``timbre_player(wav, result.features, partials=result.partials)``.
        result.partials = partials if self.wavelet_roughness else None
        return result

    def _worker(self, idx, func, args, kwargs) -> np.ndarray:
        print(self.feature_names[idx], end=' ... ')
        pace = timer()
        res = func(*args, **kwargs)
        pace = timer() - pace
        self.pace[idx] = pace
        print(f'{pace:.4} s.')
        return res


class TimbreTrackCorrGram:
    """Compute timbre track of an audio file.

    Legacy class with fixed, sample-based parameters (the audio file must
    match ``stft_params.fps``).
    """
    def __init__(self,
                 stft_params: Optional[container.StftParams] = None,
                 corr_dim_params: Optional[container.CorrDimParams] = None,
                 corr_gram_params: Optional[container.CorrGramParams] = None) -> None:
        """
        Args:
        """
        self.params = TimbreTrackCorrGramParams(stft_params or STFT_DEFAULT,
                                        corr_dim_params or CORR_DIM_DEFAULT,
                                        corr_gram_params or CORR_GRAM_DEFAULT)

        self.cutter = Segmentation(self.params.stft.n_perseg,
                                   self.params.stft.n_overlap,
                                   self.params.stft.extend,
                                   self.params.stft.pad)
        self.stft = StftSegments(self.params.stft.fps, self.params.stft.window,
                                 self.params.stft.n_fft)

        self.feature_names = ('SpectralCentroid', 'SpectralSpread',
                              'SpectralFlux', 'Roughness', 'Sharpness',
                              'SPL', 'CorrelationDimension', 'Correlogram')

        self.funcs = [features.spectral_centroid,
                      features.spectral_spread,
                      features.spectral_flux,
                      features.roughness_helmholtz,
                      features.sharpness,
                      features.spl,
                      features.cdim,
                      features.correlogram]

        self.pace = np.zeros(self.n_features)
        self.verbose = False

    @property
    def n_features(self) -> int:
        """Number of features on track"""
        return len(self.feature_names)

    def extract(self, path) -> pd.DataFrame:
        """Perform extraction.
        """
        snd = AudioFile(path)
        if snd.fps != self.params.stft.fps:
            snd.close()
            raise ValueError('Sample rate of {snd!str} differs from init.')

        segs = self.cutter.transform(snd.data.squeeze())
        sxx = self.stft.transform(segs)

        args = [(sxx.frqs, sxx.power),
                (sxx.frqs, sxx.power),
                (sxx.abs,),
                (sxx.d_frq, sxx.abs, 15000),
                (sxx.frqs, sxx.abs),
                (segs.data,),
                (segs.data,),
                (segs.data,)]

        kwargs = [{}, {}, {}, {}, {}, {}, self.params.corr_dim.to_dict(),
                  self.params.corr_gram.to_dict()]

        out = np.zeros((segs.n_segs, self.n_features))
        for i, (fun, arg, kwarg) in enumerate(zip(self.funcs, args, kwargs)):
            out[:, i] = self._worker(i, fun, arg, kwarg)
        snd.close()

        meta = TrackMeta(comsar.__version__, datetime.utcnow(), snd.file_name)
        out = pd.DataFrame(data=out, columns=self.feature_names)
        return TrackResult(meta, self.params, out)

    def _worker(self, idx, func, args, kwargs) -> np.ndarray:
        print(self.feature_names[idx], end=' ... ')
        pace = timer()
        res = func(*args, **kwargs)
        pace = timer() - pace
        self.pace[idx] = pace
        print(f'{pace:.4} s.')
        return res
