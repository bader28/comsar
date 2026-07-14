"""comsar/tracks/_impulse.py -- Impulse Pattern extraction from a pitch track.

The impulse pattern turns the waveform into a sequence of impulses, one per
pitch period. For a detected fundamental frequency ``f0`` the period
``T = 1 / f0`` defines one impulse. The onset of an impulse is

  a) a **rising** zero crossing of the waveform (negative -> positive), and
  b) the zero crossing **before the strongest amplitude** within ``T``.

Only parts of the waveform whose level is above ``dbmin`` (SPL, default 52 dB)
are analysed; silence resets the chain. Starting from the first impulse after
silence, the next impulse is placed one period ahead, where the local pitch
gives the new ``T``. Sustained tones therefore yield a fairly regular impulse
pattern, while transients / noise give an irregular one. Each impulse carries
the maximum ``|amplitude|`` between it and the following impulse.

``ImpulsePattern.extract(wav_path, pitch_result)`` returns an
:class:`ImpulsePatternResult` whose ``.impulses`` is a list-like DataFrame with
columns ``[time_s, amplitude]`` -- one row per impulse, analogous to the pitch
track.
"""
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from apollon.audio import AudioFile

import comsar
from .utilities import TrackMeta, SourceMeta


DBMIN_DEFAULT = 52.0     # SPL threshold in dB; only louder parts are analysed


class ImpulsePatternResult:
    """Result of :meth:`ImpulsePattern.extract`.

    Attributes:
        impulses:  ``pandas.DataFrame`` with columns ``[time_s, amplitude]`` --
                   the onset time (seconds) and peak amplitude of each impulse.
    """
    def __init__(self, meta: TrackMeta, params: dict,
                 impulses: pd.DataFrame) -> None:
        self._meta = meta
        self.params = params
        self.impulses = impulses

    def to_csv(self, path) -> None:
        """Write the impulse list to ``path`` (CSV)."""
        self.impulses.to_csv(path, index=False)

    def to_dict(self) -> dict:
        return {"meta": self._meta.to_dict(), "params": self.params,
                "impulses": self.impulses.to_dict(orient="list")}


class ImpulsePattern:
    """Impulse-pattern analysis of a waveform, driven by a pitch track.

    Args:
        dbmin:  SPL threshold in dB. Waveform parts whose (interpolated) SPL is
                below this are treated as silence and skipped. Default 52 dB.
    """
    def __init__(self, dbmin: float = DBMIN_DEFAULT) -> None:
        self.dbmin = float(dbmin)
        self.verbose = False

    def extract(self, wav_path, pitch_result) -> ImpulsePatternResult:
        """Extract the impulse pattern from ``wav_path`` using ``pitch_result``.

        Args:
            wav_path:      Path to the audio file (any sample rate).
            pitch_result:  A :class:`comsar.PitchTrack` result (uses its
                           ``.features`` with ``Pitch`` and ``SPL`` columns) or
                           a DataFrame indexed by ``time_s`` with those columns.

        Returns:
            :class:`ImpulsePatternResult`.
        """
        snd = AudioFile(wav_path)
        fps = snd.fps
        file_name = snd.file_name
        file_hash = snd.hash
        x = snd.data.squeeze()
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = np.asarray(x, dtype=np.float64)
        snd.close()

        feats = pitch_result.features if hasattr(pitch_result, "features") \
            else pitch_result
        frame_t = np.asarray(feats.index, dtype=float)
        f0 = feats["Pitch"].to_numpy(dtype=float)
        spl = feats["SPL"].to_numpy(dtype=float) if "SPL" in feats.columns \
            else np.full(frame_t.size, np.inf)

        n_samples = x.size
        sample_t = np.arange(n_samples) / fps
        f0_s = np.interp(sample_t, frame_t, f0)          # f0 per sample
        spl_s = np.interp(sample_t, frame_t, spl)        # SPL per sample
        active = (spl_s >= self.dbmin) & (f0_s > 0.0)
        ax = np.abs(x)

        # all rising zero crossings (x[j] <= 0 < x[j+1]) as sample indices j
        rising = np.flatnonzero((x[:-1] <= 0.0) & (x[1:] > 0.0))

        if self.verbose:
            print("impulse pattern ...", end=" ")
        pace = timer()

        imp = []
        n = 0
        while n < n_samples - 1:
            if not active[n]:
                # jump to the next active sample
                nxt = np.argmax(active[n:])
                if active[n + nxt]:
                    n = n + nxt
                    continue
                break
            f0n = f0_s[n]
            t_len = int(round(fps / f0n)) if f0n > 0 else 0
            if t_len < 2:
                n += 1
                continue
            win_end = min(n_samples, n + t_len)
            peak = n + int(np.argmax(ax[n:win_end]))
            # rising zero crossing before the peak, not earlier than n
            k = np.searchsorted(rising, peak, side="right") - 1
            t_imp = int(rising[k]) if k >= 0 and rising[k] >= n else n
            imp.append(t_imp)
            nxt = t_imp + t_len
            n = nxt if nxt > n else n + 1

        imp = np.asarray(imp, dtype=int)
        # amplitude = max |x| between this impulse and the next
        amps = np.zeros(imp.size)
        for i in range(imp.size):
            a = imp[i]
            if i + 1 < imp.size:
                b = imp[i + 1]
            else:
                t_len = int(round(fps / max(f0_s[a], 1e-9)))
                b = min(n_samples, a + max(t_len, 1))
            if b > a:
                amps[i] = ax[a:b].max()

        if self.verbose:
            print(f"{timer() - pace:.4} s. ({imp.size} impulses)")

        impulses = pd.DataFrame({"time_s": np.round(imp / fps, 6),
                                 "amplitude": amps})
        meta = TrackMeta(comsar.__version__, datetime.utcnow(),
                         SourceMeta(*file_name.split("."), file_hash))
        params = {"dbmin": self.dbmin}
        return ImpulsePatternResult(meta, params, impulses)
