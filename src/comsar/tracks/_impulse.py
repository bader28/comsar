"""comsar/tracks/_impulse.py -- Impulse Pattern extraction from a waveform.

The impulse pattern turns the waveform into a sequence of impulses, one per
period. For a local period ``T`` the onset of an impulse is

  a) a **rising** zero crossing of the waveform (negative -> positive), and
  b) the zero crossing **before the strongest amplitude** within ``T``.

Only parts of the waveform whose level is above ``dbmin`` (SPL, default 52 dB)
are analysed; silence resets the chain. Each impulse carries the maximum
``|amplitude|`` between it and the following impulse.

**Local period.** Instead of trusting the (time-gridded, octave-error-prone)
pitch track for ``T``, the period is measured *from the waveform itself* by a
short autocorrelation starting at the previous impulse (``period_source='wave'``,
the default). The pitch f0 only provides a search hint and a fallback for
noisy / aperiodic parts (where a regular period is undefined, giving the
intended complex impulse pattern for transients and noise). This avoids two
failure modes of a pitch-driven period:

* octave errors in f0 make ``T = 1/f0`` too long, so a whole period is skipped;
* interpolating f0 across an unvoiced (f0 = 0) frame yields a tiny f0 and thus
  an enormous ``T`` that jumps across a long, loud passage.

Set ``period_source='pitch'`` to restore the plain ``T = 1 / f0`` behaviour.

``ImpulsePattern.extract(wav_path, pitch_result)`` returns an
:class:`ImpulsePatternResult` whose ``.impulses`` is a list-like DataFrame with
columns ``[time_s, amplitude]`` -- one row per impulse.
"""
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from apollon.audio import AudioFile

import comsar
from .utilities import TrackMeta, SourceMeta


DBMIN_DEFAULT = 52.0      # SPL threshold in dB; only louder parts are analysed
FMIN_DEFAULT = 60.0       # lowest period frequency (Hz) -> longest period
FMAX_DEFAULT = 1200.0     # highest period frequency (Hz) -> shortest period
MIN_PERIODICITY = 0.3     # min normalised autocorrelation to accept a period


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
    """Impulse-pattern analysis of a waveform.

    Args:
        dbmin:          SPL threshold in dB. Waveform parts whose (interpolated)
                        SPL is below this are treated as silence and skipped.
        f_min, f_max:   Frequency bounds (Hz) of the local period search;
                        the period is limited to ``[1/f_max, 1/f_min]``.
        period_source:  ``'wave'`` (default) measures the local period from the
                        waveform by autocorrelation (robust to f0 octave
                        errors); ``'pitch'`` uses ``T = 1 / f0`` from the pitch
                        track (the previous behaviour).
    """
    def __init__(self, dbmin: float = DBMIN_DEFAULT, f_min: float = FMIN_DEFAULT,
                 f_max: float = FMAX_DEFAULT, period_source: str = 'wave') -> None:
        self.dbmin = float(dbmin)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.period_source = period_source
        self.verbose = False

    def _local_period(self, x, t, fps, tmin, tmax, f0h):
        """Local period (in samples) at sample ``t`` via autocorrelation.

        The pitch ``f0h`` (Hz) anchors the search: the autocorrelation peak is
        taken near ``fps / f0h``, then corrected for clear octave errors -- a
        shorter sub-multiple is preferred only if it is nearly as periodic
        (>= 85 %). This locks onto the perceived fundamental period while fixing
        f0 octave errors. Returns the lag, or ``None`` if not clearly periodic.
        """
        n = x.size
        lw = min(n - t, 2 * tmax)
        if lw < tmin + 4:
            return None
        w = x[t:t + lw]
        w = w - w.mean()
        nfft = 1
        while nfft < 2 * lw:
            nfft <<= 1
        f = np.fft.rfft(w, nfft)
        ac = np.fft.irfft(f * np.conj(f))[:tmax + 2]
        if ac[0] <= 0:
            return None
        acn = ac / ac[0]
        if f0h <= 0:
            return None
        # the pitch-derived period, refined to the nearest autocorrelation peak
        t0 = int(min(tmax, max(tmin, round(fps / f0h))))
        a0, a1 = max(tmin, t0 - 3), min(tmax, t0 + 3)
        base = a0 + int(np.argmax(acn[a0:a1 + 1]))
        base_val = acn[base]
        if base_val < MIN_PERIODICITY:
            return None
        # octave-down correction: if f0 was an octave/harmonic too low, a clean
        # sub-multiple of the period is almost as periodic -> take the shortest
        # such period; otherwise keep the f0 period (faithful to T = 1/f0).
        for k in (4, 3, 2):
            cand = int(round(base / k))
            if cand < tmin:
                continue
            c0, c1 = max(tmin, cand - 2), min(tmax, cand + 2)
            cpk = c0 + int(np.argmax(acn[c0:c1 + 1]))
            if acn[cpk] >= 0.85 * base_val:
                return cpk
        return base

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
        f0_s = np.interp(sample_t, frame_t, f0)
        spl_s = np.interp(sample_t, frame_t, spl)
        active = spl_s >= self.dbmin
        ax = np.abs(x)
        rising = np.flatnonzero((x[:-1] <= 0.0) & (x[1:] > 0.0))

        tmin = max(2, int(round(fps / self.f_max)))       # shortest period
        tmax = max(tmin + 2, int(round(fps / self.f_min)))  # longest period
        use_wave = self.period_source == 'wave'

        if self.verbose:
            print("impulse pattern ...", end=" ")
        pace = timer()

        def period_at(t):
            if use_wave:
                lag = self._local_period(x, t, fps, tmin, tmax, f0_s[t])
                if lag is not None:
                    return lag
            # fallback (noise / aperiodic parts, or period_source='pitch'):
            # bounded pitch period
            f0h = f0_s[t]
            if f0h <= 0:
                return tmax
            return int(min(tmax, max(tmin, round(fps / f0h))))

        imp = []
        n = 0
        while n < n_samples - 1:
            if not active[n]:
                nxt = np.argmax(active[n:])
                if active[n + nxt]:
                    n = n + nxt
                    continue
                break
            t_len = period_at(n)
            win_end = min(n_samples, n + t_len)
            if win_end - n < 2:
                n += 1
                continue
            peak = n + int(np.argmax(ax[n:win_end]))
            k = np.searchsorted(rising, peak, side="right") - 1
            t_imp = int(rising[k]) if k >= 0 and rising[k] >= n else n
            imp.append(t_imp)
            # advance by one *measured* period ahead of this impulse
            step = period_at(t_imp)
            nxt = t_imp + step
            n = nxt if nxt > n else n + 1

        imp = np.asarray(imp, dtype=int)
        amps = np.zeros(imp.size)
        for i in range(imp.size):
            a = imp[i]
            if i + 1 < imp.size:
                b = imp[i + 1]
            else:
                b = min(n_samples, a + period_at(a))
            if b > a:
                amps[i] = ax[a:b].max()

        if self.verbose:
            print(f"{timer() - pace:.4} s. ({imp.size} impulses)")

        impulses = pd.DataFrame({"time_s": np.round(imp / fps, 6),
                                 "amplitude": amps})
        meta = TrackMeta(comsar.__version__, datetime.utcnow(),
                         SourceMeta(*file_name.split("."), file_hash))
        params = {"dbmin": self.dbmin, "f_min": self.f_min, "f_max": self.f_max,
                  "period_source": self.period_source}
        return ImpulsePatternResult(meta, params, impulses)
