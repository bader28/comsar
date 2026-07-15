"""comsar/tracks/_impulse.py -- Impulse Pattern extraction from a waveform.

The impulse pattern turns the waveform into a sequence of impulses, one per
period. For a local period ``T`` the onset of an impulse is

  a) a **rising** zero crossing of the waveform (negative -> positive), and
  b) the zero crossing **before the strongest amplitude** within ``T``.

Only parts of the waveform whose short-time level is above ``dbmin`` (dBFS,
default -36 dB) are analysed; silence resets the chain. The level is measured
from the waveform itself (25 ms RMS, 0 dBFS = digital full scale), so the gate
is independent of any external SPL scaling. Each impulse carries the maximum
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


DBMIN_DEFAULT = -36.0     # level gate in dBFS; only louder parts are analysed
FMIN_DEFAULT = 60.0       # lowest period frequency (Hz) -> longest period
FMAX_DEFAULT = 1200.0     # highest period frequency (Hz) -> shortest period
MIN_PERIODICITY = 0.3     # min normalised autocorrelation to accept a period


class ImpulsePatternResult:
    """Result of :meth:`ImpulsePattern.extract`.

    Attributes:
        impulses:  ``pandas.DataFrame`` with columns
                   ``[time_s, amplitude, correlation]`` -- the onset time
                   (seconds), the peak amplitude until the next impulse, and the
                   normalised correlation (-1..1) between the waveform period
                   before and after the impulse (~1 for quasi-stationary sounds,
                   low for transients).
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
        dbmin:          Level gate in **dBFS** (0 dBFS = digital full scale, so
                        values are negative). Waveform parts whose short-time
                        RMS level is below this are treated as silence and
                        skipped. Default -36 dB. The level is measured from the
                        waveform directly (assuming normalised float audio, as
                        returned by apollon), independent of any SPL scaling.
        f_min, f_max:   Frequency bounds (Hz) of the local period search;
                        the period is limited to ``[1/f_max, 1/f_min]``.
        period_source:  ``'wave'`` (default) measures the local period from the
                        waveform by autocorrelation (robust to f0 octave
                        errors); ``'pitch'`` uses ``T = 1 / f0`` from the pitch
                        track (the previous behaviour).
        correlation_threshold:  While the normalised correlation of consecutive
                        periods stays **at or above** this value the impulse
                        onset is locked to the current zero-crossing slope
                        (positive or negative); **below** it (transients) the
                        slope may flip to the zero crossing nearest to one
                        period ahead. Default 0.2.
    """
    def __init__(self, dbmin: float = DBMIN_DEFAULT, f_min: float = FMIN_DEFAULT,
                 f_max: float = FMAX_DEFAULT, period_source: str = 'wave',
                 correlation_threshold: float = 0.2) -> None:
        self.dbmin = float(dbmin)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.period_source = period_source
        self.correlation_threshold = float(correlation_threshold)
        self.verbose = False

    def _local_period(self, x, t, fps, tmin, tmax, f0h, prev_lag=None):
        """Local period (in samples) at sample ``t`` via autocorrelation.

        The pitch ``f0h`` (Hz) anchors the search: the autocorrelation peak is
        taken near ``fps / f0h``. Two mechanisms keep the period stable:

        * **Continuity.** If ``prev_lag`` (the period of the preceding impulse)
          is given and the waveform is *still clearly periodic* there -- the
          autocorrelation at ``prev_lag`` is above :data:`MIN_PERIODICITY` and
          within 75 % of the f0-anchored peak -- that period is kept. This is
          what stops sudden octave jumps on quasi-stationary tones (a strong
          harmonic making the autocorrelation at ``T/2`` look almost as high, or
          a momentary f0 octave glitch). On transients the autocorrelation at
          ``prev_lag`` collapses, so the search is released and the period is
          re-measured freely.
        * **Octave-down correction** (only at a note onset, ``prev_lag=None``):
          if f0 was an octave/harmonic too low, a clean sub-multiple of the
          period is almost as periodic -> the shortest such period is taken.

        Returns the lag, or ``None`` if not clearly periodic.
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
        # f0-anchored peak (the pitch-derived period, refined to nearest peak)
        base = None
        base_val = 0.0
        if f0h > 0:
            t0 = int(min(tmax, max(tmin, round(fps / f0h))))
            a0, a1 = max(tmin, t0 - 3), min(tmax, t0 + 3)
            base = a0 + int(np.argmax(acn[a0:a1 + 1]))
            base_val = acn[base]
        # continuity: keep the previous period while it is still clearly
        # periodic -> prevents octave jumps during quasi-stationary passages.
        if prev_lag is not None and tmin <= prev_lag <= tmax:
            hw = max(2, prev_lag // 12)
            p0, p1 = max(tmin, prev_lag - hw), min(tmax, prev_lag + hw)
            pl = p0 + int(np.argmax(acn[p0:p1 + 1]))
            if acn[pl] >= MIN_PERIODICITY and acn[pl] >= 0.75 * base_val:
                return pl
        if base is None or base_val < MIN_PERIODICITY:
            return None
        # octave-down correction (note onset only): if f0 was an octave/harmonic
        # too low, take the shortest sub-multiple that is almost as periodic.
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
                           ``.features`` ``Pitch`` column for the period search
                           hint) or a DataFrame indexed by ``time_s`` with a
                           ``Pitch`` column. The level gate is measured from the
                           waveform, so an ``SPL`` column is no longer required.

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

        n_samples = x.size
        sample_t = np.arange(n_samples) / fps
        f0_s = np.interp(sample_t, frame_t, f0)
        # level gate in dBFS (0 dBFS = full scale), from a 25 ms RMS of the
        # waveform via a running sum -> silence is strongly negative. This makes
        # the gate independent of any external SPL scaling.
        win = max(1, int(round(0.025 * fps)))
        csum = np.concatenate(([0.0], np.cumsum(x * x)))
        idx = np.arange(n_samples)
        half = win // 2
        lo_i = np.clip(idx - half, 0, n_samples)
        hi_i = np.clip(idx - half + win, 0, n_samples)
        meansq = (csum[hi_i] - csum[lo_i]) / np.maximum(hi_i - lo_i, 1)
        level_db = 10.0 * np.log10(meansq + 1e-12)   # 10*log10(ms) = 20*log10(rms)
        active = level_db >= self.dbmin
        ax = np.abs(x)
        # zero crossings of both slopes: +1 rising (- -> +), -1 falling (+ -> -)
        rising = np.flatnonzero((x[:-1] <= 0.0) & (x[1:] > 0.0))
        falling = np.flatnonzero((x[:-1] >= 0.0) & (x[1:] < 0.0))
        zc = np.concatenate([rising, falling])
        zc_slope = np.concatenate([np.ones(rising.size, dtype=np.int8),
                                   -np.ones(falling.size, dtype=np.int8)])
        order = np.argsort(zc, kind="mergesort")
        zc = zc[order]
        zc_slope = zc_slope[order]

        tmin = max(2, int(round(fps / self.f_max)))       # shortest period
        tmax = max(tmin + 2, int(round(fps / self.f_min)))  # longest period
        use_wave = self.period_source == 'wave'
        thr = self.correlation_threshold

        if self.verbose:
            print("impulse pattern ...", end=" ")
        pace = timer()

        def period_at(t, prev_lag=None):
            if use_wave:
                lag = self._local_period(x, t, fps, tmin, tmax, f0_s[t],
                                         prev_lag)
                if lag is not None:
                    return lag
            # fallback (noise / aperiodic parts, or period_source='pitch'):
            # bounded pitch period
            f0h = f0_s[t]
            if f0h <= 0:
                return tmax
            return int(min(tmax, max(tmin, round(fps / f0h))))

        def normcorr(a, b):
            """Normalised (Pearson, zero-lag) correlation of two segments."""
            m = min(a.size, b.size)
            if m < 4:
                return 0.0
            a = a[:m] - a[:m].mean()
            b = b[:m] - b[:m].mean()
            na = float(np.sqrt(np.dot(a, a)))
            nb = float(np.sqrt(np.dot(b, b)))
            if na < 1e-12 or nb < 1e-12:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        def onset(prev, t_len, peak, allowed_slope):
            """Zero crossing starting the period of ``peak``; returns (idx, slope).

            The onset is the last (allowed-slope) crossing before ``peak`` and
            after ``prev``; if the peak precedes them, the crossing nearest to
            one period ahead of ``prev``. ``allowed_slope`` in {+1, -1} restricts
            to that slope, 0 allows both.
            """
            hz = min(n_samples - 1, prev + 2 * t_len)
            a = np.searchsorted(zc, prev, side="right")
            b = np.searchsorted(zc, hz, side="right")
            if b <= a:
                return -1, 0
            sl = zc[a:b]
            sp = zc_slope[a:b]
            if allowed_slope != 0:
                keep = sp == allowed_slope
                sl = sl[keep]
                sp = sp[keep]
                if sl.size == 0:
                    return -1, 0
            bp = np.searchsorted(sl, peak, side="right")
            k = bp - 1 if bp > 0 else int(np.argmin(np.abs(sl - (prev + t_len))))
            return int(sl[k]), int(sp[k])

        imp = []
        corr = []                    # correlation value per impulse
        prev = -1                    # last impulse (-1 = start of an active run)
        pprev = -1
        cur_slope = 1
        corr_run = 1.0               # start locked
        last_T = None                # last accepted period (continuity hint)
        i = 0
        while i < n_samples - 1:
            if not active[i]:
                prev = -1; pprev = -1; corr_run = 1.0; last_T = None
                nxt = int(np.argmax(active[i:]))
                if active[i + nxt]:
                    i = i + nxt
                    continue
                break
            if prev < 0:
                # first impulse after silence: peak in [i, i+T], nearest ZC before it
                t_len = period_at(i)
                hi = min(n_samples, i + t_len)
                if hi - i < 2:
                    i += 1
                    continue
                peak = i + int(np.argmax(ax[i:hi]))
                t_imp, s = onset(i - 1, t_len, peak, 0)
                if t_imp < i:
                    a = int(np.searchsorted(zc, i, side="left"))
                    if a < zc.size:
                        t_imp, s = int(zc[a]), int(zc_slope[a])
                    else:
                        t_imp, s = i, 1
                imp.append(t_imp); corr.append(0.0)
                cur_slope = s; prev = t_imp; pprev = -1; last_T = t_len
                i = t_imp + 1
                continue
            # subsequent impulse ~ one period ahead of ``prev``
            t_len = period_at(prev, last_T)
            last_T = t_len
            lo = prev + max(2, t_len // 2)
            hi = min(n_samples, prev + (3 * t_len) // 2)
            if hi - lo < 2:
                prev = -1; pprev = -1; corr_run = 1.0; last_T = None
                i = hi if hi > i else i + 1
                continue
            peak = lo + int(np.argmax(ax[lo:hi]))
            allowed = cur_slope if corr_run >= thr else 0   # lock or free slope
            t_imp, s = onset(prev, t_len, peak, allowed)
            if t_imp <= prev:
                t_imp, s = onset(prev, t_len, prev + t_len, 0)
                if t_imp <= prev:
                    t_imp = min(n_samples - 2, prev + t_len)
                    s = cur_slope
            if allowed == 0:
                cur_slope = s
            # correlation of the two newest periods -> the middle impulse ``prev``
            if pprev >= 0:
                corr_run = normcorr(x[pprev:prev], x[prev:t_imp])
                corr[-1] = corr_run
            if t_imp >= n_samples - 1:
                break
            imp.append(t_imp); corr.append(0.0)
            pprev = prev; prev = t_imp; i = t_imp

        imp = np.asarray(imp, dtype=int)
        corr = np.asarray(corr, dtype=float)
        amps = np.zeros(imp.size)
        for k in range(imp.size):
            a = imp[k]
            b = imp[k + 1] if k + 1 < imp.size else min(n_samples, a + period_at(a))
            if b > a:
                amps[k] = ax[a:b].max()

        if self.verbose:
            print(f"{timer() - pace:.4} s. ({imp.size} impulses)")

        impulses = pd.DataFrame({"time_s": np.round(imp / fps, 6),
                                 "amplitude": amps,
                                 "correlation": np.round(corr, 4)})
        meta = TrackMeta(comsar.__version__, datetime.utcnow(),
                         SourceMeta(*file_name.split("."), file_hash))
        params = {"dbmin": self.dbmin, "f_min": self.f_min, "f_max": self.f_max,
                  "period_source": self.period_source,
                  "correlation_threshold": self.correlation_threshold}
        return ImpulsePatternResult(meta, params, impulses)
