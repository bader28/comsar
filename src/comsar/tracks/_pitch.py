"""comsar/tracks/pitch.py -- TimbreTack implementation
License: BSD-3-Clasuse
Copyright (C) 2020, Michael Blaï¿½, michael.blass@uni-hamburg.de
"""
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any, Optional
from scipy import interpolate

import numpy as np
import pandas as pd
import apollon.audio as apa
import apollon.segment as aps
import apollon.signal.tools as ast
import apollon.tools as apt

from apollon.audio import AudioFile
"""from apollon.segment import Segmentation"""
from apollon.signal import container, features
from apollon.signal.spectral import StftSegments
from apollon.tools import time_stamp

import apollon.signal as signal

import comsar
from comsar.tracks.utilities import TrackMeta, TrackParams, TrackResult, PitchTrackParams, TonalSystemParams, ngramParams

try:
    from importlib.resources import files as _res_files
except ImportError:                     # Python < 3.9 fallback
    from importlib_resources import files as _res_files


def _load_scales() -> pd.DataFrame:
    """Load the bundled ``scales.csv`` shipped as comsar package data.

    Replaces the previous ``pd.read_csv('scales.csv')`` which only worked
    when the current working directory happened to contain the file.
    """
    with (_res_files('comsar') / 'scales.csv').open('r', encoding='utf-8') as _fh:
        return pd.read_csv(_fh, index_col=0)


STFT_DEFAULT = container.StftParams(fps=44100, window='hamming', n_fft=None,
                                    n_perseg=2205, n_overlap=0,
                                    extend=True, pad=True)

TONALSYSTEM_DEFAULT = TonalSystemParams(dcent=1, dts=0.1, minlen=3, mindev=60, noctaves=8, f0=27.5)

NGRAM_DEFAULT = ngramParams(minnotelength=10, ngram=3, ngcentmin=0, ngcentmax=1200, nngram=10)

# Legacy sample-based default (50 ms windows with 80% overlap at 44.1 kHz).
# Only used when an explicit ``seg_params`` object is passed.
SEGMENTATION_DEFAULT = aps.SegmentationParams(n_perseg=2205, n_overlap=1764, pad=False)

# Time-based defaults, equivalent to the historical values at 44.1 kHz but
# expressed in milliseconds so that any sample rate yields the same time
# resolution.
WINDOW_MS_DEFAULT = 50.0    # = 2205 samples at 44.1 kHz
OVERLAP_DEFAULT = 0.8       # = 1764 samples at 44.1 kHz

# f0-free YIN pitch detector (replaces apollon's acf_pitch)
FMIN_PITCH_DEFAULT = 60.0   # lowest detectable f0 (Hz) -> longest period
FMAX_PITCH_DEFAULT = 1200.0  # highest detectable f0 (Hz) -> shortest period
YIN_THRESHOLD = 0.12        # absolute threshold on the CMND function
YIN_VOICED_DP = 0.35        # dp above this -> treat frame as unvoiced (Pitch=0)


class PitchTrack:
    """Compute PitchTrack of an audio file.

    The analysis windowing is specified in *time* units (milliseconds and an
    overlap fraction). Window and hop size are converted to samples using the
    sample rate of each analysed file, so audio files with *different sample
    rates* are directly comparable: equal durations produce (up to rounding)
    equal numbers of analysis frames, and the returned feature table carries
    the frame time in seconds as its index.
    """
    def __init__(self,
                 window_ms: float = WINDOW_MS_DEFAULT,
                 overlap: float = OVERLAP_DEFAULT,
                 f_min: float = FMIN_PITCH_DEFAULT,
                 f_max: float = FMAX_PITCH_DEFAULT,
                 pitch_method: str = 'yin',
                 seg_params: Optional[aps.SegmentationParams] = None,
                 tonalsystem_params: Optional[TonalSystemParams] = None,
                 ngram_params: Optional[ngramParams] = None) -> None:
        """
        Args:
            window_ms:   Hop grid of the pitch track in milliseconds (the time
                         resolution). The YIN analysis window itself is sized
                         from ``f_min`` (>= two of the longest periods),
                         independent of this value.
            overlap:     Overlap between consecutive frames as a fraction of
                         ``window_ms`` (0 < overlap < 1); sets the hop.
            f_min, f_max:  Frequency bounds (Hz) of the f0-free YIN pitch
                         detector; the period is searched in ``[1/f_max, 1/f_min]``.
            seg_params:  Legacy sample-based segmentation parameters. If given,
                         ``window_ms``/``overlap`` are ignored (historical
                         behaviour with a fixed sample rate).
        """
        if seg_params is None and not 0.0 < float(overlap) < 1.0:
            raise ValueError('``overlap`` must be a fraction with '
                             f'0 < overlap < 1, got {overlap}.')

        self.window_ms = float(window_ms)
        self.overlap = float(overlap)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.pitch_method = str(pitch_method).lower()
        if self.pitch_method not in ('yin', 'acf'):
            raise ValueError("pitch_method must be 'yin' (default, f0-free YIN "
                             "detector) or 'acf' (legacy autocorrelation), got "
                             f"{pitch_method!r}.")
        self._fixed = seg_params        # legacy mode if not None

        if self._fixed is not None:
            self.params = PitchTrackParams(self._fixed)
            self.cutter = aps.Segmentation(**self._fixed.to_dict())
        else:
            self.params = None          # set per file in ``extract``
            self.cutter = None

        self.TSparams = TonalSystemParams(tonalsystem_params or TONALSYSTEM_DEFAULT)
        self.ngparams = ngramParams(ngram_params or NGRAM_DEFAULT)

        self.feature_names = ('Pitch','SPL')
        # Pitch is filled in ``extract`` by the selected detector
        # (``pitch_method``: 'yin' -> ``_yin_pitch``, 'acf' -> ``acf_pitch``);
        # the worker only needs a placeholder for the Pitch column here.
        self.funcs = [_pitch_zero, features.spl]
        self.pace = np.zeros(self.n_features)
        self.verbose = False

    @property
    def n_features(self) -> int:
        """Number of features on track"""
        return len(self.feature_names)

    def _make_cutter(self, fps: int) -> aps.Segmentation:
        """Convert the time-based window setup to samples for rate ``fps``."""
        if self._fixed is not None:
            return self.cutter
        n_perseg = max(2, int(round(self.window_ms * fps / 1000.0)))
        n_overlap = min(max(int(round(n_perseg * self.overlap)), 1),
                        n_perseg - 1)
        # pad=True guarantees enough frames at the borders; the frame count is
        # then trimmed to the rate-independent target in ``extract``.
        seg_params = aps.SegmentationParams(n_perseg=n_perseg,
                                            n_overlap=n_overlap,
                                            extend=True, pad=True)
        self.params = PitchTrackParams(seg_params)
        self.cutter = aps.Segmentation(**seg_params.to_dict())
        return self.cutter

    def extract(self, input1: None, input2: Optional[Any] = None, input3: Optional[Any] = None) -> pd.DataFrame:
        """Perform extraction.

        Any sample rate is accepted; the analysis windows are computed from
        ``window_ms``/``overlap`` for the file's own sample rate. The index of
        the returned feature table is the frame time in seconds (``time_s``).
        """
        if type(input1) is str:
            snd = apa.AudioFile(input1)
            fps = snd.fps
            cutter = self._make_cutter(fps)
            mono_full = np.asarray(snd.data.squeeze(), dtype=float)
            segs = cutter.transform(snd.data.squeeze())
            args = [(segs.data, snd.fps),
                (segs.data,)]
        else:
            snd = input1
            fps = int(input2)
            cutter = self._make_cutter(fps)
            mono_full = np.asarray(snd.squeeze(), dtype=float)
            segs = cutter.transform(snd.squeeze())
            args = [(segs.data, input2),
                (segs.data,)]
        if mono_full.ndim > 1:
            mono_full = mono_full.mean(axis=1)

        kwargs = [{},{}]

        out = np.zeros((segs.n_segs, self.n_features))
        for i, (fun, arg, kwarg) in enumerate(zip(self.funcs, args, kwargs)):
            out[:, i] = self._worker(i, fun, arg, kwarg)

        if type(input1) is str:
            meta = TrackMeta(comsar.__version__, apt.time_stamp(),
                         snd.file_name)
        else:
            meta = TrackMeta(comsar.__version__, apt.time_stamp(),
                         input3)

        out = pd.DataFrame(data=out, columns=self.feature_names)
        # frame time in seconds: with ``extend=True`` the first frame is
        # centred on t=0, so frame i sits at i * hop
        seg = self.params.segmentation
        hop = seg.n_perseg - seg.n_overlap
        if self._fixed is None:
            # Rounding the hop to whole samples plus border extension/padding
            # can yield extra frames at some sample rates. Trim to the
            # rate-independent count ceil(duration_ms / hop_ms), so files
            # of equal duration produce the same number of frames at any
            # sample rate.
            if type(input1) is str:
                n_samples = snd.data.squeeze().shape[0]
            else:
                n_samples = snd.squeeze().shape[0]
            duration_ms = n_samples / fps * 1000.0
            hop_ms = self.window_ms * (1.0 - self.overlap)
            n_expected = int(np.ceil(duration_ms / hop_ms - 1e-6))
            if len(out) > n_expected:
                out = out.iloc[:n_expected]
        out.index = np.round(np.arange(len(out)) * (hop / fps), 6)
        out.index.name = 'time_s'

        # Fill the Pitch column with the selected detector. 'yin' (default) is
        # the f0-free YIN detector evaluated at each frame centre; 'acf' is the
        # legacy per-frame autocorrelation pitch.
        if self.pitch_method == 'acf':
            pitch_acf = np.asarray(acf_pitch(segs.data, fps), dtype=float)
            out['Pitch'] = pitch_acf[:len(out)]
        else:
            centers = np.rint(out.index.to_numpy(dtype=float) * fps).astype(int)
            out['Pitch'] = self._yin_pitch(mono_full, fps, centers)

        if type(input1) is str:
            snd.close()

        return TrackResult(meta, self.params, out)

    def _yin_pitch(self, x: np.ndarray, fps: int,
                   centers: np.ndarray) -> np.ndarray:
        """f0-free pitch (Hz) at each sample position in ``centers``.

        A YIN cumulative-mean-normalised difference function is evaluated on a
        window centred on each frame (sized from ``f_min`` so at least two of
        the longest periods fit, independent of the hop grid). Frames whose best
        dip is not clearly periodic (``dp >= YIN_VOICED_DP``) are marked unvoiced
        (``Pitch = 0``). The raw track is then octave-cleaned by snapping every
        voiced frame to the nearest octave of its local median -- this is what
        removes the octave errors of the previous autocorrelation pitch.
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        tmin = max(2, int(round(fps / self.f_max)))
        tmax = max(tmin + 2, int(round(fps / self.f_min)))
        win = 2 * tmax
        need = win + tmax
        half = need // 2
        per = np.zeros(centers.size)
        qual = np.ones(centers.size)
        for k, c in enumerate(centers):
            lo = int(c) - half
            seg = np.zeros(need)
            a = max(0, lo); b = min(n, lo + need)
            if b > a:
                seg[a - lo:b - lo] = x[a:b]
            per[k], qual[k] = _yin_frame(seg, tmin, tmax, win, YIN_THRESHOLD)
        voiced = (per > 0) & (qual < YIN_VOICED_DP)
        pitch = np.where(voiced, fps / np.where(per > 0, per, 1.0), 0.0)
        return _octave_clean(pitch)

    def extract_TonalSystem(self, data: np.ndarray, dcent: float, dts: float, minlen: int, mindev: int, noctaves: int, f0: float) -> np.ndarray:
        """Pitch cummulation and Tonal System Extraction
        
        def extract_TonalSystem(self, data: np.ndarray) -> np.ndarray:
        Input:
        data: Freuencies of adjacent segments of a sound file
        dcent: Accummulation precision in cent, examples: fine grain: dcent = 1, semitone grain. decent = 100
        dts: standard deviation of tonal system pitch entry
            used for correlation with accumulated tonal system from sound
        minlen: Minimum length of adjacent cent values to be accepted as an event (a note, etc.)
        mindev: minimum deviation allowed to qualify as note in cent:
        noctaves: number of octaves starting from f0, default: 8
        f0: start frequency of octaves, default: 27.5 Hz => subcontra A
    
        Return:
        c: Cummulated frequency spectrum
        co: Cummulated frequency spectrum within one octave, where the frequency of maximum amplitude is
            vector entry zero and co length is defined by dcent, example: with dcent = 1, co length = 1200
        maxf: Frequency of maximum amplitude of cummulated spectrum
        retNames: Names of the ten best matching scales meeting the input frequency data
        retScale: Theoretical scale values of the ten best matching scales
        retValue: Contribution of each scale step to the overall correlation for each of the ten best-matching scales
        retCorr: Overall correlation for the whole scale for each of the ten best-matching scales
        nnotes: Number of notes in sound
        notes: Notes of sound as array of pitch_type class instantiations, 
        with note type ('note', 'pause', etc.), note start, note stop, note args, where arg1 is note in cent above f0
        cn: Accumulated tonal system spectrum within one octave with precision dcent from pitch events only,
        Compared to c0 (see above), which is accululated spectrum over all pitches in data
        
        dcent = self.TSparams.dcent
        print(dcent)
        dts = self.TSparams.dts
        minlen = self.TSparams.minlen
        mindev = self.TSparams.mindev
        noctaves = self.TSparams.noctaves
        f0 = self.TSparams.f0
		"""

        scales = _load_scales()
        for i in range(0,len(scales)):
            for j in range(0,12):
                if np.isnan(scales.iloc[i, j]):
                    scales.iloc[i, j] = 0
    
        root=np.power(2,1/(1200/dcent))
        root1200=1/np.log(root)
        n=int(1200/dcent*noctaves)
        no=int(float(1200)/float(dcent))

        cent=np.zeros(data.size)
        debug=np.zeros(data.size)
        c=np.zeros(n)
        co=np.zeros(no)
    
        #Frequency to cent
        for i in range(0,cent.size):
            if data[i] >= f0:
                cent[i] = np.round(np.log(data[i]/f0) * root1200)
            else:
                cent[i] = 0
            
        #Detect Notes
        #Collection of notes
        notes = []
        #Number of notes
        nnotes = 0
        pos = 0
    
        while pos < cent.size:
            mean = np.mean(cent[pos:pos+minlen])
            exceeds_over  = len(np.where(cent[pos:pos+minlen] > mean + mindev)[0])
            exceeds_under = len(np.where(cent[pos:pos+minlen] < mean - mindev)[0])
        
            #Within a minimum note length, is there any cent deviation over or under mindev? If not, it is a valid note
            if exceeds_over  == 0 and exceeds_under == 0:
                cont = 1
                notecont = True
            
                #Maybe the note is longer than minlen. Then the note is prolonged until condition fails
                while notecont == True and pos + minlen + cont < cent.size:               
                    mean = np.mean(cent[pos:pos+minlen+cont])
                    exceeds_over  = len(np.where(cent[pos:pos+minlen+cont] > mean + mindev)[0])
                    exceeds_under = len(np.where(cent[pos:pos+minlen+cont] < mean - mindev)[0])
                
                    if exceeds_over  == 0 and exceeds_under == 0:
                        cont = cont + 1
                    #Note ends at a cent = 0 (unvoiced) frame: record the valid
                    #part *before* skipping it (otherwise a good note that simply
                    #ends on an unvoiced frame would be dropped entirely).
                    elif len(np.where(cent[pos:pos+minlen+cont] == 0)[0]) != 0:
                        cn=np.zeros(n)
                        for j in range(pos, pos + minlen + cont -1):
                            if cent[j] <= n and cent[j] > 0:
                                cn[int(cent[j])] += 1
                        if max(cn) > 0:
                            maxa = max(cn)
                            notes.append(pitch_type('note', pos, pos + minlen + cont -1, np.where(cn == maxa)[0][0], 0))
                            debug[pos] = notes[nnotes].start
                            nnotes = nnotes + 1
                        pos = pos + minlen + cont
                        notecont = False
                    #Maximum length of note arrived at minlen + cont
                    else:  
                        #Accumulate cent values in note and take maximum as pitch of note
                        cn=np.zeros(n)
                        for j in range(pos, pos + minlen + cont -1):
                            if cent[j] <= n and cent[j] > 0:
                                cn[int(cent[j])] += 1
                        maxa = max(cn)
                        notes.append(pitch_type('note', pos, pos + minlen + cont -1, np.where(cn == maxa)[0][0], 0))
                        debug[pos] = notes[nnotes].start
                        pos = pos + minlen + cont
                        nnotes = nnotes + 1
                        notecont = False

                #Note ran to the very end of the data: record it (otherwise the
                #last note of a file is dropped when the loop hits the end).
                if notecont:
                    hi = min(pos + minlen + cont, cent.size)
                    cn = np.zeros(n)
                    for j in range(pos, hi):
                        if 0 < cent[j] <= n:
                            cn[int(cent[j])] += 1
                    if max(cn) > 0:
                        maxa = max(cn)
                        notes.append(pitch_type('note', pos, hi - 1, np.where(cn == maxa)[0][0], 0))
                        debug[pos] = notes[nnotes].start
                        nnotes = nnotes + 1
                    pos = hi
                    notecont = False

            pos = pos + 1

        #Detect tonal system
        """Find strongest pitch in cent over all octaves"""
        for i in range(1,nnotes):
            for j in range(notes[i].start, notes[i].stop):
                if cent[j] <= n and cent[j] > 0:
                    c[int(cent[j])] += 1
                
        #Accummulate cents into octave
        maxa = max(c)
        maxs = np.where(c == maxa)[0][0]
        maxf = f0 * np.power(root, maxs)
    
        #Accumulate pitches into ocatve
        cn=np.zeros(no)
        for i in range(0,nnotes):
            cn[int(np.mod(notes[i].arg1-maxs,no))] += 1
    
        """Cummulate cent values into one octave with strongest cent, maxs, as fundamental frequency of tonal system"""
        for i in range(1,n):
                co[int(np.mod(i-maxs,no))] += c[i]

        co = co/np.linalg.norm(co)
    
        """Sum amplitudes in co at scale positions matching cent values of all theoretical scales in valiable ts"""
        ts=np.zeros(scales.shape[0])

        ar=np.arange(1200)
        for i in range(0,scales.shape[0]):
            ts[i] += co[0]
            for j in range(0,11):
                if np.logical_not(np.isnan(scales.iloc[i, j]/dcent)):
                    #Correlate each pitch of tonal system as a gauss shape with calculated tonal system from sound
                    ts[i] += np.sum(co*2.718281**(-(scales.iloc[i, j]/dcent-1-ar)**2/dts**2))
                    #aa=np.zeros(no)
                    #aa[int(scales.iloc[i, j]/dcent-1)] = 1
                    #ts[i] += np.sum(co*aa)
                    #ts[i] += co[int(scales.iloc[i, j]/dcent-1)]

        """Detecting the nret = 10 best matching scales in variable tss"""            
        nret=10
        retNames = np.empty([nret],dtype='object')
        retCorr = np.empty([nret],dtype=float)
        retScale = np.zeros([nret,13],dtype=int)
        retValue = np.zeros([nret,13],dtype=float)
        tss = ts
        for i in range(0,nret):
            maxts = max(tss)
            retCorr[i] = maxts
            maxts = np.where(tss == maxts)[0][0]
            retNames[i] = scales.index[maxts]
            retScale[i][0] = 0
            retValue[i][0] = co[0]
            for j in range(0,11):
                if np.logical_not(np.isnan(scales.iloc[maxts, j])):
                    retScale[i][j+1] = scales.iloc[maxts, j]
                    retValue[i][j+1] = co[int(scales.iloc[maxts, j]/dcent)-1]
            tss[maxts] = 0
    
        return c, co, maxf, retNames, retScale, retValue, retCorr, nnotes, notes, cn

    def extract_ngram(self, notes: np.ndarray, nnotes: int, dcent: int, minnotelength: int, ngram: int, ngcentmin: int, ngcentmax: int, nngram: int) -> np.ndarray:

        """
        Args:
    
        minnotelength: minimum length of a note to qualify as melody to be used in ngram calculation, value in analysis frames
        ngram: ngram depth, 0: no ngram calculation 2: 2-gram, 3: 3-gram, 4: 4-gram, 5: 5-gram.
        ngram calculation is performed over intervals, not absolute pitches
        ngcentmin: minimum interval step in cent to qualify interval to be used in ngram calculation
        ngcentmax: +-maximum interval to be used for ngram calculation, e.g. 1200 allows for +-1200 cent intervals
        nngram: number of largest ngram histograms to be calculated, e.g. 10
    
        Returns:
    
        ngrams: array of nngram ngrams, most frequently occuring in sound.
        E.g. 3-gram,nngram = 10, ngcentmax = 1200 -> 10 ngrams x 2 intervals (3-grams) = 20 values in array
        [ngram 1 1st interval, ngram 1, 2nd interval, ngram 2 1st interval, ngram 2, 2nd interval,...], most frequent ngram first
        ngram value coding: ngcentmax = 12000 -> +-12 intervals. ngram value = 0 -> -12 half tones, ngram value = 12 -> 0 half tones, ngram value = 24 -> +12 half tones
    
        notesinngram: notes used in ngram calculation as subset of notes applying ninnotelength condition
        """
        # NOTE: the arguments passed to this method are used directly. (An earlier
        # version overrode them with self.ngparams.mintolength / self.ngparams.n,
        # which do not exist on ngramParams and raised AttributeError.)

        ngrams = []
        if (ngram > 1 and ngram <= 5):
            ngramsall=[]
            numgram = 0 # number of different ngrams in sound
            #Calculating ngrams
            notesinngram = []
            #i = 0
            noteindex = []
            for i in range(0,nnotes-ngram):
                step = np.zeros(ngram-1)
                #Does note qualify for ngram in terms of note length
                if ((notes[i].stop - notes[i].start) >= minnotelength):
                    k = 0
                    l = 1
                    kold = 0
                    #Construct ngram
                    while k < ngram-1 and i+l < nnotes-ngram:
                        #Does note qualify for ngram in terms of note length
                        if ((notes[i+l].stop - notes[i+l].start) >= minnotelength):
                            step[k] = notes[i+l].arg1 - notes[i+kold].arg1
                            k += 1
                            kold = l
                            l += 1
                        else:
                            l += 1
                    #Is ngram within defined region of ngcentmin and ngcentmax
                    if (len([x for x in step if np.abs(x)  >= ngcentmin/dcent and np.abs(x) <= ngcentmax/dcent]) == (ngram-1)):
                        ngramsall.append(step)
                        numgram += 1
                        notesinngram.append(notes[i])
        
            #Calculating ngram histogram and seeking for nngram most frequent ngrams
            #Equidistant 12-tone tonal system used
            justint = 100
            ngrams = np.zeros((ngram-1)*nngram)
            histrange = int(ngcentmax/justint)
            sh=np.arange(ngram-1)
            for i in range(0,ngram-1):
                sh[i] = 2*histrange+1
            hist=np.zeros(shape=sh)
            for i in range(0,numgram):
                wo = (int(ngramsall[i][0]/justint+histrange),)
                for k in range(1,ngram-1):
                    wo = wo + (int(ngramsall[i][k]/justint+histrange),)
                hist[wo] += 1
        
            nmax = 0
            while nmax < nngram:
                #number of positions with maximum ngram occurance could be larger than 1
                nn=len(np.where(hist==hist.max())[0])
                #Allow only up to nngram
                if ((nmax+nn) > nngram):
                    nn = nngram - nmax
                    #print('hinaus')
                maxvals = np.where(hist==hist.max())
                for i in range(0,nn):
                    for k in range(0,ngram-1):
                        ngrams[(nmax+i)*(ngram-1)+k] = maxvals[k][i]
                    wo = (int(ngrams[(nmax+i)*(ngram-1)]),)
                    for k in range(1,ngram-1):
                        wo = wo + (int(ngrams[(nmax+i)*(ngram-1)+k]),)
                    hist[wo] = 0
                nmax += nn				
	
        return ngrams, notesinngram

    # ------------------------------------------------------------------
    # Stage 3+4: melody / notes and tonal system (clean wrappers around the
    # verified ``extract_TonalSystem`` algorithm; frames -> seconds/Hz).
    # ------------------------------------------------------------------
    def _analyse_scale(self, pitch_result, dcent, dts, minlen, mindev,
                       noctaves, f0):
        feats = pitch_result.features if hasattr(pitch_result, 'features') \
            else pitch_result
        data = np.asarray(feats['Pitch'].to_numpy(), dtype=float)
        frame_t = np.asarray(feats.index, dtype=float)
        res = self.extract_TonalSystem(data, dcent, dts, minlen, mindev,
                                       noctaves, f0)
        return res, frame_t

    @staticmethod
    def _notes_df(notes, frame_t, f0):
        nf = frame_t.size
        rows = []
        for nt in notes:
            s = int(nt.start); e = int(nt.stop)
            start_s = float(frame_t[min(s, nf - 1)]) if nf else 0.0
            stop_s = float(frame_t[min(e, nf - 1)]) if nf else 0.0
            cent = float(nt.arg1)
            freq = f0 * 2.0 ** (cent / 1200.0)
            midi = int(round(69 + 12 * np.log2(freq / 440.0))) if freq > 0 else 0
            rows.append({'start_s': round(start_s, 6), 'stop_s': round(stop_s, 6),
                         'duration_s': round(stop_s - start_s, 6),
                         'frequency': round(freq, 3), 'cent': round(cent, 1),
                         'midi': midi})
        return pd.DataFrame(rows, columns=['start_s', 'stop_s', 'duration_s',
                                           'frequency', 'cent', 'midi'])

    def notes(self, pitch_result, dcent=1, minlen=15, mindev=60, noctaves=8,
              f0=27.5, dts=0.1):
        """Detect the **melody**: segment the f0 track into notes.

        A note is a run of at least ``minlen`` frames whose pitch stays within
        ``mindev`` cent of its mean. Returns a DataFrame (one row per note) with
        ``[start_s, stop_s, duration_s, frequency, cent, midi]`` -- times in
        seconds, pitch in Hz / cent above ``f0`` / MIDI number.

        Args:
            pitch_result:  A :class:`PitchTrack` result (or its ``.features``).
            dcent:         Pitch resolution in cent.
            minlen:        Minimum note length in frames.
            mindev:        Maximum pitch deviation within a note, in cent.
            noctaves, f0:  Analysed octave span and reference frequency (Hz).
        """
        res, frame_t = self._analyse_scale(pitch_result, dcent, dts, minlen,
                                           mindev, noctaves, f0)
        return self._notes_df(res[8], frame_t, f0)

    def tonal_system(self, pitch_result, dcent=1, minlen=15, mindev=60,
                     noctaves=8, f0=27.5, dts=0.1, n_best=10):
        """Determine the **tonal system** (scale) of the recording.

        The measured pitches are accumulated into one octave and correlated
        with 900+ theoretical scales (``scales.csv``). Returns a
        :class:`TonalSystemResult` with the best-matching scales, the measured
        one-octave distribution and the detected notes (melody).
        """
        res, frame_t = self._analyse_scale(pitch_result, dcent, dts, minlen,
                                           mindev, noctaves, f0)
        c, co, maxf, retNames, retScale, retValue, retCorr, nnotes, nts, cn = res
        scales = []
        for i in range(min(n_best, len(retNames))):
            degrees = [0] + [int(v) for v in retScale[i] if v > 0]
            scales.append({'rank': i + 1, 'name': str(retNames[i]),
                           'correlation': float(retCorr[i]),
                           'degrees_cent': degrees})
        scales_df = pd.DataFrame(scales)
        note_df = self._notes_df(nts, frame_t, f0)
        return TonalSystemResult(scales_df, np.asarray(co, dtype=float),
                                 float(maxf), float(f0), note_df)

    def ngrams(self, pitch_result, ngram=3, minnotelength=10, ngcentmin=0,
               ngcentmax=1200, nngram=10, dcent=1, minlen=15, mindev=60,
               noctaves=8, f0=27.5, dts=0.1):
        """Melodic **n-gram patterns**: the most frequent interval sequences.

        The melody notes are turned into a sequence of pitch **intervals** (in
        cent), and the ``nngram`` most frequent interval n-grams are returned.
        Interval patterns (not absolute pitches) make the fingerprint
        transposition-invariant.

        Args:
            pitch_result:   A :class:`PitchTrack` result (or its ``.features``).
            ngram:          n-gram depth (2..5); a 3-gram spans two intervals.
            minnotelength:  Minimum note length (in frames) to take part.
            ngcentmin:      Minimum absolute interval in cent to be counted.
            ngcentmax:      Maximum absolute interval in cent (also the axis
                            range of the interval histogram).
            nngram:         Number of most frequent n-grams to return.

        Returns:
            A DataFrame with ``ngram - 1`` columns
            (``interval_1_cent`` … ``interval_{ngram-1}_cent``), most frequent
            n-gram first; each value is a melodic interval in cent (positive =
            up, negative = down), quantised to semitones (100 cent).
        """
        res, _ = self._analyse_scale(pitch_result, dcent, dts, minlen, mindev,
                                     noctaves, f0)
        notes, nnotes = res[8], res[7]
        ng, _ = self.extract_ngram(notes, nnotes, dcent, minnotelength, ngram,
                                   ngcentmin, ngcentmax, nngram)
        step = max(1, ngram - 1)
        histrange = int(ngcentmax / 100)
        rows = []
        ng = np.asarray(ng, dtype=float).ravel()
        for i in range(nngram):
            vals = ng[i * step:(i + 1) * step]
            if vals.size < step or np.all(vals == 0):
                continue
            rows.append([int(round((float(v) - histrange) * 100)) for v in vals])
        cols = ['interval_%d_cent' % (k + 1) for k in range(step)]
        return pd.DataFrame(rows, columns=cols)

    def _worker(self, idx, func, args, kwargs) -> np.ndarray:
        print(self.feature_names[idx], end=' ... ')
        pace = timer()
        res = func(*args, **kwargs)
        pace = timer() - pace
        self.pace[idx] = pace
        print(f'{pace:.4} s.')
        return res


class TonalSystemResult:
    """Result of :meth:`PitchTrack.tonal_system`.

    Attributes:
        scales:       DataFrame of the best-matching scales, columns
                      ``[rank, name, correlation, degrees_cent]`` (``degrees_cent``
                      is a list of scale-step positions in cent within an octave,
                      the tonic included as 0).
        octave:       measured one-octave pitch distribution (array of length
                      ``1200 / dcent``) -- the "measured tonal system".
        fundamental:  strongest frequency of the recording in Hz.
        f0_ref:       reference frequency used for the cent axis (Hz).
        notes:        melody DataFrame (see :meth:`PitchTrack.notes`).
    """
    def __init__(self, scales, octave, fundamental, f0_ref, notes):
        self.scales = scales
        self.octave = octave
        self.fundamental = fundamental
        self.f0_ref = f0_ref
        self.notes = notes

    @property
    def best(self):
        """The single best-matching scale (a Series), or ``None``."""
        return self.scales.iloc[0] if len(self.scales) else None

    def scale_frequencies(self, f_lo=20.0, f_hi=8000.0, rank=0):
        """Absolute frequencies (Hz) of the ``rank``-th scale's degrees.

        The scale (its cent degrees within an octave) is laid out over every
        octave between ``f_lo`` and ``f_hi``, anchored to the recording's
        fundamental. Useful to draw the tonal system as reference lines.
        """
        if len(self.scales) == 0:
            return []
        degrees = self.scales.iloc[rank]['degrees_cent']
        base = self.fundamental if self.fundamental > 0 else self.f0_ref
        while base > f_lo * 2.0:
            base /= 2.0
        freqs = []
        for o in range(0, 12):
            for d in degrees:
                fr = base * 2.0 ** (o + d / 1200.0)
                if f_lo <= fr <= f_hi:
                    freqs.append(fr)
            if base * 2.0 ** o > f_hi:
                break
        return sorted(set(round(f, 3) for f in freqs))


def _pitch_zero(sig: np.ndarray, fps: int, **kwargs) -> np.ndarray:
    """Placeholder pitch worker (Pitch is filled by the YIN detector)."""
    return np.zeros(np.atleast_2d(sig).shape[1])


def _yin_frame(seg: np.ndarray, tmin: int, tmax: int, win: int,
               thr: float) -> tuple:
    """YIN period (samples) and its dip value for one window.

    Returns ``(period, dp)`` where ``period`` is a sub-sample-refined lag (0 if
    the window is too short) and ``dp`` is the cumulative-mean-normalised
    difference at that lag (small = clearly periodic). Being f0-free and using
    the *first* dip below ``thr`` (not the global autocorrelation maximum), it is
    robust against the octave errors that plague plain autocorrelation.
    """
    if seg.size < win + tmax:
        return 0.0, 1.0
    s = seg[:win + tmax]
    cum = np.concatenate(([0.0], np.cumsum(s * s)))
    taus = np.arange(tmax + 1)
    e0 = cum[win] - cum[0]
    e_tau = cum[taus + win] - cum[taus]           # energy of the shifted window
    nfft = 1
    while nfft < 2 * (win + tmax):
        nfft <<= 1
    f = np.fft.rfft(s[:win], nfft)
    g = np.fft.rfft(s, nfft)
    r = np.fft.irfft(np.conj(f) * g)[:tmax + 1]    # cross terms sum s_j s_{j+tau}
    d = e0 + e_tau - 2 * r                          # difference function d(tau)
    d[d < 0] = 0.0
    cs = np.cumsum(d[1:])
    dp = np.ones(tmax + 1)
    tt = np.arange(1, tmax + 1)
    dp[1:] = d[1:] * tt / np.where(cs > 0, cs, 1.0)  # cumulative mean normalise
    # first local minimum below the absolute threshold, else the global min
    t = tmin
    cand = None
    while t < tmax:
        if dp[t] < thr:
            while t + 1 <= tmax and dp[t + 1] < dp[t]:
                t += 1
            cand = t
            break
        t += 1
    if cand is None:
        cand = tmin + int(np.argmin(dp[tmin:tmax + 1]))
    q = float(dp[cand])
    # parabolic interpolation around the dip for a sub-sample period
    period = float(cand)
    if 0 < cand < tmax:
        y0, y1, y2 = dp[cand - 1], dp[cand], dp[cand + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            shift = 0.5 * (y0 - y2) / denom
            if -1.0 < shift < 1.0:
                period = cand + shift
    return period, q


def _octave_clean(pitch: np.ndarray) -> np.ndarray:
    """Snap every voiced frame to the nearest octave of its local median.

    Works in log2 space over a +/-5 frame neighbourhood of voiced frames, so a
    single octave-slipped frame is pulled back onto the melodic line without
    disturbing genuine (sub-octave) pitch moves.
    """
    v = pitch > 0
    if int(v.sum()) < 3:
        return pitch
    cur = pitch.copy()
    idx = np.where(v)[0]
    # two passes: the second uses the already-cleaned track as reference, which
    # fixes isolated octave slips at the very edge of a voiced run (few neighbours)
    for _ in range(2):
        logf = np.zeros(pitch.size)
        logf[v] = np.log2(cur[v])
        ref = logf.copy()
        for j in idx:
            lo = max(0, j - 7); hi = min(pitch.size, j + 8)
            win = logf[lo:hi][v[lo:hi]]
            if win.size:
                m = float(np.median(win))
                val = logf[j]
                while val - m > 0.5:
                    val -= 1.0
                while m - val > 0.5:
                    val += 1.0
                ref[j] = val
        cur = np.zeros(pitch.size)
        cur[v] = 2.0 ** ref[v]
    return cur


def acf_pitch(sig: np.ndarray, fps: int, **kwargs) -> np.ndarray:
    """Pitch estimation with auto-correlation (legacy, no longer used)."""
    ptch = np.zeros(sig.shape[1])
    acf_seg = np.array([ast.acf(__s) for __s in np.atleast_2d(sig).T])
    first_zero_d_acf = np.argmax(np.diff(acf_seg<0), axis=1)
    n_perseg = sig.shape[0]
    

    for i, (fzda, acs) in enumerate(zip(first_zero_d_acf, acf_seg)):
        if fzda > 0:
            max_acf = np.max(acs[fzda:])
            max_idx = np.argmax(acs==max_acf)

            ptch[i] = fps/max_idx

    #Detect artifacts       
            R = np.mod(n_perseg,fps/(ptch[i]*2))
            p = fps/(ptch[i]*2)
            if R < p/2:
                ptch[i] = ptch[i] - p * np.sin(2 * np.pi *  R / ( 2 *p)) /(n_perseg)
            else:
                ptch[i] = ptch[i] - p * np.sin(2 * np.pi * (p - R)/(2 * p)) /(n_perseg)

            if ptch[i] > 1720:
                ninterpol=10
                sec = range(0,sig.data.shape[0])
                secnew = np.arange(0,sig.data.shape[0],1/ninterpol)
                org = sig[:,i]
                tck =  interpolate.splrep(sec , org, s=0)
                f =interpolate.splev(secnew, tck, der=0)    

                amp = np.zeros(ninterpol*2)
                for j in range(0,ninterpol*2):
                    delay = int(fps/ptch[i])*ninterpol-ninterpol+j
                    f1 = np.concatenate((np.zeros(delay),f[0:f.size-delay]))
                    amp[j] = f/np.linalg.norm(f) @ f1/np.linalg.norm(f1)

                maxa=np.max(amp)
                maxw=np.argmax(amp==maxa)
    
                ptch[i] = fps/(max_idx-1+maxw/ninterpol)    
    return ptch

def wavelet(sig: np.array, fps: int, waveletnum: int) -> float:
    """Wavelet Transform"""


class pitch_type:
    
    def __init__(self, ptype: str, pstart: int, pstop: int, pa1: float, pa2: float) -> None:
               
        """Possible pitch types"""
        self.pitch_types = ('note', 'pause', 'transient', 'vibrato', 'melisma')
        """Amount of types"""
        ntypes = 5
            
        self.type = ptype
        self.start = pstart
        self.stop = pstop
        self.arg1 = pa1
        self.arg2 = pa2

