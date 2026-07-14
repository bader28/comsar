# comsar example notebooks

These Jupyter notebooks demonstrate the main workflows of the
apollon / chainsaddiction / comsar stack. They are provided as documentation and
starting points — copy one and adapt the audio paths to your own material.

## Prerequisites

Install the stack (see the [main README](../README.md) / the
[manual](../docs/MANUAL.md)) and Jupyter:

```
pip install bader-comsar jupyterlab
jupyter lab
```

## Audio and data files

The notebooks read audio (`.wav`) and metadata (`.csv`) from paths that point to
the original research collections and are **not** shipped with this repository
(size / licensing). Before running a notebook, change the file paths near the top
of the notebook to point to your own audio files. Any mono/stereo `.wav` works
for the feature/pitch/timbre extraction notebooks.

## Notebooks

| Notebook | What it shows |
|---|---|
| `TimbreTrack_SimpleExample.ipynb` | **Start here.** Extract the 7 timbre features and show an interactive player: grey waveform + coloured feature tracks, a play button and a cursor that follows the audio. Self-contained — the sample audio `CHI-109_Luguhu_Hulusheng.wav` ships next to it. |
| `WaveletRoughness_Example.ipynb` | Wavelet/Gabor roughness: Bader/Helmholtz and Sethares roughness per frame plus exact partial frequencies; player with both roughness curves and a partial-gram panel (grey = amplitude). |
| `Feature_Extract_From_Wav.ipynb` | Extract low-level audio features from a WAV file with apollon. |
| `TimbreTrack_Extract_Features.ipynb` | Compute a `TimbreTrack` (spectral/timbre feature track) from audio. |
| `PitchTrack_f0_Extract.ipynb` | The full pitch-track pipeline: **f0**, **impulse pattern**, **melody/notes** (`PitchTrack.notes`) and **tonal system** (`PitchTrack.tonal_system`), saved as CSV and shown together in the interactive `pitch_player` (waveform, f0, impulse lines, note bars and tonal-system reference lines, with play/cursor and horizontal zoom). |
| `PitchTrack_Melody.ipynb` | Turn a pitch track into a melody (note segmentation). |
| `PitchTrack_Note_TonalSystem_Extract.ipynb` | Detect notes and estimate the tonal system / scale (`extract_TonalSystem`). |
| `SOM.ipynb` | Train a Self-Organizing Map on extracted features (apollon SOM). |
| `TimbreTrack_SOM.ipynb` | Combine timbre features with a SOM for corpus visualisation. |

## Typical API entry points

```python
from comsar import PitchTrack, TimbreTrack

pt = PitchTrack()
result = pt.extract("my_audio.wav")      # -> TrackResult (features as DataFrame)

tt = TimbreTrack()
result = tt.extract("my_audio.wav")
```

See [`docs/MANUAL.md`](../docs/MANUAL.md) for the full API and analysis
background.
