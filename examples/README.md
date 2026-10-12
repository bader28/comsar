# comsar example notebooks

These Jupyter notebooks demonstrate the main workflows of the
apollon / chainsaddiction / comsar stack. They are documentation and starting
points — copy one and adapt it to your own material.

## Prerequisites

```
pip install bader-comsar jupyterlab
jupyter lab
```

## Data files

Most notebooks are **self-contained**: the small analysis files they need ship
next to them, so they run without the original audio (the `.wav` collections are
not redistributed for size / licensing reasons).

| File | Used by |
|---|---|
| `CAM-5_SARA PANH.wav` | `PitchTrack_Example` (a solo vocal recording) |
| `CHI-109_Luguhu_Hulusheng.wav` | `TimbreTrack_SimpleExample` |
| `HipHop_features.csv` | `SOM_HipHop` (pre-extracted collection features) |
| `Luguhu_roughness.csv`, `Luguhu_partials.csv` | `WaveletRoughness_Example` |

## Notebooks

| Notebook | What it shows |
|---|---|
| `TimbreTrack_SimpleExample.ipynb` | **Start here.** Extract the seven psychoacoustic timbre features and show an interactive player (grey waveform + coloured feature tracks, play button, cursor); includes a section on the two integrated wavelet-roughness measures. |
| `PitchTrack_Example.ipynb` | The full pitch pipeline on one vocal piece: **f0 extraction → note / melody detection → tonal system (scale) → melodic n-grams**, each step explained, ending in the interactive `pitch_player`. |
| `SOM_HipHop.ipynb` | A **Self-Organizing Map** of a Hip Hop collection, clustering tracks by era from pre-extracted `TimbreTrack` features — feature selection by name, year metadata, component planes. Runs with no audio. |
| `WaveletRoughness_Example.ipynb` | **Wavelet / Gabor roughness**: Bader/Helmholtz and Sethares sensory-dissonance curves plus exact partial frequencies and a partial-gram. Runs from the shipped CSVs, no audio. |

## Typical API entry points

```python
from comsar import PitchTrack, TimbreTrack, WaveletRoughness

pt = PitchTrack()                        # f0 (YIN by default; pitch_method='acf' for the legacy engine)
result = pt.extract("my_audio.wav")      # -> TrackResult (features as a DataFrame)

tt = TimbreTrack()
result = tt.extract("my_audio.wav")
```

See [`docs/MANUAL.md`](../docs/MANUAL.md) for the full API and background.
