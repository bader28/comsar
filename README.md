# comsar — Computational Music and Sound Archiving

**comsar** is a Python toolkit for content-based analysis of music and sound,
with a focus on ethnomusicology (timbre, pitch/melody, tonal systems, rhythm and
corpus visualisation with Self-Organizing Maps).

It is the top layer of a three-package stack maintained by **Rolf Bader**:

| Package | Role | Repository |
|---|---|---|
| [**apollon**](https://codeberg.org/rbader/apollon) | Backbone: audio I/O, feature extraction, HMM, SOM (C-accelerated) | `codeberg.org/rbader/apollon` |
| [**chainsaddiction**](https://codeberg.org/rbader/chainsaddiction) | Poisson Hidden-Markov-Models in C (used for rhythm / time-series) | `codeberg.org/rbader/chainsaddiction` |
| [**comsar**](https://codeberg.org/rbader/comsar) | High-level music analysis (`PitchTrack`, `TimbreTrack`, tonal systems, SOM) | `codeberg.org/rbader/comsar` |

This fork combines the modern comsar code base with the pitch-tracking and
tonal-system analysis (`PitchTrack`) merged back from the older comsar version.

> **Distribution names.** The original PyPI names `apollon`, `comsar` and
> `chainsaddiction` belong to the upstream authors. This fork is published as
> `bader-apollon`, `bader-comsar` and `bader-chainsaddiction`. The **import**
> names are unchanged: you still write `import comsar`, `import apollon`, etc.

## Installation (no compiler needed)

Pre-compiled wheels are provided for **Windows, macOS and Linux**
(CPython 3.9–3.13). Installing comsar pulls apollon and chainsaddiction
automatically:

```
pip install bader-comsar
```

To also run the example notebooks:

```
pip install bader-comsar jupyterlab
```

For source builds, step-by-step Windows/macOS/Linux instructions, and
troubleshooting, see the **[full manual](docs/MANUAL.md)**.

## Quick start

```python
from comsar import PitchTrack, TimbreTrack

# Pitch / melody / tonal-system analysis
pt = PitchTrack()
result = pt.extract("my_audio.wav")
print(result.features)          # pandas DataFrame

# Timbre feature track
tt = TimbreTrack()
timbre = tt.extract("my_audio.wav")
```

Runnable notebooks are in [`examples/`](examples/).

## Documentation

* **[docs/MANUAL.md](docs/MANUAL.md)** — full manual: architecture,
  installation, module reference, example walkthrough, troubleshooting,
  and how releases/wheels are built.
* **[examples/](examples/)** — Jupyter notebooks for the main workflows.

## License

BSD-3-Clause. Original work © Michael Blaß; this fork maintained by Rolf Bader.
