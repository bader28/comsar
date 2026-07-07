# comsar — Manual

Computational Music and Sound Archiving: a manual for the
**apollon / chainsaddiction / comsar** stack.

- [1. Overview and architecture](#1-overview-and-architecture)
- [2. Installation](#2-installation)
  - [2.1 The easy way — install from PyPI (no compiler)](#21-the-easy-way--install-from-pypi-no-compiler)
  - [2.2 Recommended: use a virtual environment](#22-recommended-use-a-virtual-environment)
  - [2.3 Windows — step by step](#23-windows--step-by-step)
  - [2.4 macOS — step by step](#24-macos--step-by-step)
  - [2.5 Linux — step by step](#25-linux--step-by-step)
  - [2.6 Building from source](#26-building-from-source)
  - [2.7 Jupyter](#27-jupyter)
- [3. Troubleshooting](#3-troubleshooting)
- [4. Quick start](#4-quick-start)
- [5. API reference](#5-api-reference)
  - [5.1 comsar.PitchTrack](#51-comsarpitchtrack)
  - [5.2 comsar.TimbreTrack](#52-comsartimbretrack)
  - [5.3 comsar.tracks.utilities](#53-comsartracksutilities)
  - [5.4 comsar.tracks.helpers](#54-comsartrackshelpers)
  - [5.5 apollon (backbone)](#55-apollon-backbone)
  - [5.6 chainsaddiction (Poisson HMM)](#56-chainsaddiction-poisson-hmm)
- [6. Example notebooks](#6-example-notebooks)
- [7. Command-line scripts](#7-command-line-scripts)
- [8. For maintainers: releasing and building wheels](#8-for-maintainers-releasing-and-building-wheels)
- [9. What changed in this fork](#9-what-changed-in-this-fork)

---

## 1. Overview and architecture

The stack has three packages, each in its own repository on Codeberg:

```
        ┌──────────────────────────────────────────────┐
        │  comsar   (high-level music analysis)         │
        │  PitchTrack · TimbreTrack · tonal systems ·   │
        │  n-grams · SOM helpers                         │
        └───────────────┬──────────────────────────────┘
                        │ depends on
        ┌───────────────▼──────────────────────────────┐
        │  apollon   (backbone)                          │
        │  audio I/O · signal features (C) · SOM (C) ·   │
        │  HMM · segmentation · onsets                   │
        └───────────────┬──────────────────────────────┘
                        │ depends on
        ┌───────────────▼──────────────────────────────┐
        │  chainsaddiction   (Poisson HMM, pure C)       │
        └──────────────────────────────────────────────┘
```

- **chainsaddiction** — discrete-time Hidden Markov Models with Poisson-distributed
  latent variables, implemented in C as a NumPy extension. Used for time-series /
  rhythm modelling.
- **apollon** — the analysis backbone: reading audio, extracting spectral and
  temporal features (parts written in C for speed), Self-Organizing Maps (C),
  Hidden Markov Models, onset detection and segmentation.
- **comsar** — the layer researchers use directly. It provides `PitchTrack`
  (fundamental frequency, melody, note and tonal-system detection, n-grams) and
  `TimbreTrack` (timbre feature tracks), plus helpers to visualise corpora with
  Self-Organizing Maps.

> **Import names vs. distribution names.** On PyPI the packages are called
> `bader-apollon`, `bader-chainsaddiction` and `bader-comsar` (the plain names
> belong to the upstream projects). In Python you still import them under their
> normal names: `import apollon`, `import chainsaddiction`, `import comsar`.

---

## 2. Installation

### 2.1 The easy way — install from PyPI (no compiler)

Pre-compiled **wheels** are provided for Windows, macOS and Linux and CPython
3.9–3.13. A wheel already contains the compiled C code, so **no C compiler is
required**. Installing comsar automatically pulls apollon and chainsaddiction:

```
pip install bader-comsar
```

That single command is all most users need.

### 2.2 Recommended: use a virtual environment

A virtual environment keeps this stack isolated from other Python projects and
avoids version clashes. Create one per project:

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install bader-comsar
```

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install bader-comsar
```

Verify the installation:
```bash
python -c "import comsar, apollon, chainsaddiction; print('comsar', comsar.__version__)"
```

### 2.3 Windows — step by step

1. Install **Python 3.9–3.13** from <https://www.python.org/downloads/windows/>
   (tick *"Add python.exe to PATH"*). The `py` launcher is installed with it.
2. Open **PowerShell** and create a virtual environment (see 2.2).
3. `pip install bader-comsar`.

That is all — the wheels contain the compiled code, so **Visual Studio / MSVC is
not needed** for a normal installation. (You only need a compiler if you choose
to build from source, see 2.6.)

### 2.4 macOS — step by step

1. Install **Python 3.9–3.13** (from python.org or via Homebrew:
   `brew install python`).
2. Create a virtual environment (see 2.2).
3. `pip install bader-comsar`.

Wheels are built as macOS *universal2* (Apple Silicon **and** Intel), so no Xcode
is required for a normal installation.

### 2.5 Linux — step by step

1. Use your distribution's Python 3.9–3.13 (and `python3-venv`).
2. Create a virtual environment (see 2.2).
3. `pip install bader-comsar`.

`manylinux` wheels are provided; no compiler needed.

### 2.6 Building from source

You only need this if you are developing the packages or a wheel is not available
for your platform/Python version. You then need a C compiler and NumPy:

| OS | Compiler |
|---|---|
| Windows | [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (MSVC) |
| macOS | Xcode Command Line Tools: `xcode-select --install` |
| Linux | `gcc` (e.g. `sudo apt install build-essential`) |

Then, respecting the dependency order (chainsaddiction → apollon → comsar):

```bash
pip install "git+https://codeberg.org/rbader/chainsaddiction.git"
pip install "git+https://codeberg.org/rbader/apollon.git"
pip install "git+https://codeberg.org/rbader/comsar.git"
```

or, from a local clone of a repository, `pip install .` in its root.

### 2.7 Jupyter

The examples are Jupyter notebooks:

```bash
pip install jupyterlab
jupyter lab
```

Make sure Jupyter runs in the **same virtual environment** in which you installed
comsar (start `jupyter lab` from the activated venv), otherwise the kernel will
not find the packages.

---

## 3. Troubleshooting

**`ModuleNotFoundError: No module named 'comsar'` even though you installed it.**
Check that Jupyter/Python runs in the environment where you installed comsar:
`python -c "import sys; print(sys.executable)"`.

**An old version is imported instead of the installed one — the dreaded
`PYTHONPATH`.**
If the environment variable `PYTHONPATH` points at a source checkout (e.g.
`C:\...\comsar\src`), that path is prepended to `sys.path` in *every* Python,
including virtual environments, and shadows the installed package. Symptoms:
`comsar.__file__` points to a `...\src\comsar\__init__.py` rather than to
`...\site-packages\comsar`. Fix: remove those entries from `PYTHONPATH`, or start
Python with a cleared variable:
```powershell
$env:PYTHONPATH=""      # PowerShell, current session
```
```bash
PYTHONPATH= python ...  # bash, single command
```

**`error: Microsoft Visual C++ 14.0 or greater is required` (Windows, source
build).** You are building from source without a compiler. Either install the
C++ Build Tools (see 2.6) *or* — much simpler — install the wheel:
`pip install bader-comsar`.

**`Schema ... not found` from apollon.** This was a packaging bug in older
apollon builds where the JSON schema data files were missing from the wheel. It
is fixed in this fork (the schemas ship inside the wheel). Upgrade:
`pip install -U bader-apollon`.

**`ImportError: cannot import name 'derivative' from 'scipy.misc'`.** Removed in
SciPy ≥ 1.12. Fixed in this fork's `comsar` (the unused import was removed).
Upgrade `bader-comsar`.

**NumPy ABI errors (`numpy.dtype size changed`).** The wheels are built against
NumPy 2.x, which is runtime-compatible with NumPy ≥ 1.23. If you see ABI errors,
upgrade NumPy: `pip install -U numpy`.

**Name confusion with the upstream packages.** If you previously installed the
upstream `apollon` / `comsar` / `chainsaddiction`, uninstall them first to avoid
two copies of the same import name:
```
pip uninstall apollon comsar chainsaddiction
pip install bader-comsar
```

---

## 4. Quick start

```python
from comsar import PitchTrack, TimbreTrack

# --- Pitch / melody / tonal system ---------------------------------------
pt = PitchTrack()
result = pt.extract("my_audio.wav")     # -> TrackResult
print(result.features)                  # pandas DataFrame with 'Pitch', 'SPL'
result.to_csv("pitch.csv")

# --- Timbre feature track -------------------------------------------------
tt = TimbreTrack()
timbre = tt.extract("my_audio.wav")
print(timbre.features)
```

`extract()` accepts either a path to a WAV file (as above) or a NumPy array plus
a sample rate: `pt.extract(signal_array, sample_rate, source_name)`.

---

## 5. API reference

### 5.1 comsar.PitchTrack

`PitchTrack(seg_params=None, tonalsystem_params=None, ngram_params=None)`

Fundamental-frequency / melody / tonal-system analysis.

- **`extract(input1, input2=None, input3=None) -> TrackResult`**
  Extract the pitch (auto-correlation based f0) and sound-pressure-level track.
  `input1` is a WAV path (string) or a signal array; for an array pass the sample
  rate as `input2` and an optional source label as `input3`.
- **`extract_TonalSystem(data, dcent, dts, minlen, mindev, noctaves, f0)`**
  From a sequence of segment frequencies, detect notes and estimate the tonal
  system by correlating the accumulated pitch histogram against 900+ theoretical
  scales (bundled as `scales.csv`). Returns a tuple
  `(c, co, maxf, retNames, retScale, retValue, retCorr, nnotes, notes, cn)`:
  cumulated spectrum, one-octave spectrum, strongest frequency, names/values of
  the ten best-matching scales, note count, note list, and per-note histogram.
- **`extract_ngram(notes, nnotes, dcent, minnotelength, ngram, ngcentmin, ngcentmax, nngram)`**
  Compute interval n-grams (2- to 5-grams) over detected notes — a melodic
  fingerprint. Returns the most frequent n-grams and the notes used.

Parameter objects (`from comsar.tracks.utilities import ...`):
`PitchTrackParams`, `TonalSystemParams` (`dcent, dts, minlen, mindev, noctaves,
f0`), `ngramParams` (`minnotelength, ngram, ngcentmin, ngcentmax, nngram`).

### 5.2 comsar.TimbreTrack

`TimbreTrack(...)` — computes a track of timbre/spectral features (via apollon's
STFT and correlation-dimension features) from audio. `extract(path_or_array,
...)` returns a `TrackResult`.

### 5.3 comsar.tracks.utilities

- **`TrackResult`** — wraps extracted features together with metadata and
  parameters. Useful methods/properties: `.features` (DataFrame), `.data`
  (ndarray), `.features_names`, `.z_score`, `.to_csv(path)`, `.to_json(path)`,
  `.to_pickle(path)`, and the class methods `read_json`, `read_pickle`.
- **Parameter dataclasses**: `TrackMeta`, `TrackParams`, `TimbreTrackParams`,
  `PitchTrackParams`, `TonalSystemParams`, `ngramParams`.

### 5.4 comsar.tracks.helpers

Plotting / analysis helpers for Self-Organizing Maps: `init_pca`, `match_counts`,
`plot_counts`, `plot_umatrix`, `plot_component`, `plot_feature_importance`,
`mean_feat_dist`, `unit_info`, and more. Used by the SOM example notebooks.

### 5.5 apollon (backbone)

Key entry points used by comsar and the notebooks:

- `apollon.audio.AudioFile` — read audio files.
- `apollon.segment` / `Segmentation` — split signals into analysis frames.
- `apollon.signal` — `container`, `features` (spectral features, many C-accelerated),
  `spectral.StftSegments`, `tools`.
- `apollon.som.som.IncrementalMap` and `apollon.som.utilities` — Self-Organizing
  Maps (distance computation in C).
- `apollon.hmm` — Hidden Markov Models.
- `apollon.io.io` — `save_to_pickle`, `load_from_pickle`, JSON I/O.

### 5.6 chainsaddiction (Poisson HMM)

`import chainsaddiction`; the compiled submodules `chainsaddiction.poishmm` and
`chainsaddiction.utils` provide Poisson HMM fitting (forward–backward, EM). Used
under the hood for time-series / rhythm modelling.

---

## 6. Example notebooks

See [`examples/`](../examples/) and its
[README](../examples/README.md). The notebooks read audio and metadata from
paths that point to the original research collections — change those paths to
your own files before running. Overview:

| Notebook | Workflow |
|---|---|
| `Feature_Extract_From_Wav.ipynb` | Low-level feature extraction |
| `TimbreTrack_Extract_Features.ipynb` | Timbre feature track |
| `PitchTrack_f0_Extract.ipynb` | f0 / pitch track |
| `PitchTrack_Melody.ipynb` | Melody / note segmentation |
| `PitchTrack_Note_TonalSystem_Extract.ipynb` | Notes + tonal system / scale |
| `SOM.ipynb` | Self-Organizing Map on features |
| `TimbreTrack_SOM.ipynb` | Timbre + SOM corpus visualisation |

---

## 7. Command-line scripts

The package also ships a few argparse-based scripts under `comsar.cli`
(advanced / batch use), runnable with `python -m`:

```
python -m comsar.cli.comsar_features --help
python -m comsar.cli.apollon_hmm --help
python -m comsar.cli.apollon_onsets --help
python -m comsar.cli.apollon_position --help
```

---

## 8. For maintainers: releasing and building wheels

The canonical repositories live on **Codeberg**; a mirror on **GitHub**
(`github.com/bader28/...`) is used only as a build machine, because GitHub
Actions provides free Windows, macOS and Linux runners.

**Remotes** (already configured in each local clone):
```
origin  -> https://codeberg.org/rbader/<pkg>.git   (canonical)
github  -> https://github.com/bader28/<pkg>.git    (CI / wheel builder)
```

**How a release produces wheels for everyone:**

1. One-time PyPI setup per project: create the project (or a *pending publisher*)
   on PyPI and add a **Trusted Publisher** pointing at GitHub
   (owner `bader28`, repo `<pkg>`, workflow `wheels.yml`, environment `pypi`).
   No API token is stored anywhere.
2. Push the code to both remotes: `git push origin main` and `git push github main`.
3. Tag a release and push the tag to **github** (that triggers the build):
   ```
   git tag v0.0.3
   git push github v0.0.3
   ```
4. The GitHub Actions workflow `.github/workflows/wheels.yml` uses
   [cibuildwheel](https://cibuildwheel.readthedocs.io/) to build wheels for
   Windows/macOS/Linux × CPython 3.9–3.13, builds the sdist, and publishes
   everything to PyPI via Trusted Publishing.
5. Users then simply `pip install bader-<pkg>` and receive a pre-compiled wheel.

`comsar` is pure Python (a single universal wheel); `apollon` and
`chainsaddiction` contain C extensions and produce one wheel per
platform/Python-version.

You can also build a wheel locally (needs a compiler):
```
pip install build
python -m build          # -> dist/*.whl and dist/*.tar.gz
```

---

## 9. What changed in this fork

Relative to the upstream `ifsm/apollon`, `teagum/chainsaddiction` and
`ifsm/comsar`:

- **Distribution renamed** to `bader-apollon`, `bader-chainsaddiction`,
  `bader-comsar` (import names unchanged). `__init__` version lookups and the
  inter-package dependencies were updated accordingly.
- **comsar merge**: the modern comsar code base was kept as the base (its more
  advanced `TimbreTrack`), and the full **`PitchTrack`** (f0, note and
  tonal-system detection, n-grams), the SOM `helpers`, `scales.csv`, the JSON
  schemas and the `cli` scripts were merged back from the older comsar version.
  `PitchTrack` and `TimbreTrack` are re-exported at the top level
  (`from comsar import PitchTrack, TimbreTrack`), and a `comsar._tracks`
  compatibility shim keeps legacy notebook imports working.
- **Install/robustness fixes**:
  - apollon's JSON **schema files now ship inside the wheel** (previously missing
    → `Schema ... not found`).
  - Removed the unused `from scipy.misc import derivative` (broke on SciPy ≥ 1.12).
  - `scales.csv` is loaded as **package data** instead of from the current working
    directory.
  - Fixed pandas chained-indexing / assignment (`scales.iloc[i][j]`) that would
    break under pandas 3 Copy-on-Write.
  - Corrected invalid `python_requires` syntax in `setup.cfg`.
  - Dropped `setuptools-scm` from chainsaddiction's build requirements (it failed
    when building without git tags).
- **CI**: added `cibuildwheel` configuration and a GitHub Actions workflow to
  build and publish multi-platform wheels (see section 8).

A few very old notebooks referenced an intermediate API
(`import comsar.Pitchtrack.…`) that no longer exists in any version; those need a
one-line import change to `from comsar.tracks.utilities import …`.
