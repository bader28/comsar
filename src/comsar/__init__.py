"""comsar -- Computational Music and Sound Archiving.

Top-level convenience re-exports so that user code / notebooks can simply do::

    from comsar import PitchTrack, TimbreTrack
"""
try:
    from importlib.metadata import version as _version, PackageNotFoundError
except ImportError:                       # Python < 3.8 fallback
    from importlib_metadata import version as _version, PackageNotFoundError

try:
    # Distribution name on PyPI is "bader-comsar"; the import name stays "comsar".
    __version__ = _version("bader-comsar")
except PackageNotFoundError:              # e.g. running from a source checkout
    __version__ = "0.0.8"

# __version__ is set *before* importing the tracks subpackage, because
# comsar.tracks._pitch references ``comsar.__version__`` at runtime.
from .tracks import TimbreTrack, PitchTrack, WaveletRoughness, ImpulsePattern
from .viz import timbre_player, pitch_player

__all__ = ["TimbreTrack", "PitchTrack", "WaveletRoughness", "ImpulsePattern",
           "timbre_player", "pitch_player", "__version__"]
