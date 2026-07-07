"""Backward-compatibility shim.

Older comsar versions kept the implementation under ``comsar._tracks``.
The current layout lives under ``comsar.tracks``. This package re-exports the
old import paths so that legacy notebooks keep working, e.g.::

    from comsar._tracks.utilities import ngramParams, TonalSystemParams
"""
import sys as _sys

from comsar.tracks import utilities, helpers
from comsar.tracks import _pitch as pitch
from comsar.tracks import _timbre as timbre

# Register the modules under their legacy dotted paths so that
# ``from comsar._tracks.<name> import ...`` resolves without separate files.
for _name, _mod in (("utilities", utilities), ("helpers", helpers),
                    ("pitch", pitch), ("timbre", timbre)):
    _sys.modules[f"{__name__}.{_name}"] = _mod

del _sys, _name, _mod
