"""comsar/tracks/untilities.py -- Utilities
License: BSD-3-Clasuse
Copyright (C) 2020, Michael Blaß, michael.blass@uni-hamburg.de
"""
import pathlib
import pickle
from typing import ClassVar, Type, TypeVar, Union

from dataclasses import dataclass
import numpy as np
import pandas as pd

from apollon import io
from apollon import container
from apollon import signal
from apollon.tools import standardize
from apollon import types


T = TypeVar('T')

@dataclass
class TrackMeta(container.Params):
    """Track meta data."""
    _schema: ClassVar[types.Schema] = None
    version: str
    time_stamp: str
    source: str


@dataclass
class TrackParams(container.Params):
    """Track parameter base class."""
    _schema: ClassVar[types.Schema] = None


@dataclass
class TimbreTrackParams(TrackParams):
    """Parameter set for TimbreTrack"""
    stft: signal.container.StftParams
    corr_dim: signal.container.CorrDimParams
    corr_gram: signal.container.CorrGramParams

@dataclass
class PitchTrackParams(TrackParams):
    """Parameter set for TimbreTrack"""
    segmentation: signal.container.StftParams


@dataclass
class TonalSystemParams(container.Params):
    """Parameter set for Tonal System analysis"""
    _schema: ClassVar[types.Schema] = io.json.load_schema('TonalSystem')
    dcent: int = 1
    dts: float = 0.1
    minlen: int = 3
    mindev: int = 60
    noctaves: int = 8
    f0: float = 27.5

@dataclass
class ngramParams(container.Params):
    """Parameter set for n-gram analysis"""
    _schema: ClassVar[types.Schema] = io.json.load_schema('ngram')
    minnotelength: int = 10
    ngram: int = 3
    ngcentmin: int = 0
    ngcentmax: int = 1200
    nngram: int = 10

class TrackResult:
    """Provide track results."""
    def __init__(self, meta: TrackMeta, params: TrackParams,
                 data: pd.DataFrame) -> None:
        self._meta = meta
        self._params = params
        self._data = data

    @property
    def data(self) -> np.ndarray:
        """Return the raw data array."""
        return self._data.to_numpy()

    @property
    def features(self) -> pd.DataFrame:
        """Extracted feautures."""
        return self._data

    @property
    def features_names(self) -> list:
        """Name of each feature."""
        return self._data.columns.to_list()

    @property
    def z_score(self) -> pd.DataFrame:
        """Z-score of extracted features."""
        return standardize(self.features)

    def to_csv(self, path: Union[str, pathlib.Path]) -> None:
        """Serialize features to csv file.

        This does not save parameters, and meta data.

        Args:
            path:  Destination path.
        """
        self._data.to_csv(path)

    def to_dict(self) -> dict:
        """Serialize TrackResults to dictionary."""
        return {'meta': self._meta.to_dict(),
                'params': self._params.to_dict(),
                'data': self._data.to_dict()}

    def to_json(self, path: Union[str, pathlib.Path]) -> None:
        """Serialize TrackResults to JSON."""
        io.json.dump(self.to_dict(), path)


    def to_pickle(self, path: Union[str, pathlib.Path]) -> None:
        """Serialize Track Results to pickle."""
        path = pathlib.Path(path)
        with path.open('wb') as fobj:
            pickle.dump(self, fobj)

    @classmethod
    def read_json(cls: Type[T], path: Union[str, pathlib.Path]) -> T:
        """Read TrackResults form json."""
        raw = io.json.load(path)
        meta = TrackMeta.from_dict(raw['meta'])
        params = TimbreTrackParams.from_dict(raw['params'])
        data = pd.DataFrame(raw['data'])
        return cls(meta, params, data)

    @classmethod
    def read_pickle(cls: Type[T], path: Union[str, pathlib.Path]) -> T:
        """Read pickled TrackResults."""
        path = pathlib.Path(path)
        with path.open('rb') as fobj:
            return pickle.load(fobj)
