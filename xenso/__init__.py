"""xenso — ENSO indices and operations using xarray structures."""

from .core import compute_anomaly, compute_climatology, xconvolve
from .ecindex import ECindex
from .regions import nino_regions

__all__ = [
    "compute_climatology",
    "compute_anomaly",
    "xconvolve",
    "ECindex",
    "nino_regions",
]

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"
