"""xenso — ENSO indices and operations using xarray structures."""

from .core import compute_anomaly, compute_climatology, detrend, smooth, xconvolve
from .ecindex import ECindex
from .events import composite, detect_events, event_duration, event_windows, seasonal_series
from .regions import REGIONS, nino_regions, oni, region_mean, roni
from .stats import compare, linregress, rmse, split_regression

__all__ = [
    "compute_climatology",
    "compute_anomaly",
    "detrend",
    "smooth",
    "xconvolve",
    "ECindex",
    "seasonal_series",
    "detect_events",
    "event_windows",
    "composite",
    "event_duration",
    "REGIONS",
    "nino_regions",
    "region_mean",
    "oni",
    "roni",
    "linregress",
    "split_regression",
    "rmse",
    "compare",
]

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"
