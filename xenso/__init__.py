"""xenso — ENSO indices and operations using xarray structures."""

from .core import compute_anomaly, compute_climatology, detrend, smooth, xconvolve
from .diagnostics import (
    bjerknes_feedback,
    enso_amplitude,
    enso_diversity,
    enso_duration,
    enso_seasonality,
    enso_skewness,
    heat_flux_feedback,
    ocean_driven_sst_change,
    thermocline_feedback,
    wind_ssh_feedback,
)
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
    "enso_amplitude",
    "enso_seasonality",
    "enso_skewness",
    "enso_duration",
    "enso_diversity",
    "bjerknes_feedback",
    "heat_flux_feedback",
    "thermocline_feedback",
    "wind_ssh_feedback",
    "ocean_driven_sst_change",
]

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"
