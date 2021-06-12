"""
ioos_pkg_skeleton is not a real package, just a set of best practices examples.
"""

from .core import compute_anomaly, compute_climatology, xconvolve

__all__ = ["compute_climatology", "compute_anomaly", "xconvolve"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
