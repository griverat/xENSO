"""
ioos_pkg_skeleton

My awesome ioos_pkg_skeleton
"""

from typing import Literal

import numpy as np
import xarray as xr
from scipy.ndimage import convolve1d

from .utils import _check_dimensions


def compute_climatology(
    data: xr.DataArray,
    base_period: tuple = (None, None),
) -> xr.DataArray:
    """
    Computes the seasonal mean of a DataArray that has a time
    dimension

    Parameters
    ----------
    data
    base_period
    """
    _check_dimensions(data)
    return data.sel(time=slice(*base_period)).groupby("time.month").mean()


def compute_anomaly(
    data: xr.DataArray,
    climatology: xr.DataArray | None = None,
    base_period: tuple[str, str] | None = None,
) -> xr.DataArray:
    """
    Computes the anomaly of a field in the time dimension
    """
    _check_dimensions(data)
    if climatology is None:
        if base_period is None:
            raise ValueError(
                "You need to provide a climatology or",
                "the base period to compute it from the",
                "`compute_climatology` function",
            )
        else:
            climatology = compute_climatology(data, base_period)
    return data.groupby("time.month") - climatology


def xconvolve(
    data: xr.DataArray,
    kernel: xr.DataArray,
    dim: str | None = None,
) -> xr.DataArray:
    """
    Convolution using xarray data structures by using
    xr.apply_ufunc
    """
    res = xr.apply_ufunc(
        convolve1d,
        data,
        kernel,
        input_core_dims=[[dim], [dim]],
        exclude_dims={dim},
        output_core_dims=[[dim]],
    )
    res[dim] = data[dim]
    return res


_SMOOTHING_WEIGHTS = {
    "square": lambda half: np.ones(2 * half + 1),
    "triangle": lambda half: 1.0 + half - np.abs(np.arange(-half, half + 1)),
    "gaussian": lambda half: np.exp(-((4 * np.arange(-half, half + 1) / (2 * half + 1)) ** 2)),
}


def smooth(
    data: xr.DataArray,
    window: int = 5,
    method: Literal["triangle", "square", "gaussian"] = "triangle",
    dim: str = "time",
) -> xr.DataArray:
    """
    Weighted centred running mean, following the CLIVAR ENSO Metrics Package.

    Missing values are skipped and the weights renormalized; a point is set to
    NaN when more than half of its window is missing. The ``window // 2``
    points at each end are dropped, so the output is shorter than the input.

    Parameters
    ----------
    data
        DataArray to smooth.
    window
        Odd number of points in the running window.
    method
        Shape of the weights: "triangle" (e.g. 1-2-3-2-1), "square" (boxcar)
        or "gaussian". The CLIVAR gaussian kernel is off-centre by one point;
        the kernel used here is symmetric.
    dim
        Dimension along which to smooth.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    if method not in _SMOOTHING_WEIGHTS:
        raise ValueError(f"unknown method {method!r}, expected one of {sorted(_SMOOTHING_WEIGHTS)}")
    half = window // 2
    weights = xr.DataArray(_SMOOTHING_WEIGHTS[method](half), dims=["window"])

    windows = data.rolling({dim: window}, center=True).construct("window")
    valid = windows.notnull()
    smoothed = (windows * weights).sum("window") / weights.where(valid).sum("window")
    smoothed = smoothed.where(valid.sum("window") > half)
    return smoothed.isel({dim: slice(half, data.sizes[dim] - half)})


def detrend(
    data: xr.DataArray,
    dim: str = "time",
    keep_mean: bool = True,
) -> xr.DataArray:
    """
    Remove the least-squares linear trend along a dimension, skipping NaNs.

    The trend is fitted against the sample index rather than the coordinate
    values, so each time step is treated as equally spaced.

    Parameters
    ----------
    data
        DataArray to detrend.
    dim
        Dimension along which the trend is fitted.
    keep_mean
        Add the mean back after removing the trend, as done in the CLIVAR
        ENSO Metrics Package. If False the result has zero mean.
    """
    # fit against the sample index, like scipy.signal.detrend, so that months
    # of different lengths are equally spaced
    position = xr.DataArray(np.arange(data.sizes[dim]), dims=[dim])
    indexed = data.assign_coords({dim: position})
    coeffs = indexed.polyfit(dim, deg=1, skipna=True).polyfit_coefficients
    detrended = data - xr.polyval(position, coeffs)
    if keep_mean:
        detrended = detrended + data.mean(dim)
    return detrended.where(data.notnull())
