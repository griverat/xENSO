"""
ioos_pkg_skeleton

My awesome ioos_pkg_skeleton
"""

from typing import Optional

import xarray as xr
from scipy.ndimage import convolve1d

from .utils import _check_dimensions


def compute_climatology(
    data: xr.DataArray,
    base_period: tuple = (None, None),
):
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
    climatology: Optional[xr.DataArray] = None,
    base_period: Optional[tuple[str, str]] = None,
):
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


def xconvolve(data: xr.DataArray, kernel: xr.DataArray, dim: Optional[str] = None):
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
