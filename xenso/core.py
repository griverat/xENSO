"""
ioos_pkg_skeleton

My awesome ioos_pkg_skeleton
"""

from typing import Optional

import numpy as np
import requests
import xarray as xr
from scipy.ndimage import convolve1d


def compute_climatology(data: xr.DataArray, base_period: tuple):
    """
    Computes the seasonal mean of a DataArray that has a time
    dimension

    Parameters
    ----------
    data
    base_period
    """
    return data.sel(time=slice(*base_period)).groupby("time.month").mean()


def compute_anomaly(
    data: xr.DataArray,
    climatology: Optional[xr.DataArray] = None,
    base_period: Optional[tuple[str, str]] = None,
):
    """
    Computes the anomaly of a field in the time dimension
    """
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


def xconvolve(data, kernel, dim=None):
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


def meaning_of_life(n: int) -> np.ndarray:
    """Return the meaning of life n times."""
    matrix = (n, n)
    return np.ones(matrix) * 42


def meaning_of_life_url() -> str:
    """
    Fetch the meaning of life from http://en.wikipedia.org.
    """
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/Monty_Python's_The_Meaning_of_Life"
    r = requests.get(url)
    r.raise_for_status()
    j = r.json()
    return j["extract"]
