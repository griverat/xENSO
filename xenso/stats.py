"""
Regressions and model-observation scores.

The definitions follow the CLIVAR ENSO Metrics Package
(https://github.com/CLIVAR-PRP/enso_metrics), which remains the reference
implementation. Differences from it are noted in each docstring.
"""

from typing import Literal

import numpy as np
import xarray as xr

Number = float | xr.DataArray


def linregress(
    y: xr.DataArray,
    x: xr.DataArray,
    dim: str = "time",
    sign: Literal["positive", "negative"] | None = None,
) -> xr.Dataset:
    """
    Ordinary least-squares regression of ``y`` on ``x`` along ``dim``.

    Vectorized over all other dimensions, so a map can be regressed onto an
    index. Pairs with a missing value are skipped.

    Parameters
    ----------
    y
        Dependent variable.
    x
        Independent variable, broadcastable against ``y``.
    dim
        Dimension along which the regression is computed.
    sign
        Only use the points where ``x > 0`` ("positive") or ``x < 0``
        ("negative"), as in CLIVAR's CustomLinearRegression.

    Returns
    -------
    Dataset with ``slope``, ``intercept`` and ``stderr`` (standard error of
    the slope, as in ``scipy.stats.linregress``).
    """
    valid = x.notnull() & y.notnull()
    if sign == "positive":
        valid = valid & (x > 0)
    elif sign == "negative":
        valid = valid & (x < 0)
    elif sign is not None:
        raise ValueError(f"sign must be 'positive', 'negative' or None, got {sign!r}")

    x, y = x.where(valid), y.where(valid)
    n = valid.sum(dim)
    dx = x - x.mean(dim)
    dy = y - y.mean(dim)
    sxx = (dx**2).sum(dim)
    sxy = (dx * dy).sum(dim)
    syy = (dy**2).sum(dim)

    slope = (sxy / sxx).where((n >= 2) & (sxx > 0))
    intercept = y.mean(dim) - slope * x.mean(dim)
    residual = (syy - slope * sxy).clip(min=0)
    stderr = np.sqrt(residual / (n - 2) / sxx).where(n > 2)
    return xr.Dataset({"slope": slope, "intercept": intercept, "stderr": stderr})


def split_regression(
    y: xr.DataArray,
    x: xr.DataArray,
    dim: str = "time",
) -> xr.Dataset:
    """
    Regress ``y`` on all, positive and negative values of ``x``.

    This is how the CLIVAR ENSO Metrics Package computes feedbacks such as
    the Bjerknes (zonal wind stress on SST) or heat-flux feedbacks, together
    with their asymmetry between El Niño and La Niña.

    Parameters
    ----------
    y
        Dependent variable, e.g. Niño 4 zonal wind stress anomalies.
    x
        Independent variable, e.g. Niño 3 SST anomalies.
    dim
        Dimension along which the regressions are computed.

    Returns
    -------
    Dataset with ``slope``, ``intercept`` and ``stderr`` along a ``branch``
    dimension ("all", "positive", "negative"), plus ``nonlinearity`` (negative
    minus positive slope) and ``nonlinearity_stderr`` (sum of both errors).
    """
    branches = ["all", "positive", "negative"]
    fits = [linregress(y, x, dim=dim, sign=None if b == "all" else b) for b in branches]
    result = xr.concat(fits, dim="branch").assign_coords(branch=branches)

    negative = result.sel(branch="negative", drop=True)
    positive = result.sel(branch="positive", drop=True)
    result["nonlinearity"] = negative.slope - positive.slope
    result["nonlinearity_stderr"] = negative.stderr + positive.stderr
    return result


def rmse(
    model: xr.DataArray,
    obs: xr.DataArray,
    dim: str | list[str],
    weights: xr.DataArray | None = None,
    centered: bool = False,
) -> xr.DataArray:
    """
    Root-mean-square difference between two fields, skipping missing values.

    Parameters
    ----------
    model, obs
        Fields on the same grid.
    dim
        Dimension(s) to reduce, e.g. "lon" for a zonal section.
    weights
        Optional weights, e.g. cos(latitude). They are renormalized over the
        points where both fields are valid.
    centered
        Remove the mean difference first. The CLIVAR metrics use the
        uncentered RMSE.
    """
    diff = model - obs

    def _mean(field):
        return field.mean(dim) if weights is None else field.weighted(weights).mean(dim)

    if centered:
        diff = diff - _mean(diff)
    return np.sqrt(_mean(diff**2))


def compare(
    model: Number,
    obs: Number,
    method: Literal[
        "difference", "ratio", "relative_difference", "abs_relative_difference"
    ] = "abs_relative_difference",
    model_err: Number | None = None,
    obs_err: Number | None = None,
) -> tuple[Number, Number | None]:
    """
    Turn a model and an observed diagnostic into a metric value.

    Parameters
    ----------
    model, obs
        Diagnostic values, e.g. the Niño 3.4 standard deviation.
    method
        "difference": model - obs
        "ratio": model / obs
        "relative_difference": (model - obs) / obs
        "abs_relative_difference": 100 * |model - obs| / |obs|, the default
        for scalar metrics in the CLIVAR collections.
    model_err, obs_err
        Optional errors on the diagnostics, propagated linearly. For ratios
        this is ``(|obs| * model_err + |model| * obs_err) / obs**2``, which
        matches CLIVAR's formula when both values are positive.

    Returns
    -------
    The metric value and its error (None if either input error is None).
    """
    if method == "difference":
        value = model - obs
    elif method == "ratio":
        value = model / obs
    elif method == "relative_difference":
        value = (model - obs) / obs
    elif method == "abs_relative_difference":
        value = 100 * abs((model - obs) / obs)
    else:
        raise ValueError(f"unknown method {method!r}")

    if model_err is None or obs_err is None:
        return value, None
    if method == "difference":
        error = model_err + obs_err
    else:
        error = (abs(obs) * model_err + abs(model) * obs_err) / obs**2
        if method == "abs_relative_difference":
            error = 100 * error
    return value, error
