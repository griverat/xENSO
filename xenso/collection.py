"""
The metrics collections of the CLIVAR ENSO Metrics Package in one call.

:func:`clivar_collection` computes the metrics of the ENSO_perf, ENSO_proc or
ENSO_tel collection of the CLIVAR ENSO Metrics Package
(https://github.com/CLIVAR-PRP/enso_metrics) for one model against one
reference dataset, with the default settings of each metric. This is not the
official implementation: use the original package for results that must be
comparable with published CLIVAR ENSO metrics.
"""

from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np
import xarray as xr

from . import diagnostics, metrics
from .stats import compare

Fields = Mapping[str, xr.DataArray]
Rows = dict[str, tuple]


def _diagnostic(function: Callable, *variables: str) -> Callable[[Fields, Fields], Rows]:
    """Scalar diagnostic: the metric is the absolute relative difference in %."""

    def compute(model, obs):
        model_result = function(*(model[v] for v in variables))
        obs_result = function(*(obs[v] for v in variables))
        value, _ = compare(model_result, obs_result)
        return {"": (value, model_result["value"], obs_result["value"])}

    return compute


def _rmse(function: Callable, *variables: str, **kwargs) -> Callable[[Fields, Fields], Rows]:
    """RMSE metric of fields ``(model[v], obs[v])`` for each variable in turn."""

    def compute(model, obs):
        fields = [field for v in variables for field in (model[v], obs[v])]
        return {"": (function(*fields, **kwargs)["value"], np.nan, np.nan)}

    return compute


def _teleconnection(variable: str, season: str) -> Callable[[Fields, Fields], Rows]:
    """Teleconnection map: correlation, RMSE and standard deviation ratio, as CLIVAR reports them."""

    def compute(model, obs):
        result = metrics.enso_teleconnection(
            model["sst"], obs["sst"], model[variable], obs[variable], season=season, variable=variable
        )
        return {
            "Corr": (result.correlation, np.nan, np.nan),
            "Rmse": (result.value, np.nan, np.nan),
            "Std": (result.std_ratio, np.nan, np.nan),
        }

    return compute


# metric name -> (variables needed, computation)
_METRICS: dict[str, tuple[tuple[str, ...], Callable[[Fields, Fields], Rows]]] = {
    "BiasPrLatRmse": (("pr",), _rmse(metrics.mean_state_rmse, "pr", along="lat")),
    "BiasPrLonRmse": (("pr",), _rmse(metrics.mean_state_rmse, "pr")),
    "BiasSstLonRmse": (("sst",), _rmse(metrics.mean_state_rmse, "sst")),
    "BiasTauxLonRmse": (("taux",), _rmse(metrics.mean_state_rmse, "taux")),
    "SeasonalPrLatRmse": (("pr",), _rmse(metrics.seasonal_cycle_rmse, "pr", along="lat")),
    "SeasonalPrLonRmse": (("pr",), _rmse(metrics.seasonal_cycle_rmse, "pr")),
    "SeasonalSstLonRmse": (("sst",), _rmse(metrics.seasonal_cycle_rmse, "sst")),
    "SeasonalTauxLonRmse": (("taux",), _rmse(metrics.seasonal_cycle_rmse, "taux")),
    "EnsoSstLonRmse": (("sst",), _rmse(metrics.enso_pattern_rmse, "sst")),
    "EnsoSstTsRmse": (("sst",), _rmse(metrics.enso_lifecycle_rmse, "sst")),
    "EnsoAmpl": (("sst",), _diagnostic(diagnostics.enso_amplitude, "sst")),
    "EnsoSeasonality": (("sst",), _diagnostic(diagnostics.enso_seasonality, "sst")),
    "EnsoSstSkew": (("sst",), _diagnostic(diagnostics.enso_skewness, "sst")),
    "EnsoDuration": (("sst",), _diagnostic(diagnostics.enso_duration, "sst")),
    "EnsoSstDiversity": (("sst",), _diagnostic(diagnostics.enso_diversity, "sst")),
    "EnsodSstOce": (("sst", "thf"), _diagnostic(diagnostics.ocean_driven_sst_change, "sst", "thf")),
    "EnsoFbSshSst": (("sst", "ssh"), _diagnostic(diagnostics.thermocline_feedback, "sst", "ssh")),
    "EnsoFbSstTaux": (("sst", "taux"), _diagnostic(diagnostics.bjerknes_feedback, "sst", "taux")),
    "EnsoFbSstThf": (("sst", "thf"), _diagnostic(diagnostics.heat_flux_feedback, "sst", "thf")),
    "EnsoFbTauxSsh": (("taux", "ssh"), _diagnostic(diagnostics.wind_ssh_feedback, "taux", "ssh")),
    "EnsoPrMapDjf": (("sst", "pr"), _teleconnection("pr", "DJF")),
    "EnsoPrMapJja": (("sst", "pr"), _teleconnection("pr", "JJA")),
    "EnsoSstMapDjf": (("sst",), _teleconnection("sst", "DJF")),
    "EnsoSstMapJja": (("sst",), _teleconnection("sst", "JJA")),
}

COLLECTIONS: dict[str, list[str]] = {
    "ENSO_perf": [
        "BiasPrLatRmse",
        "BiasPrLonRmse",
        "BiasSstLonRmse",
        "BiasTauxLonRmse",
        "SeasonalPrLatRmse",
        "SeasonalPrLonRmse",
        "SeasonalSstLonRmse",
        "SeasonalTauxLonRmse",
        "EnsoSstLonRmse",
        "EnsoSstTsRmse",
        "EnsoAmpl",
        "EnsoSeasonality",
        "EnsoSstSkew",
        "EnsoDuration",
        "EnsoSstDiversity",
    ],
    "ENSO_proc": [
        "BiasSstLonRmse",
        "BiasTauxLonRmse",
        "EnsoAmpl",
        "EnsoSeasonality",
        "EnsoSstLonRmse",
        "EnsoSstSkew",
        "EnsodSstOce",
        "EnsoFbSshSst",
        "EnsoFbSstTaux",
        "EnsoFbSstThf",
        "EnsoFbTauxSsh",
    ],
    "ENSO_tel": [
        "EnsoAmpl",
        "EnsoSeasonality",
        "EnsoSstLonRmse",
        "EnsoPrMapDjf",
        "EnsoPrMapJja",
        "EnsoSstMapDjf",
        "EnsoSstMapJja",
    ],
}


def clivar_collection(
    model: Fields,
    obs: Fields,
    collection: Literal["ENSO_perf", "ENSO_proc", "ENSO_tel"] = "ENSO_perf",
) -> xr.Dataset:
    """
    Compute the metrics of a CLIVAR ENSO metrics collection.

    Scalar diagnostics (amplitude, seasonality, feedbacks, ...) are compared
    with the absolute relative difference in %, as in CLIVAR; the other
    metrics are RMSEs. Teleconnection maps give three metrics each: the
    pattern correlation (``Corr``), RMSE (``Rmse``) and ratio of standard
    deviations (``Std``). Metrics whose variables are missing from ``model``
    or ``obs`` are skipped.

    Parameters
    ----------
    model, obs
        Mappings, e.g. Datasets, of monthly fields with time, lat and lon
        dimensions, named "sst" (°C), "pr", "taux", "ssh" and "thf" (net
        surface heat flux, positive into the ocean). Fields of the same name
        must be on the same grid in both. Use ocean-only fields (e.g. "tos",
        or "ts" masked with "sftlf"): land points are averaged like any other.
    collection
        "ENSO_perf" (performance), "ENSO_proc" (processes) or "ENSO_tel"
        (teleconnections).

    Returns
    -------
    Dataset along a ``metric`` dimension with the metric as ``value`` and,
    for scalar diagnostics, the model and observed diagnostics as ``model``
    and ``obs``.
    """
    if collection not in COLLECTIONS:
        raise ValueError(f"unknown collection {collection!r}, expected one of {sorted(COLLECTIONS)}")

    names, rows = [], []
    for metric in COLLECTIONS[collection]:
        variables, compute = _METRICS[metric]
        if not all(v in model and v in obs for v in variables):
            continue
        for suffix, row in compute(model, obs).items():
            names.append(metric + suffix)
            rows.append([float(x) for x in row])

    table = np.array(rows, dtype=float).reshape(-1, 3)
    return xr.Dataset(
        {name: ("metric", table[:, i]) for i, name in enumerate(["value", "model", "obs"])},
        coords={"metric": names},
        attrs={"collection": collection},
    )
