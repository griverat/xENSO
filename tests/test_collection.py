import numpy as np
import pytest
import xarray as xr
from synthetic import TIME, event_series

import xenso

LAT = np.arange(-30.0, 31.0, 5.0)
LON = np.arange(0.0, 360.0, 5.0)


def expected_metrics(collection):
    maps = {"EnsoPrMapDjf", "EnsoPrMapJja", "EnsoSstMapDjf", "EnsoSstMapJja"}
    names = []
    for metric in xenso.COLLECTIONS[collection]:
        names += [metric + s for s in ("Corr", "Rmse", "Std")] if metric in maps else [metric]
    return names


@pytest.fixture(scope="module")
def fields():
    rng = np.random.default_rng(1)
    events = event_series().values[:, None, None]
    shape = (TIME.size, LAT.size, LON.size)

    def field(scale, noise):
        values = scale * events + noise * rng.normal(size=shape)
        return xr.DataArray(values, coords=[("time", TIME), ("lat", LAT), ("lon", LON)])

    return xr.Dataset(
        {
            "sst": field(1.0, 0.3),
            "pr": field(2.0, 0.5),
            "taux": field(-0.02, 0.005),
            "ssh": field(5.0, 1.0),
            "thf": field(-15.0, 5.0),
        }
    )


@pytest.mark.parametrize("collection", ["ENSO_perf", "ENSO_proc", "ENSO_tel"])
def test_identical_datasets(fields, collection):
    result = xenso.clivar_collection(fields, fields, collection)
    assert list(result.metric.values) == expected_metrics(collection)
    assert result.attrs["collection"] == collection
    assert result.value.notnull().all()
    # the correlation and standard deviation ratio of identical maps are 1, everything else 0
    ones = result.metric.str.endswith("Corr") | result.metric.str.endswith("Std")
    np.testing.assert_allclose(result.value.where(ones, drop=True), 1)
    np.testing.assert_allclose(result.value.where(~ones, drop=True), 0, atol=1e-10)


@pytest.mark.parametrize("collection", ["ENSO_perf", "ENSO_tel"])
def test_metric_types(fields, collection):
    result = xenso.clivar_collection(fields, fields, collection)
    by_type = {kind: result.sel(metric=result.type == kind) for kind in set(result.type.values)}
    # RMSE metrics compare whole profiles or maps: no single model or observed value
    rmse = by_type["rmse"]
    assert rmse.metric.str.endswith("Rmse").all()
    assert rmse.model.isnull().all() and rmse.obs.isnull().all()
    # scalar diagnostics give both values the metric is computed from
    diagnostics = by_type["relative difference (%)"]
    assert diagnostics.model.notnull().all() and diagnostics.obs.notnull().all()
    if collection == "ENSO_tel":
        assert by_type["correlation"].metric.str.endswith("Corr").all()
        assert by_type["std ratio"].metric.str.endswith("Std").all()


def test_diagnostics_and_relative_difference(fields):
    model = fields.assign(sst=2 * fields.sst)
    result = xenso.clivar_collection(model, fields, "ENSO_perf")
    amplitude = result.sel(metric="EnsoAmpl")
    np.testing.assert_allclose(amplitude.model, 2 * amplitude.obs)
    np.testing.assert_allclose(amplitude.value, 100)
    # scale-free diagnostics are unchanged
    np.testing.assert_allclose(
        result.value.sel(metric=["EnsoSeasonality", "EnsoSstSkew"]), 0, atol=1e-10
    )
    # RMSE metrics have no diagnostic values
    assert result.model.sel(metric="BiasSstLonRmse").isnull()


def test_missing_variables_are_skipped(fields):
    result = xenso.clivar_collection(fields, fields.drop_vars("ssh"), "ENSO_proc")
    assert "EnsoFbSshSst" not in result.metric
    assert "EnsoFbTauxSsh" not in result.metric
    assert "EnsoFbSstTaux" in result.metric


def test_unknown_collection(fields):
    with pytest.raises(ValueError):
        xenso.clivar_collection(fields, fields, "ENSO_all")
