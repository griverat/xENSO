## xENSO

[![Tests](https://github.com/DangoMelon/xENSO/actions/workflows/tests.yml/badge.svg)](https://github.com/DangoMelon/xENSO/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/DangoMelon/xENSO/graph/badge.svg?token=04AV0EO0MY)](https://codecov.io/gh/DangoMelon/xENSO)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

ENSO indices, diagnostics and model-evaluation metrics using xarray structures.

- **Indices**: Niño regions, ONI, relative ONI and the E and C indices.
- **ENSO events**: detection, event composites and life cycles, durations.
- **ENSO metrics**: the diagnostics and model-observation metrics of the
  [CLIVAR ENSO Metrics Package](https://github.com/CLIVAR-PRP/enso_metrics)
  (amplitude, seasonality, feedbacks, patterns, teleconnections, ...).

### Installation

For `conda` users you can

```shell
conda install --channel conda-forge xenso
```

or, if you are a `pip` user

```shell
pip install xenso
```

### Example

```python
import xarray as xr
import xenso

sst = xr.open_dataarray("sst.nc")  # monthly SST with time, lat and lon

# Oceanic Niño Index
oni = xenso.oni(sst, base_period=("1991-01-01", "2020-12-31"))

# El Niño years, from the detrended and smoothed Niño 3.4 anomaly
nino34 = xenso.compute_anomaly(xenso.region_mean(sst, "nino34"), base_period=("1991", "2020"))
el_nino = xenso.detect_events(xenso.smooth(xenso.detrend(nino34)), kind="nino")

# ENSO amplitude of a model compared with observations, in %
value, error = xenso.compare(xenso.enso_amplitude(model_sst), xenso.enso_amplitude(sst))

# all the metrics of the CLIVAR ENSO performance collection
table = xenso.clivar_collection(model_ds, obs_ds, "ENSO_perf")
```

### Relation to the CLIVAR ENSO Metrics Package

`xenso.diagnostics`, `xenso.metrics`, `xenso.collection`, `xenso.events`,
`xenso.stats`, `xenso.smooth`, `xenso.detrend` and the boxes in
`xenso.REGIONS` re-implement definitions of the
[CLIVAR ENSO Metrics Package](https://github.com/CLIVAR-PRP/enso_metrics)
(Planton et al., 2021, [doi:10.1175/BAMS-D-19-0337.A](https://doi.org/10.1175/BAMS-D-19-0337.A))
with a vectorized xarray interface. They are not the official implementation:
results can differ in details, which are noted in the docstrings. Model
evaluations that need to be comparable with published CLIVAR ENSO metrics
should use the original package.

Inputs must be monthly, ocean-only fields (`tos`, or `ts` masked with
`sftlf`), and the model-observation metrics expect both datasets on the same
grid; CLIVAR regrids them to 1° beforehand, which is left to the user.

## Get in touch

Report bugs, suggest features or view the source code on [GitHub](https://github.com/DangoMelon/xENSO/issues).

## License and copyright

xENSO is licensed under BSD 3-Clause "New" or "Revised" License (BSD-3-Clause).

Development occurs on GitHub at <https://github.com/DangoMelon/xENSO>.
