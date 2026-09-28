## xENSO

[![Tests](https://github.com/DangoMelon/xENSO/actions/workflows/tests.yml/badge.svg)](https://github.com/DangoMelon/xENSO/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/DangoMelon/xENSO/graph/badge.svg?token=04AV0EO0MY)](https://codecov.io/gh/DangoMelon/xENSO)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Quick description

### Documentation and code

URLs for the docs and code.

### Installation

For `conda` users you can

```shell
conda install --channel conda-forge xenso
```

or, if you are a `pip` users

```shell
pip install xenso
```

### Example

```python
import xenso


xenso.meaning_of_life_url()
```


### Relation to the CLIVAR ENSO Metrics Package

Parts of xENSO (`xenso.events`, `xenso.stats`, `xenso.smooth`, `xenso.detrend`
and the boxes in `xenso.REGIONS`) re-implement diagnostics defined by the
[CLIVAR ENSO Metrics Package](https://github.com/CLIVAR-PRP/enso_metrics)
(Planton et al., 2021, [doi:10.1175/BAMS-D-19-0337.A](https://doi.org/10.1175/BAMS-D-19-0337.A))
with a vectorized xarray interface. They are not the official implementation:
results can differ in details, which are noted in the docstrings. Model
evaluations that need to be comparable with published CLIVAR ENSO metrics
should use the original package.

## Get in touch

Report bugs, suggest features or view the source code on [GitHub](https://github.com/ioos/ioos_pkg_skeleton/issues).


## License and copyright

ioos_pkg_skeleton is licensed under BSD 3-Clause "New" or "Revised" License (BSD-3-Clause).

Development occurs on GitHub at <https://github.com/DangoMelon/xENSO>.
