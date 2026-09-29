Getting started
===============

Installation
------------

For ``conda`` users:

.. code-block:: shell

   conda install --channel conda-forge xenso

or with ``pip``:

.. code-block:: shell

   pip install xenso

Overview
--------

xENSO works on :class:`xarray.DataArray` objects of monthly data with ``time``,
``lat`` and ``lon`` dimensions, in any longitude convention.

- **Indices**: the Niño regions (:func:`xenso.nino_regions`), ONI
  (:func:`xenso.oni`), relative ONI (:func:`xenso.roni`) and the E and C
  indices (:class:`xenso.ECindex`).
- **ENSO events**: detection (:func:`xenso.detect_events`), event windows and
  composites (:func:`xenso.event_windows`, :func:`xenso.composite`) and
  durations (:func:`xenso.event_duration`).
- **ENSO metrics**: the diagnostics and model-observation metrics of the CLIVAR
  ENSO Metrics Package, described in :doc:`enso_metrics`.

.. code-block:: python

   import xarray as xr
   import xenso

   sst = xr.open_dataarray("sst.nc")

   # Oceanic Niño Index
   oni = xenso.oni(sst, base_period=("1991-01-01", "2020-12-31"))

   # El Niño years, from the detrended and smoothed Niño 3.4 anomaly
   nino34 = xenso.compute_anomaly(xenso.region_mean(sst, "nino34"), base_period=("1991", "2020"))
   el_nino = xenso.detect_events(xenso.smooth(xenso.detrend(nino34)), kind="nino")

   # ENSO amplitude of a model compared with observations, in %
   value, error = xenso.compare(xenso.enso_amplitude(model_sst), xenso.enso_amplitude(sst))
