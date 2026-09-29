ENSO metrics
============

xENSO re-implements the diagnostics and metrics of the
`CLIVAR ENSO Metrics Package <https://github.com/CLIVAR-PRP/enso_metrics>`_
(Planton et al., 2021, `doi:10.1175/BAMS-D-19-0337.A <https://doi.org/10.1175/BAMS-D-19-0337.A>`_)
with an xarray interface.

.. warning::

   This is not the official implementation. Results can differ in details,
   which are noted in the docstrings. Model evaluations that need to be
   comparable with published CLIVAR ENSO metrics should use the original
   package.

Inputs
------

- Monthly fields with ``time``, ``lat`` and ``lon`` dimensions. Longitudes
  can be in -180 to 180 or 0 to 360.
- **Ocean-only fields.** Land points are averaged like any other point, so
  use ``tos``, or ``ts`` masked with ``sftlf``. CLIVAR masks every grid cell
  with some land.
- **A common grid.** The model-observation metrics raise a ``ValueError`` if
  the model and the observations are on different grids. CLIVAR regrids both
  to 1° beforehand, which is left to the user. The longitude smoothing of
  :func:`xenso.enso_diversity` assumes a 1° grid.
- **Units.** No units are converted. CLIVAR uses °C for SST, mm/day for
  precipitation, 1e-3 N m-2 for zonal wind stress, cm for sea surface height
  and W m-2 for the net surface heat flux, positive into the ocean.

Usage
-----

Diagnostics of one dataset return a Dataset with ``value`` and, where CLIVAR
defines one, ``error``. :func:`xenso.compare` turns a model and an observed
diagnostic into a metric:

.. code-block:: python

   import xenso

   model = xenso.enso_amplitude(model_sst)
   obs = xenso.enso_amplitude(obs_sst)
   value, error = xenso.compare(model, obs)  # absolute relative difference, in %

Metrics comparing two datasets return the metric as ``value`` and the compared
profiles or maps as ``model`` and ``obs``:

.. code-block:: python

   result = xenso.enso_pattern_rmse(model_sst, obs_sst)
   result.value  # RMSE along the equator
   result.model.plot()

A whole collection can be computed at once from mappings of fields named
``sst``, ``pr``, ``taux``, ``ssh`` and ``thf``; metrics whose variables are
missing are skipped:

.. code-block:: python

   table = xenso.clivar_collection(model_ds, obs_ds, "ENSO_perf")
   table.to_dataframe()

Correspondence with CLIVAR
--------------------------

Each result stores the CLIVAR name of the metric in its ``clivar_name``
attribute.

.. list-table::
   :header-rows: 1

   * - CLIVAR
     - xENSO
   * - EnsoAmpl, EnsoSeasonality, EnsoSstSkew
     - :func:`xenso.enso_amplitude`, :func:`xenso.enso_seasonality`, :func:`xenso.enso_skewness`
   * - EnsoDuration
     - :func:`xenso.enso_duration`
   * - NinoSstDur, NinaSstDur
     - :func:`xenso.enso_event_duration`
   * - EnsoSstDiversity, NinoSstDiversity, NinaSstDiversity, NinoSstDiv, NinaSstDiv
     - :func:`xenso.enso_diversity`
   * - EnsoFbSstTaux, EnsoFbSstThf, EnsoFbSshSst, EnsoFbTauxSsh
     - :func:`xenso.bjerknes_feedback`, :func:`xenso.heat_flux_feedback`,
       :func:`xenso.thermocline_feedback`, :func:`xenso.wind_ssh_feedback`
   * - EnsodSstOce
     - :func:`xenso.ocean_driven_sst_change`
   * - Bias*LonRmse, Bias*LatRmse
     - :func:`xenso.mean_state_rmse`
   * - Seasonal*LonRmse, Seasonal*LatRmse
     - :func:`xenso.seasonal_cycle_rmse`
   * - EnsoSstLonRmse, NinoSstLonRmse, NinaSstLonRmse
     - :func:`xenso.enso_pattern_rmse`
   * - EnsoSstTsRmse, EnsoPrTsRmse, EnsoTauxTsRmse, NinoSstTsRmse, NinaSstTsRmse
     - :func:`xenso.enso_lifecycle_rmse`
   * - EnsoSstMap*, EnsoPrMap*, EnsoSlpMap*, NinoSstMap, NinaSstMap, ...
     - :func:`xenso.enso_teleconnection`
   * - ENSO_perf, ENSO_proc, ENSO_tel collections
     - :func:`xenso.clivar_collection`

Known differences
-----------------

- Fields are averaged over each box before, rather than after, taking
  anomalies, detrending and smoothing. Anomalies are taken before detrending.
  Both change the Niño 3.4 index by a few hundredths of a degree, enough to
  move an event sitting exactly at the detection threshold.
- The diversity is the interquartile range of the peak longitude of each
  event. An event whose profile has two nearly equal maxima can switch
  between them, so the diversity changes in steps with small changes of the
  input.
- The teleconnection maps leave out a box over the equatorial Pacific
  (15°S-15°N, 120°E-75°W) instead of CLIVAR's basin mask; pass ``keep`` to
  use another mask.
