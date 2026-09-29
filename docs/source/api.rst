.. currentmodule:: xenso

.. _api:

#############
API reference
#############

This page provides an auto-generated summary of xENSO's API. All functions are
available from the top-level ``xenso`` namespace.

Anomalies and filters
=====================

.. autosummary::
   :toctree: generated/

   compute_climatology
   compute_anomaly
   detrend
   smooth
   xconvolve

Regions and indices
===================

.. autosummary::
   :toctree: generated/

   region_mean
   nino_regions
   oni
   roni
   ECindex

The boxes available to :func:`region_mean` and the ENSO metrics are listed in
``xenso.REGIONS``.

ENSO events
===========

.. autosummary::
   :toctree: generated/

   seasonal_series
   detect_events
   event_windows
   composite
   event_duration

Statistics
==========

.. autosummary::
   :toctree: generated/

   linregress
   split_regression
   rmse
   compare

ENSO diagnostics
================

Diagnostics of a single dataset, following the CLIVAR ENSO Metrics Package
(see :doc:`enso_metrics`).

.. autosummary::
   :toctree: generated/

   enso_amplitude
   enso_seasonality
   enso_skewness
   enso_duration
   enso_event_duration
   enso_diversity
   bjerknes_feedback
   heat_flux_feedback
   thermocline_feedback
   wind_ssh_feedback
   ocean_driven_sst_change

Model-observation metrics
=========================

.. autosummary::
   :toctree: generated/

   mean_state_rmse
   seasonal_cycle_rmse
   enso_pattern_rmse
   enso_lifecycle_rmse
   enso_teleconnection

Metrics collections
===================

.. autosummary::
   :toctree: generated/

   clivar_collection
