"""Indonesia captive-RE siting with garudata — replaces the geodata notebook.

End-to-end reimplementation of ``captive-indonesia-geodata/indonesia_captive_RE_analysis.ipynb``
without geodata, avoiding the upstream defects that made the original un-runnable or silently
wrong (bug IDs from the geodata audit; fixes contributed upstream separately):

* latitude_optimal raising NotImplementedError south of the equator (C10) and treating
  azimuth degrees as radians (C2)  -> convert.pv_cf_ghi / pv_cf_poa, hemisphere-aware;
* Mask merge_layer AND corrupting uniform windows (M1), min+max filter no-op (M5),
  nodata leaking through slope filters (M6), in-RAM national rasters
  -> land.suitable_area_by_cell with nodata-aware samplers, windowed;
* the notebook's own fragilities: mutable merged_mask ordering, placeholder KLHK codes
  -> suitability passed explicitly; profiles.indonesia.require_klhk_classes() fails loudly.

Run stages independently (each is idempotent); edit the DATA paths below first.
"""

from __future__ import annotations

import logging

import pandas as pd

import garudata as gd
from garudata.profiles import indonesia as idn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# --- Inputs (edit to your paths) -------------------------------------------------
# GARUDA-convention suitability-score raster (0=unsuitable/nodata, 2, 3) —
# the operational land layer; see profiles.indonesia.SUITABILITY_SCORES.
LANDCOVER_TIF = "landcover_suitability.tif"
SLOPE_TIF = "data/USGS_slope_indonesia.tif"         # slope raster in PERCENT (verify units!)
FACILITY_CSV = "data/industrial_facilities.csv"     # cols: Facility, Latitude, Longitude
YEAR = idn.YEAR_CAPTIVE                             # 2018
FINE_RES_DEG = 0.01                                 # suitability grid (~1.1 km); notebook used raster res

SUITABLE_CLASSES = idn.SUITABLE_SCORES              # score >= 2 (strict variant: SUITABLE_SCORES_STRICT)

# --- 1) Weather: download + lazy cutout ------------------------------------------
gd.weather.download_era5_wind_solar(
    years=[YEAR], months=range(1, 13), bounds_wsen=idn.BBOX_INDONESIA
)
cut = gd.cutout.Cutout(config="wind_solar_hourly", years=[YEAR], bounds_wsen=idn.BBOX_INDONESIA)
ds = cut.canonical()

# --- 2) Capacity factors ----------------------------------------------------------
# GHI model (GARUDA reference). For plane-of-array with the KANEKA panel use
# pv_cf_poa(ds, panel=idn.PANEL) once pvlib is installed — it handles lat<0 correctly.
solar_cf_ts = gd.convert.pv_cf_ghi(ds)
wind_cf_ts = gd.convert.wind_cf(ds, turbine=idn.TURBINE)
solar_cf = solar_cf_ts.mean("time")
wind_cf = wind_cf_ts.mean("time")

# --- 3) Suitability on a fine grid (windowed, native-CRS, nodata-aware) -----------
import numpy as np

lons = np.arange(idn.BBOX_INDONESIA[0] + FINE_RES_DEG / 2, idn.BBOX_INDONESIA[2], FINE_RES_DEG)
lats = np.arange(idn.BBOX_INDONESIA[1] + FINE_RES_DEG / 2, idn.BBOX_INDONESIA[3], FINE_RES_DEG)

suit_solar = gd.land.suitable_area_by_cell(
    LANDCOVER_TIF, SUITABLE_CLASSES, lons, lats,
    exclusion_samplers=[gd.land.terrain_sampler(SLOPE_TIF, idn.SLOPE_MAX_SOLAR_PCT)],
)["fraction"]
suit_wind = gd.land.suitable_area_by_cell(
    LANDCOVER_TIF, SUITABLE_CLASSES, lons, lats,
    exclusion_samplers=[gd.land.terrain_sampler(SLOPE_TIF, idn.SLOPE_MAX_WIND_PCT)],
)["fraction"]

# --- 4) Facility buffers + per-facility potential ---------------------------------
facilities = pd.read_csv(FACILITY_CSV)
buffers = gd.siting.facility_buffers(facilities, radius_km=idn.FACILITY_BUFFER_KM)

solar_potential = gd.siting.facility_potential(
    suit_solar, solar_cf, buffers, packing_mw_per_km2=idn.PACKING["captive_solar"]
)
wind_potential = gd.siting.facility_potential(
    suit_wind, wind_cf, buffers, packing_mw_per_km2=idn.PACKING["captive_wind"]
)

# --- 5) Results -------------------------------------------------------------------
out = gd.writers.write_captive_potential(
    solar_potential, wind_potential, YEAR, f"captive_potential_{YEAR}.csv"
)
print(f"wrote {out}")
print(solar_potential.head(10))
