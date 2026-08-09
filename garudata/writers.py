"""GARUDA artifact writers — byte-compatible CSV schemas (hard contract).

These functions reproduce, byte-for-byte in schema, the artifacts that
GARUDA's downstream tooling consumes (interface spec §4.2):

``village_solar_cf_hourly.csv``
    Column 1 ``time``, then one ``village_<id>`` column per point,
    8760 rows of CF in [0, 1], local-time-shifted
    (consumed by ``tools/ntt/wire_era5_solar.py``).
``village_solar_annual.csv``
    ``village, mean_cf, full_load_hours`` with FLH = mean_cf * 8760.
``captive_potential_<year>.csv``
    ``Facility, suitable_area_km2, solar_CF, solar_MW, wind_CF, wind_MW``
    — the outer-merge schema of captive notebook cell ``80316ae2``.

Also provides :func:`sample_points`, the grid -> per-point sampler feeding
the hourly writer (nearest-cell selection + UTC -> local-time roll, matching
``tools/solar_resource_era5.py:139-164``).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .convert import normalize_coords

logger = logging.getLogger(__name__)

HOURLY_FILENAME = "village_solar_cf_hourly.csv"
ANNUAL_FILENAME = "village_solar_annual.csv"

#: id / latitude / longitude column aliases (order = detection priority),
#: matching ``tools/solar_resource_era5.py:153-155``.
_ID_ALIASES = ("id", "village", "name", "Village")
_LAT_ALIASES = ("lat", "Latitude", "latitude")
_LON_ALIASES = ("lon", "Longitude", "longitude", "lng")


def _detect(columns, aliases) -> Optional[str]:
    return next((c for c in aliases if c in columns), None)


# --------------------------------------------------------------------------- #
# grid -> point sampling                                                      #
# --------------------------------------------------------------------------- #

def sample_points(
    cf: xr.DataArray,
    points_df: pd.DataFrame,
    *,
    id_col: Optional[str] = None,
    prefix: str = "village_",
    utc_offset_hours: int = 8,
) -> pd.DataFrame:
    """Sample a CF grid at the nearest cell to each point, in local time.

    Reproduces the sampling stage of ``tools/solar_resource_era5.py``:
    roll the UTC series by ``utc_offset_hours`` (lines 139-145), then
    nearest-neighbor ``.sel`` per point (line 162).

    Parameters
    ----------
    cf : xarray.DataArray
        CF grid with dims time/y/x (any recognized aliases: ``valid_time``,
        ``latitude``/``lat``, ``longitude``/``lon`` are normalized).
    points_df : pandas.DataFrame
        Point table with latitude/longitude columns (aliases: ``lat|Latitude|
        latitude``, ``lon|Longitude|longitude|lng``) and optionally an id
        column (aliases: ``id|village|name|Village``; else the index is used).
    id_col : str, optional
        Explicit id column name, overriding auto-detection.
    prefix : str, optional
        Column-name prefix for the output (default ``"village_"``; use
        ``"site_"`` / ``"ip_"`` for other GARUDA site kinds).
    utc_offset_hours : int, optional
        Hours to roll from UTC to local time (default +8 = WITA; WIB = +7,
        WIT = +9).  **Pass 0 if the array was already shifted** (e.g. via
        ``pv_cf_ghi(..., tz_offset_hours=...)``) — rolling twice corrupts the
        diurnal alignment.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``time`` (local-shifted timestamps of the input axis),
        one ``<prefix><id>`` column per point.  Points with missing
        coordinates are skipped with a warning.
    """
    cf = normalize_coords(cf)
    if utc_offset_hours:
        cf = cf.roll({"time": int(utc_offset_hours)}, roll_coords=False)
        logger.debug("sample_points: shifted UTC -> local by +%d h", utc_offset_hours)

    latc = _detect(points_df.columns, _LAT_ALIASES)
    lonc = _detect(points_df.columns, _LON_ALIASES)
    if not (latc and lonc):
        raise KeyError(
            f"points_df needs latitude/longitude columns "
            f"(aliases {_LAT_ALIASES} / {_LON_ALIASES}); got {list(points_df.columns)}"
        )
    if id_col is None:
        id_col = _detect(points_df.columns, _ID_ALIASES)
    ids = (
        points_df[id_col].astype(str)
        if id_col is not None
        else points_df.index.astype(str)
    )

    time_index = pd.DatetimeIndex(np.asarray(cf["time"].values), name="time")
    cols: dict[str, np.ndarray] = {}
    for pid, lat, lon in zip(ids, points_df[latc], points_df[lonc]):
        if pd.isna(lat) or pd.isna(lon):
            logger.warning("sample_points: point %r has missing coordinates; skipped", pid)
            continue
        ser = cf.sel(y=float(lat), x=float(lon), method="nearest")
        cols[f"{prefix}{pid}"] = np.asarray(ser.values, dtype=float)

    out = pd.DataFrame(cols, index=time_index)
    logger.info("sample_points: sampled %d points x %d hours", out.shape[1], out.shape[0])
    return out


# --------------------------------------------------------------------------- #
# hourly + annual village CF artifacts                                        #
# --------------------------------------------------------------------------- #

def write_village_cf_hourly(
    cf_by_point: pd.DataFrame,
    out_dir: str,
) -> tuple[str, str]:
    """Write ``village_solar_cf_hourly.csv`` + ``village_solar_annual.csv``.

    Byte-compatible with ``tools/solar_resource_era5.py:166-174`` (interface
    spec §4.2 hard contract):

    * hourly file: column 1 ``time``, then ``village_<id>`` columns, one row
      per hour (8760 for a non-leap year), no index column;
    * annual file: ``village, mean_cf, full_load_hours`` with
      ``full_load_hours = mean_cf * 8760`` and ``village`` carrying the bare
      id (prefix stripped).

    Parameters
    ----------
    cf_by_point : pandas.DataFrame
        Hourly CF table indexed by time (as returned by
        :func:`sample_points`).  Columns not already prefixed with
        ``village_`` are prefixed on write.
    out_dir : str
        Output directory (created if missing).

    Returns
    -------
    tuple of str
        ``(hourly_csv_path, annual_csv_path)``.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = cf_by_point.copy()
    df.columns = [
        c if str(c).startswith("village_") else f"village_{c}" for c in df.columns
    ]
    if len(df) not in (8760, 8784):
        logger.warning(
            "hourly CF table has %d rows (expected 8760, or 8784 for a leap year)",
            len(df),
        )

    hourly = df.reset_index()
    hourly = hourly.rename(columns={hourly.columns[0]: "time"})
    hourly_path = os.path.join(out_dir, HOURLY_FILENAME)
    hourly.to_csv(hourly_path, index=False)

    annual_rows = []
    for col in df.columns:
        vid = str(col)[len("village_"):]
        mean_cf = float(np.nanmean(df[col].to_numpy(dtype=float)))
        annual_rows.append((vid, mean_cf, mean_cf * 8760.0))
    annual = pd.DataFrame(annual_rows, columns=["village", "mean_cf", "full_load_hours"])
    annual_path = os.path.join(out_dir, ANNUAL_FILENAME)
    annual.to_csv(annual_path, index=False)

    logger.info(
        "wrote %s (%d h x %d villages) and %s (mean CF %.3f)",
        hourly_path, len(hourly), df.shape[1], annual_path,
        float(annual["mean_cf"].mean()) if len(annual) else float("nan"),
    )
    return hourly_path, annual_path


# --------------------------------------------------------------------------- #
# captive-RE potential artifact                                               #
# --------------------------------------------------------------------------- #

def write_captive_potential(
    solar_df: pd.DataFrame,
    wind_df: pd.DataFrame,
    year: int,
    out_path: str,
) -> str:
    """Write ``captive_potential_<year>.csv`` (captive notebook cell ``80316ae2``).

    Reproduces the notebook's outer-merge exactly: the solar table (renamed
    ``mean_CF -> solar_CF``, ``capacity_MW -> solar_MW``) keeps all its
    columns — including ``suitable_area_km2`` — and the wind table
    contributes only ``Facility, wind_CF, wind_MW`` via an outer merge on
    ``Facility``.  Resulting schema (spec §4.2)::

        Facility, suitable_area_km2, solar_CF, solar_MW, wind_CF, wind_MW

    Row order is preserved from ``solar_df`` (the notebook pre-sorts it by
    mean CF descending; this writer does not re-sort).  The garudata-internal
    ``flagged`` diagnostic column (added by
    :func:`garudata.siting.facility_potential`) is dropped from both inputs,
    and the six contract columns are emitted in the order above (any other
    extra solar columns follow after).

    Parameters
    ----------
    solar_df : pandas.DataFrame
        Per-facility solar potential with columns ``Facility,
        suitable_area_km2`` and either ``mean_CF/capacity_MW`` (notebook
        naming) or already-renamed ``solar_CF/solar_MW``.
    wind_df : pandas.DataFrame
        Per-facility wind potential with ``Facility`` and either
        ``mean_CF/capacity_MW`` or ``wind_CF/wind_MW``.
    year : int
        Analysis year (used in the default filename).
    out_path : str
        Output CSV path; if it does not end in ``.csv`` it is treated as a
        directory (created if missing) and the file is named
        ``captive_potential_<year>.csv`` inside it.

    Returns
    -------
    str
        Path of the written CSV.
    """
    solar = solar_df.rename(columns={"mean_CF": "solar_CF", "capacity_MW": "solar_MW"})
    wind = wind_df.rename(columns={"mean_CF": "wind_CF", "capacity_MW": "wind_MW"})
    # 'flagged' is a garudata-internal diagnostic added by
    # siting.facility_potential; it is not part of the §4.2 schema and must
    # not leak into the artifact.
    solar = solar.drop(columns=["flagged"], errors="ignore")
    wind = wind.drop(columns=["flagged"], errors="ignore")

    for df, needed, label in (
        (solar, ("Facility", "solar_CF", "solar_MW"), "solar_df"),
        (wind, ("Facility", "wind_CF", "wind_MW"), "wind_df"),
    ):
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"{label} is missing required columns {missing}")
    if "suitable_area_km2" not in solar.columns:
        logger.warning("solar_df has no 'suitable_area_km2' column; output will omit it")

    results = solar.merge(
        wind[["Facility", "wind_CF", "wind_MW"]],
        on="Facility",
        how="outer",
    )
    # Enforce the contract column order (§4.2), extra solar columns after.
    canonical = [
        c
        for c in ("Facility", "suitable_area_km2", "solar_CF", "solar_MW",
                  "wind_CF", "wind_MW")
        if c in results.columns
    ]
    extras = [c for c in results.columns if c not in canonical]
    results = results[canonical + extras]

    if str(out_path).lower().endswith(".csv"):
        path = str(out_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        os.makedirs(out_path, exist_ok=True)
        path = os.path.join(out_path, f"captive_potential_{year}.csv")

    results.to_csv(path, index=False)
    logger.info("wrote %s (%d facilities)", path, len(results))
    return path
