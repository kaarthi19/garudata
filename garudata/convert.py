"""Capacity-factor conversion: ERA5 weather -> PV / wind CF DataArrays.

Three entry points:

``pv_cf_ghi``
    Transparent GHI-based PV model, bit-compatible with GARUDA's reference
    implementation (``tools/solar_resource_era5.py:79-94``).  No pvlib.
``pv_cf_poa``
    Plane-of-array PV model via pvlib (lazy optional dependency): solar
    position + transposition of ERA5 direct/diffuse horizontal fluxes.
``wind_cf``
    Hub-height wind speed (log-law or power-law extrapolation from 100 m)
    through a turbine power curve from :mod:`garudata.resources`.

All outputs are ``xarray.DataArray`` named ``"cf"`` with dims
``(time, y, x)`` and values clipped to [0, 1].

Input conventions
-----------------
Functions accept either **raw** ERA5 variable names (``ssrd`` [J/m2 per h],
``fdir``, ``t2m`` [K], ``u100``/``v100``) or geodata-**canonical** names
(``influx_direct``/``influx_diffuse`` [W/m2], ``temperature`` [K],
``wnd100m``, ``roughness``), detected by name and ``units`` attribute.
Coordinates are normalized to ``time``/``y``/``x`` from any of
``valid_time|time``, ``latitude|lat|y``, ``longitude|lon|x``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .resources import PANELS, TURBINES

logger = logging.getLogger(__name__)

#: cos(zenith) guard below which DNI is set to 0 (sun elevation < ~3 deg).
COS_ZENITH_MIN = 0.0523

# --------------------------------------------------------------------------- #
# coordinate / variable normalization helpers                                 #
# --------------------------------------------------------------------------- #

_TIME_ALIASES = ("valid_time", "time")
_Y_ALIASES = ("latitude", "lat", "y")
_X_ALIASES = ("longitude", "lon", "x")


def normalize_coords(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Rename time/space coordinates to the canonical ``time``/``y``/``x``.

    Handles ERA5 coordinate-name drift (``valid_time`` vs ``time``,
    ``latitude`` vs ``lat``) without touching data values.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object whose dims/coords may use any recognized alias.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Same object with dims renamed to ``time``, ``y``, ``x`` (where the
        corresponding alias exists).
    """
    ren: dict[str, str] = {}
    names = set(obj.dims) | set(obj.coords)
    for target, aliases in (("time", _TIME_ALIASES), ("y", _Y_ALIASES), ("x", _X_ALIASES)):
        if target in names:
            continue
        for alias in aliases:
            if alias in names:
                ren[alias] = target
                break
    return obj.rename(ren) if ren else obj


def _is_watts(da: xr.DataArray) -> Optional[bool]:
    """Classify a radiation variable's units: True=W/m2, False=J/m2, None=unknown."""
    units = str(da.attrs.get("units", "")).lower()
    if not units:
        return None
    if "j" in units:
        return False
    if "w" in units:
        return True
    return None


def _ghi(ds: xr.Dataset) -> xr.DataArray:
    """Extract GHI [W/m2] from raw (``ssrd``) or canonical (``influx_*``) data.

    Raw hourly ``ssrd`` is an accumulated flux in J/m2 per hour and is divided
    by 3600 (reference: ``solar_resource_era5.py:83-84``); canonical variables
    already carrying ``W``-typed units are used as-is.
    """
    for name in ("ssrd", "surface_solar_radiation_downwards"):
        if name in ds:
            da = ds[name]
            if _is_watts(da):
                logger.debug("GHI from %s (already W/m2 per units attr)", name)
                return da
            logger.debug("GHI from %s / 3600 (J/m2 per hour -> W/m2)", name)
            return da / 3600.0
    if "influx" in ds:
        da = ds["influx"]
        return da if _is_watts(da) is not False else da / 3600.0
    if "influx_direct" in ds and "influx_diffuse" in ds:
        dirr, diff = ds["influx_direct"], ds["influx_diffuse"]
        scale = 3600.0 if _is_watts(dirr) is False else 1.0
        logger.debug("GHI = influx_direct + influx_diffuse (scale 1/%s)", scale)
        return (dirr + diff) / scale
    raise KeyError(
        "no GHI variable found: expected one of ssrd, "
        "surface_solar_radiation_downwards, influx, or "
        "influx_direct+influx_diffuse in the dataset"
    )


def _direct_diffuse(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Extract horizontal direct and diffuse fluxes [W/m2] (raw or canonical)."""
    if "influx_direct" in ds and "influx_diffuse" in ds:
        dirr, diff = ds["influx_direct"], ds["influx_diffuse"]
        if _is_watts(dirr) is False:
            dirr, diff = dirr / 3600.0, diff / 3600.0
        return dirr, diff
    if "fdir" in ds and "ssrd" in ds:
        # raw ERA5 accumulations, J/m2 per hour
        dirr = ds["fdir"] / 3600.0
        diff = (ds["ssrd"] - ds["fdir"]).clip(min=0.0) / 3600.0
        return dirr, diff
    raise KeyError(
        "plane-of-array model needs direct+diffuse components: expected "
        "influx_direct/influx_diffuse (canonical) or fdir/ssrd (raw ERA5)"
    )


def _temperature_c(ds: xr.Dataset, template: xr.DataArray) -> xr.DataArray:
    """Extract near-surface air temperature in degC; fall back to 25 degC.

    Mirrors ``solar_resource_era5.py:85-91``: Kelvin input is converted with
    ``- 273.15``; a missing temperature variable yields the STC assumption.
    """
    for name in ("t2m", "2m_temperature", "temperature"):
        if name in ds:
            da = ds[name]
            units = str(da.attrs.get("units", "")).lower()
            if units.startswith("c") or "celsius" in units or "degc" in units.replace(" ", ""):
                return da
            return da - 273.15  # K -> degC
    logger.warning("no temperature variable found; assuming 25 degC (STC)")
    return template * 0.0 + 25.0


def _roll_time(cf: xr.DataArray, hours: int) -> xr.DataArray:
    """Roll the CF series from UTC to local time (reference lines 139-145).

    ERA5 is UTC; rolling by ``hours`` aligns the solar peak with local noon so
    village demand profiles (local time) see the correct diurnal shape.
    """
    if hours:
        cf = cf.roll({"time": int(hours)}, roll_coords=False)
        logger.debug("shifted UTC -> local by +%d h (time roll)", hours)
    return cf


def _finalize(cf: xr.DataArray) -> xr.DataArray:
    """Name, order and clip a CF result: dims (time, y, x), values [0, 1]."""
    cf = cf.rename("cf").clip(0.0, 1.0)
    dims = [d for d in ("time", "y", "x") if d in cf.dims]
    other = [d for d in cf.dims if d not in dims]
    return cf.transpose(*dims, *other)


# --------------------------------------------------------------------------- #
# (a) GHI-based PV capacity factor                                            #
# --------------------------------------------------------------------------- #

def pv_cf_ghi(
    ds: xr.Dataset,
    *,
    pr: float = 0.80,
    noct: float = 45.0,
    gamma: float = -0.004,
    tz_offset_hours: int = 0,
) -> xr.DataArray:
    """GHI-based PV capacity factor, bit-compatible with GARUDA's ERA5 tool.

    Implements exactly the model of ``tools/solar_resource_era5.py:79-94``::

        GHI   = ssrd / 3600            (J/m2 per hour -> W/m2; or canonical W/m2)
        Tcell = T2m + (NOCT - 20)/800 * GHI
        CF    = clip(GHI/1000, 0, 1) * PR * (1 + gamma*(Tcell - 25)),  clipped to [0, 1]

    This is a fixed-array estimate without plane-of-array transposition: good
    for CF *shapes*; calibrate annual energy externally if a specific yield is
    required.

    Parameters
    ----------
    ds : xarray.Dataset
        ERA5 data with GHI (``ssrd`` raw or ``influx*`` canonical) and
        optionally 2 m temperature (``t2m``/``temperature``; missing
        temperature assumes 25 degC).
    pr : float, optional
        Performance ratio (default 0.80).
    noct : float, optional
        Nominal operating cell temperature [degC] (default 45.0).
    gamma : float, optional
        Power temperature coefficient [1/degC] (default -0.004).
    tz_offset_hours : int, optional
        Hours to roll the series from UTC to local time (WITA = +8, WIB = +7,
        WIT = +9; default 0 = leave in UTC).  **Do not also pass a non-zero
        ``utc_offset_hours`` to** :func:`garudata.writers.sample_points`
        **for the same array — that would double-shift.**

    Returns
    -------
    xarray.DataArray
        ``cf(time, y, x)`` in [0, 1].
    """
    ds = normalize_coords(ds)
    ghi = _ghi(ds)
    t2m = _temperature_c(ds, ghi)
    tcell = t2m + (noct - 20.0) / 800.0 * ghi
    cf = (ghi / 1000.0).clip(0, 1) * pr * (1.0 + gamma * (tcell - 25.0))
    cf = _finalize(cf)
    return _roll_time(cf, tz_offset_hours)


# --------------------------------------------------------------------------- #
# (b) plane-of-array PV capacity factor (pvlib)                               #
# --------------------------------------------------------------------------- #

def pv_cf_poa(
    ds: xr.Dataset,
    *,
    tilt: Optional[float] = None,
    azimuth: Optional[float] = None,
    panel: str = "KANEKA",
    pr: float = 0.80,
    gamma: Optional[float] = None,
) -> xr.DataArray:
    """Plane-of-array PV capacity factor via pvlib (optional dependency).

    Pipeline per grid cell: pvlib solar position -> DNI from the ERA5
    horizontal direct flux -> POA irradiance
    (``pvlib.irradiance.get_total_irradiance``, isotropic sky) -> NOCT cell
    temperature -> linear temperature-derated CF.

    .. note:: **ERA5 ``fdir`` is a HORIZONTAL flux, not DNI.**  It must be
       divided by cos(solar zenith) to obtain direct-normal irradiance, with a
       low-sun guard (``cos(zenith) > 0.0523``, i.e. sun elevation > ~3 deg,
       else DNI = 0).  Feeding ``fdir``/``influx_direct`` to pvlib directly as
       ``dni`` applies the cosine twice and biases POA low — the exact
       "fdir-as-DNI" defect found in geodata PR #104's pvlib path
       (kulcoder review §4, ``model/pvlib/_base.py:859``).

    .. note:: Hemisphere-aware defaults: ``tilt = min(|lat|, 40)`` and
       ``azimuth = 180 (south-facing) if lat >= 0 else 0 (north-facing)``,
       i.e. equator-facing, evaluated per grid cell.  Upstream geodata's
       ``latitude_optimal`` cannot provide this for Indonesia: it raises
       ``NotImplementedError`` for any negative latitude
       (``pv/orientation.py:64-65``) and returns azimuth ``180.0`` degrees
       where radians are expected (line 76).

    Parameters
    ----------
    ds : xarray.Dataset
        ERA5 data with direct+diffuse components (canonical
        ``influx_direct``/``influx_diffuse`` in W/m2, or raw ``fdir``/``ssrd``
        in J/m2 per hour) and temperature.  Time must be UTC (naive
        timestamps are treated as UTC by pvlib).
    tilt : float, optional
        Surface tilt [deg from horizontal].  Default: per-cell ``min(|lat|, 40)``.
    azimuth : float, optional
        Surface azimuth [deg, pvlib convention: 0=N, 90=E, 180=S].
        Default: per-cell equator-facing.
    panel : str, optional
        Panel name in :data:`garudata.resources.PANELS` (default ``"KANEKA"``);
        supplies NOCT and the temperature coefficient when available.
    pr : float, optional
        Performance ratio (default 0.80).
    gamma : float, optional
        Power temperature coefficient [1/degC].  Default: the panel's ``D``
        coefficient (KANEKA: -0.0035), else -0.004.

    Returns
    -------
    xarray.DataArray
        ``cf(time, y, x)`` in [0, 1], time still in UTC (shift downstream,
        e.g. via :func:`garudata.writers.sample_points`).

    Raises
    ------
    ImportError
        If pvlib is not installed.
    KeyError
        If ``panel`` is unknown or required variables are missing.
    """
    try:
        import pvlib  # noqa: PLC0415  (lazy optional dependency)
    except ImportError as exc:
        raise ImportError(
            "pv_cf_poa requires the optional dependency 'pvlib' "
            "(pip install pvlib). Use pv_cf_ghi for the pvlib-free GHI model."
        ) from exc

    if panel not in PANELS:
        raise KeyError(f"unknown panel {panel!r}; available: {sorted(PANELS)}")
    pnl = PANELS[panel]
    noct_raw = pnl.get("NOCT", 45.0)
    noct = float(noct_raw) - 273.15 if float(noct_raw) > 200.0 else float(noct_raw)
    if gamma is None:
        gamma = float(pnl.get("D", -0.004))
        logger.debug("gamma resolved from panel %s: %s", panel, gamma)

    ds = normalize_coords(ds)
    direct, diffuse = _direct_diffuse(ds)
    tc = _temperature_c(ds, direct)

    direct = direct.transpose("time", "y", "x")
    diffuse = diffuse.transpose("time", "y", "x")
    tc = tc.broadcast_like(direct).transpose("time", "y", "x")

    times = pd.DatetimeIndex(np.asarray(ds["time"].values))
    yvals = np.asarray(ds["y"].values, dtype=float)
    xvals = np.asarray(ds["x"].values, dtype=float)
    dir_np = np.asarray(direct.values, dtype=float)
    dif_np = np.asarray(diffuse.values, dtype=float)
    tc_np = np.asarray(tc.values, dtype=float)

    out = np.empty((len(times), yvals.size, xvals.size), dtype=float)
    logger.info(
        "pv_cf_poa: %d cells x %d timesteps (panel=%s, pr=%.2f, gamma=%.4f, noct=%.1f)",
        yvals.size * xvals.size, len(times), panel, pr, gamma, noct,
    )
    for iy, lat in enumerate(yvals):
        tilt_cell = float(tilt) if tilt is not None else min(abs(float(lat)), 40.0)
        azim_cell = (
            float(azimuth) if azimuth is not None
            else (180.0 if float(lat) >= 0.0 else 0.0)
        )
        for ix, lon in enumerate(xvals):
            solpos = pvlib.solarposition.get_solarposition(times, float(lat), float(lon))
            zen = solpos["apparent_zenith"].to_numpy()
            azi = solpos["azimuth"].to_numpy()
            cosz = np.cos(np.deg2rad(zen))
            bhor = np.clip(dir_np[:, iy, ix], 0.0, None)
            # fdir is horizontal: DNI = fdir / cos(zenith), guarded at low sun
            sun_up = cosz > COS_ZENITH_MIN
            dni = np.where(sun_up, bhor / np.where(sun_up, cosz, 1.0), 0.0)
            dhi = np.clip(dif_np[:, iy, ix], 0.0, None)
            ghi = bhor + dhi
            total = pvlib.irradiance.get_total_irradiance(
                tilt_cell, azim_cell, zen, azi, dni=dni, ghi=ghi, dhi=dhi
            )
            poa = np.clip(
                np.nan_to_num(np.asarray(total["poa_global"], dtype=float), nan=0.0),
                0.0, None,
            )
            tcell = tc_np[:, iy, ix] + (noct - 20.0) / 800.0 * poa
            cf = np.clip(poa / 1000.0, 0.0, 1.0) * pr * (1.0 + gamma * (tcell - 25.0))
            out[:, iy, ix] = np.clip(cf, 0.0, 1.0)
        logger.debug("pv_cf_poa: row %d/%d done (lat=%.2f)", iy + 1, yvals.size, lat)

    return xr.DataArray(
        out,
        coords={"time": times, "y": yvals, "x": xvals},
        dims=("time", "y", "x"),
        name="cf",
    )


# --------------------------------------------------------------------------- #
# (c) wind capacity factor                                                    #
# --------------------------------------------------------------------------- #

def wind_cf(
    ds: xr.Dataset,
    turbine: str = "Siemens_SWT_107_3600kW",
    hub_height: Optional[float] = None,
) -> xr.DataArray:
    """Wind capacity factor from ERA5 100 m wind through a turbine power curve.

    Wind speed comes from canonical ``wnd100m`` when present, else
    ``sqrt(u100**2 + v100**2)`` from the raw components.  Extrapolation from
    100 m to hub height uses the log law when a roughness length is available
    (``roughness`` canonical or ``fsr`` raw)::

        v_hub = v100 * ln(hub/z0) / ln(100/z0)

    and otherwise the power law with shear exponent alpha = 0.14::

        v_hub = v100 * (hub/100)**0.14

    The power-curve lookup uses ``np.interp(v, V, CF, left=0.0, right=0.0)``,
    which returns exactly 0 below cut-in and above cut-out.  Never use
    ``scipy.interpolate.interp1d(..., fill_value="extrapolate")`` here: the
    duplicated cut-out breakpoint in every turbine curve gives a -inf slope,
    so storm hours above cut-out produce CF = -inf (confirmed geodata
    dataset-refactor bug, ``model/wind/_base.py:137``).

    Parameters
    ----------
    ds : xarray.Dataset
        ERA5 data with ``wnd100m`` or ``u100``/``v100``; optionally
        ``roughness``/``fsr`` [m] for log-law extrapolation.
    turbine : str, optional
        Turbine name in :data:`garudata.resources.TURBINES`
        (default ``"Siemens_SWT_107_3600kW"``, 3.6 MW).
    hub_height : float, optional
        Hub height [m].  Default: the turbine's library hub height
        (Siemens SWT-107: 90 m).

    Returns
    -------
    xarray.DataArray
        ``cf(time, y, x)`` in [0, 1], time in UTC.
    """
    if turbine not in TURBINES:
        raise KeyError(f"unknown turbine {turbine!r}; available: {sorted(TURBINES)}")
    cfg = TURBINES[turbine]
    hub = float(hub_height) if hub_height is not None else float(cfg["hub_height"])

    ds = normalize_coords(ds)
    if "wnd100m" in ds:
        v100 = ds["wnd100m"]
    elif "u100" in ds and "v100" in ds:
        v100 = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2)
    elif "100m_u_component_of_wind" in ds and "100m_v_component_of_wind" in ds:
        v100 = np.sqrt(
            ds["100m_u_component_of_wind"] ** 2 + ds["100m_v_component_of_wind"] ** 2
        )
    else:
        raise KeyError(
            "no 100 m wind found: expected wnd100m (canonical) or u100/v100 (raw)"
        )

    if hub == 100.0:
        v_hub = v100
    else:
        z0 = None
        for name in ("roughness", "fsr"):
            if name in ds:
                z0 = ds[name]
                break
        if z0 is not None:
            # log-law; clip roughness away from 0 (and below hub) for stability
            z0c = z0.clip(min=1e-4)
            v_hub = v100 * (np.log(hub / z0c) / np.log(100.0 / z0c))
            logger.debug("wind_cf: log-law extrapolation 100 m -> %.0f m via %s", hub, name)
        else:
            v_hub = v100 * (hub / 100.0) ** 0.14
            logger.debug(
                "wind_cf: no roughness variable; power-law (alpha=0.14) 100 m -> %.0f m",
                hub,
            )

    curve_v = np.asarray(cfg["V"], dtype=float)
    curve_cf = np.asarray(cfg["POW"], dtype=float)  # per-unit CF (POW_MW / P)

    def _lookup(speed: np.ndarray) -> np.ndarray:
        return np.interp(speed, curve_v, curve_cf, left=0.0, right=0.0)

    cf = xr.apply_ufunc(_lookup, v_hub, dask="parallelized", output_dtypes=[np.float64])
    return _finalize(cf)
