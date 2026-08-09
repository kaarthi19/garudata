"""One-month real-data validation of garudata over NTT (West Timor), Indonesia.

Downloads ERA5 ``wind_solar_hourly`` for January 2023 over the NTT box
(121E..125E, 11S..8.5S; ~10-20 MB), then exercises the full chain:

  weather.download_era5_wind_solar -> cutout.Cutout/.canonical -> convert.pv_cf_ghi
  (+ pv_cf_poa when pvlib is installed) + convert.wind_cf -> writers.sample_points
  -> writers.write_village_cf_hourly

and checks each stage against physical expectations (file shape, units, CF
ranges, local-noon solar peak, GHI-vs-POA agreement). Writes a markdown report
next to this script (``REPORT_<year>-<month>_ntt.md``) and exits non-zero on
any failed check.

Requires a CDS API key in ``~/.cdsapirc`` (https://cds.climate.copernicus.eu/how-to-api).
Re-runs are cheap: the downloaded month is kept under ``validation/store/`` and
skipped if present.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if importlib.util.find_spec("garudata") is None:
    sys.path.insert(0, str(HERE.parent))

import garudata as gd  # noqa: E402
from garudata.profiles import indonesia as idn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

YEAR, MONTH = 2023, 1
STORE = HERE / "store"
OUT = HERE / "out"
HAVE_PVLIB = importlib.util.find_spec("pvlib") is not None

# Three real NTT sites (approx town coordinates), WITA = UTC+8
SITES = pd.DataFrame(
    {
        "village": ["kupang", "soe", "atambua"],
        "lat": [-10.18, -9.86, -9.11],
        "lon": [123.58, 124.28, 124.89],
    }
)

CHECKS: list[tuple[str, str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, detail, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # -- 1. download ------------------------------------------------------------
    files = gd.weather.download_era5_wind_solar(
        years=[YEAR], months=[MONTH], bounds_wsen=idn.BBOX_NTT, store_root=STORE
    )
    f = STORE / "era5" / "wind_solar_hourly" / str(YEAR) / f"{MONTH:02d}.nc"
    check("download: file at store path", f.exists(), str(f))
    check("download: size plausible", 1e6 < f.stat().st_size < 2e8,
          f"{f.stat().st_size/1e6:.1f} MB")

    # -- 2. cutout + canonicalization ------------------------------------------
    cut = gd.cutout.Cutout(years=[YEAR], months=[MONTH], bounds_wsen=idn.BBOX_NTT,
                           store_root=STORE)
    ds = cut.canonical()
    n_hours = ds.sizes["time"]
    check("cutout: 744 hourly steps (Jan)", n_hours == 744, str(n_hours))
    check("cutout: canonical vars present",
          all(v in ds for v in ("influx_direct", "influx_diffuse", "temperature", "wnd100m")),
          str(sorted(ds.data_vars)))
    lon, lat = ds["x"].values, ds["y"].values
    check("cutout: grid inside NTT box",
          lon.min() >= 121 - 0.26 and lon.max() <= 125 + 0.26
          and lat.min() >= -11 - 0.26 and lat.max() <= -8.5 + 0.26,
          f"lon {lon.min()}..{lon.max()}, lat {lat.min()}..{lat.max()}")
    ghi_max = float((ds["influx_direct"] + ds["influx_diffuse"]).max())
    check("units: peak total influx 600-1200 W/m2 (tropics)", 600 < ghi_max < 1200,
          f"{ghi_max:.0f} W/m2")
    t_mean = float(ds["temperature"].mean()) - 273.15
    check("units: mean 2m temperature 20-32 C", 20 < t_mean < 32, f"{t_mean:.1f} C")

    # -- 3. solar CF ------------------------------------------------------------
    cf_ghi = gd.convert.pv_cf_ghi(ds)
    m = float(cf_ghi.mean())
    check("pv_cf_ghi: bounded [0,1]",
          float(cf_ghi.min()) >= 0.0 and float(cf_ghi.max()) <= 1.0)
    # Jan = NTT wet season; monthly mean CF plausibly below the 0.18-0.22 annual anchor
    check("pv_cf_ghi: monthly mean CF plausible (0.08-0.25)", 0.08 < m < 0.25, f"{m:.3f}")
    local = gd.convert.pv_cf_ghi(ds, tz_offset_hours=idn.TZ_OFFSETS_HOURS["WITA"])
    prof = local.mean(dim=("y", "x")).groupby("time.hour").mean()
    peak_hr = int(prof.argmax())
    check("pv_cf_ghi: diurnal peak at local hour 11-13", 11 <= peak_hr <= 13, f"h={peak_hr}")

    if HAVE_PVLIB:
        cf_poa = gd.convert.pv_cf_poa(ds)
        mp = float(cf_poa.mean())
        daylight = cf_ghi.values[cf_ghi.values > 0.02]
        both = (cf_ghi.values > 0.02) & np.isfinite(cf_poa.values)
        r = float(np.corrcoef(cf_ghi.values[both], cf_poa.values[both])[0, 1])
        check("pv_cf_poa: runs, bounded, finite",
              np.isfinite(mp) and 0.0 <= float(cf_poa.min()) and float(cf_poa.max()) <= 1.0,
              f"mean {mp:.3f}")
        check("pv_cf_poa: same magnitude as GHI model (ratio 0.6-1.5)",
              0.6 < mp / m < 1.5, f"poa/ghi = {mp/m:.2f}")
        check("pv_cf_poa: correlates with GHI model (r>0.95 daylight)", r > 0.95,
              f"r={r:.3f}, n={daylight.size}")
    else:
        check("pv_cf_poa: SKIPPED (pvlib not installed)", True)

    # -- 4. wind CF -------------------------------------------------------------
    cf_w = gd.convert.wind_cf(ds, turbine=idn.TURBINE, hub_height=90.0)
    wm = float(cf_w.mean())
    check("wind_cf: bounded, finite (no -inf above cut-out)",
          np.isfinite(cf_w.values).all() and float(cf_w.min()) >= 0.0
          and float(cf_w.max()) <= 1.0)
    check("wind_cf: monthly mean plausible for NTT (0.005-0.45)", 0.005 < wm < 0.45,
          f"{wm:.3f}")

    # -- 5. sampling + GARUDA artifacts ----------------------------------------
    hourly = gd.writers.sample_points(cf_ghi, SITES,
                                      utc_offset_hours=idn.TZ_OFFSETS_HOURS["WITA"])
    check("sample_points: 3 sites x 744 h, distinct profiles",
          hourly.shape == (744, 3) and hourly.nunique().min() > 10,
          str(hourly.shape))
    hourly_path, annual_path = gd.writers.write_village_cf_hourly(hourly, str(OUT))
    got = pd.read_csv(hourly_path)
    check("writers: hourly CSV schema (time + village_*)",
          got.columns[0] == "time" and all(c.startswith("village_") for c in got.columns[1:])
          and len(got) == 744, str(list(got.columns)))
    ann = pd.read_csv(annual_path)
    check("writers: annual CSV schema", list(ann.columns) == ["village", "mean_cf", "full_load_hours"])

    # -- report -----------------------------------------------------------------
    n_fail = sum(1 for *_x, ok in CHECKS if not ok)
    rep = HERE / f"REPORT_{YEAR}-{MONTH:02d}_ntt.md"
    with open(rep, "w") as fh:
        fh.write(f"# garudata real-data validation — ERA5 {YEAR}-{MONTH:02d}, NTT box {idn.BBOX_NTT}\n\n")
        fh.write(f"pvlib available: {HAVE_PVLIB}. Store: `validation/store/` "
                 f"({f.stat().st_size/1e6:.1f} MB). Result: **{len(CHECKS)-n_fail}/{len(CHECKS)} checks passed**.\n\n")
        fh.write("| check | detail | result |\n|---|---|---|\n")
        for name, detail, ok in CHECKS:
            fh.write(f"| {name} | {detail} | {'PASS' if ok else '**FAIL**'} |\n")
        fh.write(f"\nHeadline numbers: mean solar CF (GHI model) **{m:.3f}**, "
                 f"mean wind CF (SWT-107 @90 m) **{wm:.3f}**, diurnal solar peak at "
                 f"local hour **{peak_hr}**.\n\nRe-run: `python validation/run_validation.py`.\n")
    print(f"\n{len(CHECKS)-n_fail}/{len(CHECKS)} passed — report: {rep}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
