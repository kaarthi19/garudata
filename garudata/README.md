# garudata

A lean, modern-stack geodata-mimic for the [GARUDA](https://github.com/kaarthi19/garuda) model.
It covers the slice of geodata the lab actually uses — ERA5 download → cutout → PV/wind capacity
factors → land suitability → per-facility/per-village siting → GARUDA CSV artifacts — with the
30 verified upstream defects designed out (each documented with file:line evidence during the
audit; fixes are being contributed upstream separately).

**Not a fork.** No geodata dependency; targets numpy ≥ 2 / pandas ≥ 2 / xarray / rasterio ≥ 1.5 /
geopandas / pyproj in one venv. `pvlib` is optional (`convert.pv_cf_poa` only); `cdsapi` is only
imported when a download actually happens; `dask` only for multi-file lazy opens.

## Module map

| Module | What it does | Provenance / contract |
|---|---|---|
| `grids` | ERA5 0.25° / MERRA-2 grid registry, `nearest_cell`, cos-lat `cell_areas_km2` | deterministic CDS grid |
| `weather` | CDS downloads: **correct `area=[N,W,S,E]`**, zip *and* plain-netCDF responses, `.part`+atomic rename, resumable; proper monthly-means product | store layout `$GEODATA_ROOT/era5/<config>/<year>/<month:02d>.nc` — cluster-compatible with the lab store |
| `cutout` | `Cutout` = lazy xarray view over the store (no materialized copies); `canonicalize(ds)` = the master-era raw→canonical translation (÷3600, `influx_*`, `wnd100m`, `albedo`, …), idempotent | mirrors PR #104's `transform_wind_solar_dataset` |
| `convert` | `pv_cf_ghi` (bit-compatible with GARUDA `tools/solar_resource_era5.py`), `pv_cf_poa` (pvlib, **fdir→DNI done correctly**, works in both hemispheres), `wind_cf` (log/power-law hub extrapolation, `np.interp` with hard 0 outside the curve) | avoids upstream C2/C3/C10 bugs by construction |
| `resources` | Siemens SWT-107 3.6 MW + Vestas V112 3.0 MW power curves, KANEKA + CSi panel params | extracted from geodata @ master (GPL-3.0) |
| `land` | `class_area_by_cell` — windowed, native-CRS, class-resolved area/fraction engine (bounded memory, 22 GB rasters OK); `terrain_sampler` (nodata never passes), `polygon_sampler`; `suitable_area_by_cell` | line-faithful port of the validated china_landuse 79-s engine |
| `attribute` | `attribute_to_polygons` — conservative per-polygon allocation of cell values (EPSG:6933 overlay; conservation enforced) | china_landuse fig-4 method |
| `siting` | `facility_buffers` (10 km EPSG:6933), `facility_potential` (captive aggregation, mutation-order trap removed), `allocate_to_points` (shared/buffer/nearest with `usable_land.py` semantics), `hub_distance` | GARUDA plug-points |
| `writers` | `sample_points` (+UTC offset roll), `village_solar_cf_hourly.csv` / `village_solar_annual.csv` / `captive_potential_<year>.csv` | **byte-compatible** with GARUDA's `wire_era5_solar.py` inputs |
| `profiles.indonesia` | bboxes, tz offsets, packing densities, slope thresholds (units in names), connection-cost constants, regression anchors; `KLHK_SUITABLE_CLASSES = None` until real codes exist (`require_klhk_classes()` raises) | garuda audit §4.3 |

GARUDA's own writers (`tools/ntt/wire_era5_solar.py`, `solar_potential.py`,
`connection_cost.py`) are **kept and called, not reimplemented** — garudata stops at the
artifact boundary.

## Quickstart

```python
import garudata as gd
from garudata.profiles import indonesia as idn

# 1) download one year for NTT (correct CDS area order, resumable, atomic)
gd.weather.download_era5_wind_solar(years=[2023], months=range(1, 13), bounds_wsen=idn.BBOX_NTT)

# 2) lazy cutout + canonical variables (no gigabyte copies)
cut = gd.cutout.Cutout(config="wind_solar_hourly", years=[2023], bounds_wsen=idn.BBOX_NTT)
ds = cut.canonical()

# 3) capacity factors
solar_cf = gd.convert.pv_cf_ghi(ds)                          # GARUDA reference model
wind_cf  = gd.convert.wind_cf(ds, turbine=idn.TURBINE)       # 0 exactly outside the curve
# solar_poa = gd.convert.pv_cf_poa(ds, panel=idn.PANEL)      # needs pvlib; hemisphere-aware

# 4) village time series -> GARUDA artifacts
pts = ...  # DataFrame with village/lat/lon
hourly = gd.writers.sample_points(solar_cf, pts, utc_offset_hours=idn.TZ_OFFSETS_HOURS["WITA"])
gd.writers.write_village_cf_hourly(hourly, "solar_out/")     # feeds tools/ntt/wire_era5_solar.py
```

For the full captive-RE workflow (land mask + facility buffers + potential table) see
`../examples/captive_indonesia.py`.

## Install (testers start here)

From the repo root, any Python ≥ 3.10:

```bash
pip install -e ".[poa,dev]"   # poa = pvlib plane-of-array model; dev = pytest
pytest                        # 16 offline synthetic tests, ~1 min
python validation/run_validation.py   # real-data check: 1 month ERA5 over NTT (~10-20 MB, needs ~/.cdsapirc)
```

Everything is offline except the validation script. `GEODATA_ROOT` is only needed when you want
downloads/cutouts to use a shared weather store (e.g. the lab cluster store) instead of a local
directory. Indonesia (NTT) is the primary sample case — `profiles/indonesia.py` +
`examples/captive_indonesia.py`; the China land-use pipeline the land engine was ported from is
the secondary, already-validated reference.

## Validation status

- 16/16 synthetic smoke tests pass (grids arithmetic, canonicalize round-trip incl. ÷3600 and
  idempotency, GHI model bit-checks vs the GARUDA reference, wind curve endpoints incl. v>cut-out
  → exactly 0, land-engine conservation <1 %, attribution exact conservation, all three
  allocation modes with their documented failure envelopes, writer schemas).
- **Not yet validated**: real CDS downloads (request construction verified offline only);
  `pv_cf_poa` numerics (pvlib unavailable in this session); land engine vs the China golden
  numbers (22 GB raster not on this machine — the code is a line-faithful port of the cell that
  produced them, plus a deliberate nodata-sampler fix).
- Known caveats: `land` drops sub-block remainders when `window % block != 0` (defaults are
  fine: 0.02 %); EPSG:6933 planar distances are ±13–15 % anisotropic at the equator
  (`hub_distance` docstring explains; use UTM/3857 for distance fidelity); captive results are
  **blocked on real KLHK class codes** — `require_klhk_classes()` fails loudly by design.
