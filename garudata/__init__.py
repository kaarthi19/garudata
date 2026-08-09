"""garudata — a lean geodata-mimic for the GARUDA model (Indonesia).

Scope (see geodata_work_plan_v2.md §mini-package for design rationale):
  grids     — fixed ERA5 0.25° grid registry, bbox -> cell centers, nearest mapping
  weather   — CDS ERA5 downloads (correct area order, atomic writes, resumable),
              GEODATA_ROOT-compatible store layout
  cutout    — lazy xarray view over the store + raw->canonical variable transform
  convert   — PV capacity factor (GHI model bit-compatible with GARUDA's
              solar_resource_era5.py; optional pvlib plane-of-array path) and
              wind capacity factor (power-curve lookup, correct cut-out handling)
  resources — turbine power curves + panel parameters (extracted from geodata)
  land      — windowed, native-CRS, class-resolved land-fraction engine
              (ported from the validated china_landuse pipeline)
  attribute — per-polygon attribution of cell-level quantities
  siting    — facility buffers, per-facility potential, polygon->point MW
              allocation (shared / buffer / nearest)
  writers   — GARUDA artifact writers (byte-compatible CSV schemas)
  profiles  — country/region constants (profiles.indonesia)

Targets a modern stack (numpy>=2, pandas>=2, xarray, rasterio>=1.5, geopandas,
pyproj). pvlib is optional (only needed for convert.pv_cf(model="poa")).
No dependency on the geodata package itself.
"""

__version__ = "0.1.0"

from . import grids, weather, cutout, convert, resources, land, attribute, siting, writers  # noqa: F401,E501
from . import profiles  # noqa: F401
