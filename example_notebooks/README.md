# garudata example notebooks

Notebooks 01–03 walk the full garudata pipeline for the **Indonesia captive-RE
workflow** (the primary sample case) on real, repo-local data: one month of ERA5
weather (January 2023 over the NTT/Timor box) and the national land-suitability
raster. They are meant to be read in order — 01 builds the weather-to-capacity-factor
path, 02 maps developable land, and 03 combines both into per-facility captive
potential. Each runs end-to-end in seconds on a laptop with no credentials or network
access. Notebook 04 is the **secondary sample case**: the China land-use analysis the
land engine was ported from, doubling as its regression test against the validated
golden numbers (machine-local data required). Every notebook is generated
programmatically by a script in `_build/` (edit the builder, not the `.ipynb`, then
re-run it).

| Notebook | What it teaches | Data it needs | Runtime |
| --- | --- | --- | --- |
| `01_weather_cutout_and_capacity_factors.ipynb` | ERA5 store layout, lazy `Cutout`, raw-to-canonical variable translation, PV (GHI and POA) and wind capacity factors, per-village hourly CSV artifacts | `../validation/store` (ERA5 2023-01, NTT box) | ~5 s |
| `02_land_suitability.ipynb` | Windowed, native-CRS suitability analysis with `garudata.land`: score semantics, per-cell suitable area/fraction on a 0.01° grid, aggregation to the ERA5 grid, exclusion samplers | `../landcover_suitability.tif` | ~10 s |
| `03_captive_siting.ipynb` | Facility buffers, per-facility suitable area and CF, packing densities, allocation modes, and the `captive_potential_<year>.csv` artifact GARUDA consumes | both of the above, plus `_out/nb02/` if present (recomputed inline otherwise) | ~5 s |
| `04_china_landuse.ipynb` *(secondary)* | Full-China class-resolved land engine at 30 m (three national passes), terrain-exclusion policy analysis, and the golden-number regression: class totals reproduce the validated pipeline exactly (≤5e-12 %); terrain deltas fully attributed to the stricter nodata policy | machine-local: CLCD 2019 raster, China terrain rasters, ERA5 2019 store, golden outputs (paths in the notebook intro) | ~7 min |

## Running

Use the project virtual environment and run from this directory (the notebooks use
relative paths like `../validation/store` and `../landcover_suitability.tif`):

```bash
cd example_notebooks
jupyter lab            # or: jupyter nbconvert --to notebook --execute <name>.ipynb
```

Notebook 03 reuses notebook 02's saved suitability grid when it exists and recomputes
it inline when it does not, so the notebooks also run standalone in any order.

## Outputs

All artifacts (CSVs, netCDF grids) are written under `_out/`, which is disposable and
gitignored — delete it at any time and re-run the notebooks to regenerate everything.
