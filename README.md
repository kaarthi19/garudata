# garudata

A lean, modern-stack renewable-resource toolkit built for the
[GARUDA](https://github.com/kaarthi19/garuda) Indonesia energy-planning model, grown out of a
verified audit of the lab's [geodata](https://github.com/GeodataTools/geodata) package.

## Start here

| If you want to… | Go to |
|---|---|
| **Try garudata yourself** (Indonesia is the sample case) | [`garudata/README.md`](garudata/README.md) — install + quickstart; [`example_notebooks/`](example_notebooks/) — executed tutorials; [`validation/`](validation/) — a one-command real-data check |
| The original captive-RE notebook garudata replaces | [`captive-indonesia-geodata/`](captive-indonesia-geodata/), reimplemented as [`examples/captive_indonesia.py`](examples/captive_indonesia.py) |

## Quick tester path (10 minutes + one small CDS download)

```bash
git clone https://github.com/kaarthi19/garudata.git && cd garudata
python -m venv .venv && source .venv/bin/activate
pip install -e ".[poa,dev]"
pytest                       # 16 offline synthetic tests, ~1 min
python validation/run_validation.py   # 1 month of ERA5 over NTT -> CF sanity report
```

The validation needs a [CDS API key](https://cds.climate.copernicus.eu/how-to-api) in
`~/.cdsapirc`; the download is one small netCDF (~10–20 MB, NTT box, January 2023).

## Why garudata exists

geodata's active branch cannot run download→cutout→convert end-to-end, its PV orientation is
unusable south of the equator, and its masking layer cannot ingest national-scale rasters (each
verified with file:line evidence during the audit; fixes are being contributed upstream
separately). garudata keeps geodata's store layout and workflow
shape (so cluster data and habits transfer), fixes those defects by construction, and stops at
GARUDA's CSV artifact boundary so GARUDA's own validated writers stay in charge. Indonesia (NTT
/ captive-RE) is the primary profile; the China land-use engine it ports is kept as the
secondary, already-validated reference.
