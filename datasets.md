# Datasets

See below for a table of all available datasets pertaining to climate-driven risks to forest carbon.

See below for a code sample to access these data.

```
import xarray as xr
import fsspec

store = fsspec.get_mapper('https://carbonplan.blob.core.windows.net/carbonplan-forests/risks/results/web/fire.zarr')

ds = xr.open_zarr(store, consolidated=True)
```

| Variable | License | Store |
| Fire | CC-BY | https://carbonplan.blob.core.windows.net/carbonplan-forests/risks/results/web/fire.zarr |
| Insects | CC-BY | https://carbonplan.blob.core.windows.net/carbonplan-forests/risks/results/web/insects.zarr |
| Drought | CC-BY | https://carbonplan.blob.core.windows.net/carbonplan-forests/risks/results/web/drought.zarr |
| Biomass | CC-BY | https://carbonplan.blob.core.windows.net/carbonplan-forests/risks/results/web/biomass.zarr |
