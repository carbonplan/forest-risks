import os

import rioxarray
import xarray as xr
from scipy.stats import binom

from carbonplan_forest_risks import load
from carbonplan_forest_risks.utils import get_store

# flake8: noqa


account_key = os.environ.get('BLOB_ACCOUNT_KEY')

# this is only used to provide the x/y template for the insects/drought tifs
grid_template = (
    load.nlcd(store="az", year=2016).sel(band=[41, 42, 43, 90]).sum("band") > 0.5
).astype("float")

# # by passing mask as none we don't mask out any values
# # we'll pass a mask for when we do the webmap data prep
cmip_insect_url_template = "https://carbonplan.blob.core.windows.net/carbonplan-scratch/from-bill-05-03-2021/InsectProjections_Maps_5-5-21/InsectModelProjection_{}.{}.{}-{}.{}-v18climate_05-05-2021.tif"
da = load.impacts(cmip_insect_url_template, grid_template, mask=None) * 100
out_path = get_store('carbonplan-forests', 'risks/results/paper/insects_cmip_v5.zarr')
ds = xr.Dataset()
ds['probability'] = da.to_array(dim='vars').rename({'vars': 'gcm'})
ds.to_zarr(out_path, mode='w', consolidated=True)


cmip_drought_url_template = "https://carbonplan.blob.core.windows.net/carbonplan-scratch/from-bill-05-03-2021/DroughtProjections_Maps_5-5-21/DroughtModelProjection_{}.{}.{}-{}.{}-v18climate_05-05-2021.tif"

da = load.impacts(cmip_drought_url_template, grid_template, mask=None) * 100
out_path = get_store('carbonplan-forests', 'risks/results/paper/drought_cmip_v5.zarr')
ds = xr.Dataset()
ds['probability'] = da.to_array(dim='vars').rename({'vars': 'gcm'})
ds.to_zarr(out_path, mode='w', consolidated=True)


# load in historical runs to create drought_terraclimate and insects_terraclimate
terraclimate_insect_url_template = "https://carbonplan.blob.core.windows.net/carbonplan-scratch/from-bill-05-03-2021/Fig2_TerraClimateHistModels_4-22-21/InsectModel_ModeledTerraClimateFIAlong_{}-{}_04-22-2021.tif"
ds = xr.Dataset()
ds['probability'] = (
    load.impacts(
        terraclimate_insect_url_template,
        grid_template,
        mask=None,
        period_start=1990,
        period_end=2020,
        met_data='terraclimate',
    )
    * 100
)
out_path = get_store('carbonplan-forests', 'risks/results/paper/insects_terraclimate.zarr')
ds.to_zarr(out_path, mode='w', consolidated=True)


terraclimate_drought_url_template = "https://carbonplan.blob.core.windows.net/carbonplan-scratch/from-bill-05-03-2021/Fig2_TerraClimateHistModels_4-22-21/DroughtModel_ModeledTerraClimateFIAlong_{}-{}_04-22-2021.tif"
ds = xr.Dataset()
ds['probability'] = (
    load.impacts(
        terraclimate_drought_url_template,
        grid_template,
        mask=None,
        period_start=1990,
        period_end=2020,
        met_data='terraclimate',
    )
    * 100
)
out_path = get_store('carbonplan-forests', 'risks/results/paper/drought_terraclimate.zarr')
ds.to_zarr(out_path, mode='w', consolidated=True)
