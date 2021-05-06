import os

import rioxarray
import xarray as xr
from scipy.stats import binom

from carbonplan_forest_risks import load
from carbonplan_forest_risks.utils import get_store

# flake8: noqa


account_key = os.environ.get('BLOB_ACCOUNT_KEY')

# this is only used to provide the x/y template for the insects/drought tifs
website_mask = (
    load.nlcd(store="az", year=2016).sel(band=[41, 42, 43, 90]).sum("band") > 0.5
).astype("float")

insect_url_template = "https://carbonplan.blob.core.windows.net/carbonplan-scratch/from-bill-04-09-2021/InsectProjections_EnsembleBaseline_4-8-21/InsectModelProjection_{}.{}.{}-{}.{}-v15climate_{}.tif"
da = load.impacts(insect_url_template, website_mask, mask=None) * 100
out_path = get_store('carbonplan-forests', 'risks/results/paper/insects_cmip.zarr')
ds = xr.Dataset()
ds['probability'] = da.to_array(dim='vars').rename({'vars': 'gcm'})
ds.to_zarr(out_path, mode='w', consolidated=True)


drought_url_template = "https://carbonplan.blob.core.windows.net/carbonplan-scratch/from-bill-04-09-2021/DroughtProjections_EnsembleBaseline_4-8-21/DroughtModelProjection_{}.{}.{}-{}.{}-v15climate_{}.tif"

da = load.impacts(drought_url_template, website_mask, mask=None) * 100
out_path = get_store('carbonplan-forests', 'risks/results/paper/drought_cmip.zarr')
ds = xr.Dataset()
ds['probability'] = da.to_array(dim='vars').rename({'vars': 'gcm'})
ds.to_zarr(out_path, mode='w', consolidated=True)


# TODO: drought_terraclimate and insects_terraclimate
