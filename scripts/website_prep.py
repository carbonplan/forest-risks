import os
import warnings

import numpy as np
import rioxarray
import xarray as xr
from scipy.stats import binom

from carbonplan_forest_risks import load
from carbonplan_forest_risks.utils import get_store

# flake8: noqa


warnings.filterwarnings('ignore')

impacts_to_consolidate = ['fire']
account_key = os.environ.get('BLOB_ACCOUNT_KEY')
rolling = True
# specify the kind of mask you want to use
mask_for_website = True
website_mask = (
    load.nlcd(store="az", year=2016).sel(band=[41, 42, 43, 90]).sum("band") > 0.5
).astype("float")
gcms = [
    ("ACCESS-CM2", "r1i1p1f1"),
    ("ACCESS-ESM1-5", "r10i1p1f1"),
    ("MRI-ESM2-0", "r1i1p1f1"),
    ("MIROC-ES2L", "r1i1p1f2"),
    ("MPI-ESM1-2-LR", "r10i1p1f1"),
    ("CanESM5-CanOE", "r3i1p2f1"),
]

if 'fire' in impacts_to_consolidate:
    ds = xr.open_zarr(
        get_store(
            'carbonplan-forests', 'risks/results/paper/fire_cmip.zarr', account_key=account_key
        )
    )
    ds = ds.assign_coords(
        {
            "x": website_mask.x,
            "y": website_mask.y,
        }
    )
    print('here!')
    ds = ds.groupby('time.year').sum().coarsen(year=10).mean().compute()
    print('here!')
    ds = ds.assign_coords({'year': np.arange(1970, 2100, 10)})
    ds = ds.rolling(year=2).mean().drop_sel(year=1970)
    ds = ds.assign_coords({'year': list(map(lambda x: str(x), np.arange(1980, 2100, 10)))})
    ds = ds.mean(dim='gcm').probability.to_dataset(dim='scenario')

    if mask_for_website:
        ds = ds.where(website_mask)

    out_path = get_store(
        'carbonplan-forests', 'risks/results/web/fire.zarr', account_key=account_key
    )
    ds.to_zarr(out_path, mode='w')
