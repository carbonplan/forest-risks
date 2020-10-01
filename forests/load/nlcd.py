import rasterio
import xarray as xr
import numpy as np

from .. import setup

def load_rio(f):
    src = rasterio.open(f)
    return src.read(1)

def nlcd(store='gcs', classes=[41,42,43,51,52,90], return_type='xarray', coarsen=None):
    path = setup.loading(store)
    bands = xr.concat(
        [xr.open_rasterio(path / f'processed/nlcd/conus/4000m/2001_c{c}.tif') for c in classes],
        dim=xr.Variable('band', classes))
    mask = bands.sum('band', keep_attrs=True)

    if coarsen:
        mask = mask.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()

    if return_type == 'xarray':
        mask.load()
        return mask

    if return_type == 'numpy':
        return mask.values
