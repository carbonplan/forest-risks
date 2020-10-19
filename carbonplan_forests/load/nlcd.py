import rasterio
import xarray as xr

from .. import setup


def load_rio(f):
    src = rasterio.open(f)
    return src.read(1)


def nlcd(store='gcs', classes=[41, 42, 43, 90], year=2001, return_type='xarray', coarsen=None):
    path = setup.loading(store)

    if classes == 'all':
        classes = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95]

    bands = xr.concat(
        [xr.open_rasterio(path / f'processed/nlcd/conus/4000m/{year}_c{c}.tif') for c in classes],
        dim=xr.Variable('band', classes),
    )
    mask = bands.sum('band', keep_attrs=True)

    if coarsen:
        mask = mask.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()

    if return_type == 'xarray':
        mask.load()
        return mask

    if return_type == 'numpy':
        return mask.values
