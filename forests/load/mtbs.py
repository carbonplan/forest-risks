import fsspec
import xarray as xr

from .. import setup


def mtbs(store='gcs', return_type='xarray', coarsen=None):
    path = setup.loading(store)
    mapper = fsspec.get_mapper((path / 'processed/mtbs/conus/4000m/raster.zarr').as_uri())
    mtbs = xr.open_zarr(mapper)
    mtbs['x'] = range(len(mtbs['x']))
    mtbs['y'] = range(len(mtbs['y']))
    mtbs = mtbs.drop('x')
    mtbs = mtbs.drop('y')

    if coarsen:
        mtbs = mtbs.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()

    if return_type == 'xarray':
        mtbs.load()
        return mtbs

    if return_type == 'numpy':
        time = mtbs['time'].values
        y = {'burned_area': mtbs['burned_area'].values}

        return y, time
