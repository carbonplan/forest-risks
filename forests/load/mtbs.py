import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import rasterio
import fsspec
import xarray as xr
import numpy as np

from ..utils import setup

def load_rio(f):
    src = rasterio.open(f)
    return src.read(1)

def mtbs(store='gcs', return_type='xarray', coarsen=None):
    path = setup(store)

    X = xr.Dataset()

    mapper = fsspec.get_mapper(path / 'terraclimate/conus/4000m/raster.zarr')

    def weighted_mean(ds, *args, **kwargs):
        weights = ds.time.dt.days_in_month
        return ds.weighted(weights).mean(dim='time')

    tc = xr.open_zarr(mapper)

    base = tc['tmax'].resample(time='AS')
    X['tmax_max'] = base.max('time').sel(time=slice('1984','2018'))
    X['tmax_min'] = base.min('time').sel(time=slice('1984','2018'))
    X['tmax_mean'] = base.map(weighted_mean, dim='time').sel(time=slice('1984','2018'))

    base = tc['tmin'].resample(time='AS')
    X['tmin_max'] = base.max('time').sel(time=slice('1984','2018'))
    X['tmin_min'] = base.min('time').sel(time=slice('1984','2018'))
    X['tmin_mean'] = base.map(weighted_mean, dim='time').sel(time=slice('1984','2018'))

    base = tc['ppt'].resample(time='AS')
    X['ppt_max'] = base.max('time').sel(time=slice('1984','2018'))
    X['ppt_min'] = base.min('time').sel(time=slice('1984','2018'))
    X['ppt_sum'] = base.sum('time').sel(time=slice('1984','2018'))

    # classes 
    classes = [41, 42, 43, 51, 52, 90]

    mask = np.tile(np.asarray([
        xr.open_rasterio(path / f'nlcd/conus/4000m/2001_c{c}.tif').values for c in classes
    ]).sum(axis=0).squeeze(), [35, 1, 1])
    X['forested'] = X['tmax_max']
    X['forested'].values = mask

    y = xr.Dataset()

    mapper = fsspec.get_mapper(path / 'mtbs/conus/4000m/raster.zarr')
    mtbs = xr.open_zarr(mapper)
    y['burned_area'] = mtbs['burned_area']

    y['x'] = X['x']
    y['y'] = X['y']
    y = y.drop('x')
    y = y.drop('y')

    if coarsen:
        X_coarse = xr.Dataset()
        y_coarse = xr.Dataset()
        for var in X.data_vars:
            X_coarse[var] = X[var].coarsen(x=coarsen, y=coarsen, boundary='trim').mean()
        for var in y.data_vars:
            y_coarse[var] = y[var].coarsen(x=coarsen, y=coarsen, boundary='trim').mean()
        X = X_coarse
        y = y_coarse

    if return_type == 'xarray':
        X.load()
        y.load()
        return X, y

    if return_type == 'numpy':
        time = X['time'].values
        lat = X['lat'].values
        lon = X['lon'].values
        X = {var:X[var].values for var in X}
        y = {'burned_area': y['burned_area'].values}
        return X, y, time, lat, lon
