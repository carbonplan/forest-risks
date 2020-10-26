import warnings

import fsspec
import xarray as xr
import numpy as np
from pyproj import Proj, transform
from rasterio import Affine
from rasterio.transform import rowcol

from .. import setup, utils


def cmip(
    store='gcs',
    df=None,
    tlim=None,
    model=None,
    scenario=None,
    mean=True,
    coarsen=None,
    data_vars=['ppt', 'tmax'],
    data_aggs=['sum', 'mean'],
    return_type='xarray',
):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ResourceWarning)
        warnings.simplefilter('ignore', category=FutureWarning)
        warnings.simplefilter('ignore', category=RuntimeWarning)

        if scenario is None:
            raise ValueError('must specify scenario')
        if model is None:
            raise ValueError('must specify model')

        path = setup.loading(store)
        mapper = fsspec.get_mapper(
            (path / f'scratch/downscaling/bias-correction-annual/{model}.{scenario}.zarr').as_uri()
        )

        ds = xr.open_zarr(mapper, consolidated=True)

        ds['tavg_mean'] = (ds['tmin_mean'] + ds['tmax_mean']) / 2

        X = xr.Dataset()

        keys = [var + '_' + agg for var, agg in zip(data_vars, data_aggs)]

        # no aggregation over months because we've precomputed that for now
        for key in keys:
            X[key] = ds[key][0]
            if 'tmax' in key or 'tmin' in key or 'tavg' in key:
                X[key] = X[key] - 273.15

        if tlim is not None:
            tlim = list(map(str, tlim))
            X = X.sel(time=slice(*tlim))

        if coarsen:
            X_coarse = xr.Dataset()
            for key in keys:
                X_coarse[key] = X[key].coarsen(x=coarsen, y=coarsen, boundary='trim').mean()
            X = X_coarse

        if df is not None:
            t = Affine(*utils.albers_conus_transform(4000))
            p1 = Proj(utils.albers_conus_crs())
            p2 = Proj(proj='latlong', datum='WGS84')
            x, y = transform(p2, p1, df['lon'].values, df['lat'].values)
            rc = rowcol(t, x, y)
            ind_r = xr.DataArray(rc[0], dims=['c'])
            ind_c = xr.DataArray(rc[1], dims=['c'])

            base = X[keys].isel(y=ind_r, x=ind_c).load()
            for key in keys:
                df[key + '_mean'] = base[key].mean('time').values
                df[key + '_min'] = base[key].min('time').values
                df[key + '_max'] = base[key].max('time').values
            for key in keys:
                df = df[~np.isnan(df[key + '_mean'])]
            df = df.reset_index(drop=True)
            return df

        if return_type == 'xarray':
            X.load()
            return X

