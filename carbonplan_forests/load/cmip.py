import os
import warnings

import fsspec
import numpy as np
import xarray as xr
import zarr
from pyproj import Proj, transform
from rasterio import Affine
from rasterio.transform import rowcol

from .. import setup, utils


def cmip(
    store='az',
    df=None,
    tlim=None,
    model=None,
    scenario=None,
    coarsen=None,
    data_vars=None,
    data_aggs=None,
    remove_nans=False,
    annual=False,
):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ResourceWarning)
        warnings.simplefilter('ignore', category=FutureWarning)
        warnings.simplefilter('ignore', category=RuntimeWarning)

        if scenario is None:
            raise ValueError('must specify scenario')
        if model is None:
            raise ValueError('must specify model')
        if data_vars is None:
            raise ValueError('must specify data_vars')

        path = setup.loading(store)

        providers = {
            'BCC-CSM2-MR': 'BCC',
            'ACCESS-ESM1-5': 'CSIRO',
            'CanESM5': 'CCCma',
            'MIROC6': 'MIROC',
            'MPI-ESM1-2-LR': 'MPI-M',
        }
        provider = providers[model]
        pattern = f'ScenarioMIP.{provider}.{model}.{scenario}.Amon.gn'

        if store == 'az':
            if annual:
                prefix = f'downscaling/bias-correction-annual/{pattern}'
            else:
                prefix = f'downscaling/bias-correction/{pattern}'
            mapper = zarr.storage.ABSStore(
                'carbonplan-scratch',
                prefix=prefix,
                account_name='carbonplan',
                account_key=os.environ['BLOB_ACCOUNT_KEY'],
            )
        else:
            if annual:
                prefix = (
                    path / f'carbonplan-scratch/downscaling/bias-correction-annual/{pattern}'
                ).as_uri()
            else:
                prefix = (
                    path / f'carbonplan-scratch/downscaling/bias-correction/{pattern}'
                ).as_uri()
            mapper = fsspec.get_mapper(prefix)

        if annual and data_aggs is False:
            raise ValueError('must specify data_aggs when using annual data')

        if annual:
            ds = xr.open_zarr(mapper, consolidated=True)
            ds['tavg_mean'] = (ds['tmin_mean'] + ds['tmax_mean']) / 2

            X = xr.Dataset()
            keys = [var + '_' + agg for var, agg in zip(data_vars, data_aggs)]

            for key in keys:
                X[key] = ds[key][0]
                if 'tmax' in key or 'tmin' in key or 'tavg' in key:
                    X[key] = X[key] - 273.15
        else:
            ds = xr.open_zarr(mapper, consolidated=True)
            ds = ds.rename({'pr': 'ppt', 'tasmax': 'tmax', 'tasmin': 'tmin'})
            ds['tavg'] = (ds['tmax'] + ds['tmin']) / 2

            X = xr.Dataset()
            if data_aggs is not None:
                keys = [var + '_' + agg for var, agg in zip(data_vars, data_aggs)]
                for key in keys:
                    var, agg = key.split('_')
                    base = ds[var][0].resample(time='AS')
                    if agg == 'sum':
                        X[key] = base.sum('time')
                    elif agg == 'mean':
                        X[key] = base.map(utils.weighted_mean, dim='time')
                    elif agg == 'max':
                        X[key] = base.max('time')
                    elif agg == 'min':
                        X[key] = base.min('time')
                    else:
                        raise ValueError(f'agg method {agg} not supported')
            else:
                keys = data_vars
                for key in keys:
                    X[key] = ds[key][0]
            for key in keys:
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
            if remove_nans:
                for key in keys:
                    df = df[~np.isnan(df[key + '_mean'])]
            df = df.reset_index(drop=True)
            return df

        X = X.drop(['x', 'y'])
        X.load()
        return X
