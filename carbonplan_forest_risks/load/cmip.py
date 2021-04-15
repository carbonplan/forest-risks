import warnings

import fsspec
import numpy as np
import xarray as xr
import zarr
from pyproj import Proj, transform
from rasterio import Affine
from rasterio.transform import rowcol
from tenacity import retry, stop_after_attempt

from .. import setup, utils

members = {
    'CanESM5-CanOE': 'r3i1p2f1',
    'MIROC-ES2L': 'r1i1p1f2',
    'ACCESS-CM2': 'r1i1p1f1',
    'ACCESS-ESM1-5': 'r10i1p1f1',
    'MRI-ESM2-0': 'r1i1p1f1',
    'MPI-ESM1-2-LR': 'r10i1p1f1',
}


@retry(stop=stop_after_attempt(5))
def cmip(
    store='az',
    df=None,
    tlim=None,
    model=None,
    scenario=None,
    coarsen=None,
    variables=['ppt', 'tmean'],
    mask=None,
    member=None,
    method='bias-corrected',
    sampling='annual',
    historical=False,
    remove_nans=False,
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

        prefix = f'cmip6/{method}/conus/4000m/{sampling}/{model}.{scenario}.{member}.zarr'

        if store == 'az':
            mapper = zarr.storage.ABSStore(
                'carbonplan-downscaling', prefix=prefix, account_name='carbonplan'
            )
        else:
            mapper = fsspec.get_mapper((path / 'carbonplan-downscaling' / prefix).as_uri())

        ds = xr.open_zarr(mapper, consolidated=True)

        if historical:
            prefix = f'cmip6/{method}/conus/4000m/{sampling}/{model}.historical.{member}.zarr'

            if store == 'az':
                mapper = zarr.storage.ABSStore(
                    'carbonplan-downscaling', prefix=prefix, account_name='carbonplan'
                )
            else:
                mapper = fsspec.get_mapper((path / 'carbonplan-downscaling' / prefix).as_uri())

            ds_historical = xr.open_zarr(mapper, consolidated=True)

            ds = xr.concat([ds_historical, ds], 'time')

        ds['cwd'] = ds['def']
        ds['pdsi'] = ds['pdsi'].clip(-16, 16)

        X = xr.Dataset()
        keys = variables

        for key in keys:
            X[key] = ds[key]

        if tlim is not None:
            tlim = list(map(str, tlim))
            X = X.sel(time=slice(*tlim))

        if mask is not None:
            vals = mask.values
            vals[vals == 0] = np.NaN
            X = X * vals

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
