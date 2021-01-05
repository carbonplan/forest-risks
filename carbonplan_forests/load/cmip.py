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
    member='r10i1p1f1',
    coarsen=None,
    variables=['ppt', 'tmean'],
    mask=None,
    sampling='annual',
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

        prefix = f'cmip6/bias-corrected/conus/4000m/{sampling}/{model}.{scenario}.{member}.zarr'

        if store == 'az':
            mapper = zarr.storage.ABSStore(
                'carbonplan-downscaling', prefix=prefix, account_name='carbonplan'
            )
        else:
            mapper = fsspec.get_mapper((path / 'carbonplan-downscaling' / prefix).as_uri())

        ds = xr.open_zarr(mapper, consolidated=True)

        ds['cwd'] = ds['pet'] - ds['aet']
        ds['pdsi'] = ds['pdsi'].where(ds['pdsi'] > -999, 0)
        ds['pdsi'] = ds['pdsi'].where(ds['pdsi'] > -4, -4)
        ds['pdsi'] = ds['pdsi'].where(ds['pdsi'] < 4, 4)

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
