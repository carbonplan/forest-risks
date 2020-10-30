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


def terraclim(
    store='az',
    df=None,
    tlim=None,
    coarsen=None,
    data_vars=['ppt', 'tmax'],
    data_aggs=None,
    mask=None,
    group_repeats=False,
    remove_nans=False,
):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ResourceWarning)
        warnings.simplefilter('ignore', category=FutureWarning)
        warnings.simplefilter('ignore', category=RuntimeWarning)

        path = setup.loading(store)

        if store == 'az':
            prefix = 'processed/terraclimate/conus/4000m/raster.zarr'
            mapper = zarr.storage.ABSStore(
                'carbonplan-data',
                prefix=prefix,
                account_name='carbonplan',
                account_key=os.environ['BLOB_ACCOUNT_KEY'],
            )
        else:
            prefix = (
                path / 'carbonplan-data/processed/terraclimate/conus/4000m/raster.zarr'
            ).as_uri()
            mapper = fsspec.get_mapper(prefix)

        ds = xr.open_zarr(mapper, consolidated=True)

        ds['cwd'] = ds['pet'] - ds['aet']
        ds['tavg'] = (ds['tmin'] + ds['tmax']) / 2

        X = xr.Dataset()

        if data_aggs is not None:
            keys = [var + '_' + agg for var, agg in zip(data_vars, data_aggs)]
            for key in keys:
                var, agg = key.split('_')
                base = ds[var].resample(time='AS')
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
                X[key] = ds[key]

        if tlim:
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

            if not group_repeats:
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
            else:
                base = X[keys].isel(y=ind_r, x=ind_c).load()
                for key in keys:
                    array = base[key].values.T
                    time = np.arange(X['time.year'].min(), X['time.year'].max() + 1)
                    maxyear = max(
                        map(
                            lambda x: int(x.split('_')[1]),
                            df.columns[['year' in c for c in df.columns]],
                        )
                    )
                    pairs = [(str(y), str(y + 1)) for y in range(maxyear)]
                    for pair in pairs:
                        tlims = [
                            (time > tmin) & (time <= tmax)
                            if (~np.isnan(tmin) & ~np.isnan(tmax))
                            else []
                            for (tmin, tmax) in zip(df[f'year_{pair[0]}'], df[f'year_{pair[1]}'])
                        ]

                        def get_stats(a, t):
                            if len(t) > 0:
                                selection = a[t]
                                return {
                                    'min': selection.min(),
                                    'max': selection.max(),
                                    'mean': selection.mean(),
                                }
                            else:
                                return {'min': np.NaN, 'max': np.NaN, 'mean': np.NaN}

                        stats = [get_stats(a, t) for (a, t) in zip(array, tlims)]

                        df[key + '_min_' + pair[1]] = [d['min'] for d in stats]
                        df[key + '_max_' + pair[1]] = [d['max'] for d in stats]
                        df[key + '_mean_' + pair[1]] = [d['mean'] for d in stats]

                if remove_nans:
                    for key in keys:
                        df = df[~np.isnan(df[key + '_mean_1'])]
                df = df.reset_index(drop=True)
                return df
        else:
            X.load()
            return X
