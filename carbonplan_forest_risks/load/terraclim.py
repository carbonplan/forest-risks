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


@retry(stop=stop_after_attempt(5))
def terraclim(
    store='az',
    df=None,
    tlim=None,
    coarsen=None,
    sampling='annual',
    variables=['ppt', 'tmean'],
    mask=None,
    group_repeats=False,
    remove_nans=False,
):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ResourceWarning)
        warnings.simplefilter('ignore', category=FutureWarning)
        warnings.simplefilter('ignore', category=RuntimeWarning)

        path = setup.loading(store)
        prefix = f'obs/conus/4000m/{sampling}/terraclimate_plus_v3.zarr'

        if store == 'az':
            mapper = zarr.storage.ABSStore(
                'carbonplan-downscaling', prefix=prefix, account_name='carbonplan'
            )
        else:
            mapper = fsspec.get_mapper((path / 'carbonplan-downscaling' / prefix).as_uri())

        ds = xr.open_zarr(mapper, consolidated=True)

        ds['cwd'] = ds['pet'] - ds['aet']
        ds['pdsi'] = ds['pdsi'].clip(-16, 16)

        X = xr.Dataset()

        for key in variables:
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
            for key in variables:
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
                base = X[variables].isel(y=ind_r, x=ind_c).load()
                for key in variables:
                    df[key + '_mean'] = base[key].mean('time').values
                    df[key + '_min'] = base[key].min('time').values
                    df[key + '_max'] = base[key].max('time').values
                if remove_nans:
                    for key in variables:
                        df = df[~np.isnan(df[key + '_mean'])]
                df = df.reset_index(drop=True)
                return df
            else:
                base = X[variables].isel(y=ind_r, x=ind_c).load()
                for key in variables:
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
                            if (len(t) > 0) and (t.sum() > 0):
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
                    for key in variables:
                        df = df[~np.isnan(df[key + '_mean_1'])]
                df = df.reset_index(drop=True)
                return df
        else:
            X.load()
            return X
