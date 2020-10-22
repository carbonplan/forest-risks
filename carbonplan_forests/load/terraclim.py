import warnings

import fsspec
import numpy as np
import xarray as xr
from pyproj import Proj, transform
from rasterio import Affine
from rasterio.transform import rowcol

from .. import setup, utils


def terraclim(
    store='gcs',
    df=None,
    tlim=None,
    mean=False,
    coarsen=None,
    data_vars=['ppt', 'tmax'],
    aggs=None,
    mask=True
):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ResourceWarning)
        warnings.simplefilter('ignore', category=FutureWarning)
        warnings.simplefilter('ignore', category=RuntimeWarning)

        path = setup.loading(store)
        mapper = fsspec.get_mapper(
            (path / 'processed/terraclimate/conus/4000m/raster.zarr').as_uri()
        )

        ds = xr.open_zarr(mapper)

        X = xr.Dataset()

        def weighted_mean(ds, *args, **kwargs):
            weights = ds.time.dt.days_in_month
            return ds.weighted(weights).mean(dim='time')

        if aggs is not None:
            keys = [var + '_' + agg for var, agg in zip(data_vars, aggs)]
            for key in keys:
                var, agg = key.split('_')
                base = ds[var].resample(time='AS')
                if agg == 'sum':
                    X[key] = base.sum('time')
                elif agg == 'mean':
                    X[key] = base.map(weighted_mean, dim='time')
                elif agg == 'max':
                    X[key] = base.max('time')
                elif agg == 'min':
                    X[key] = base.min('time')
                else:
                    raise ValueError(f'agg method {agg} not supported')
                if mean is True:
                    X[key] = X[key].mean('time')
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
            ind_r = xr.DataArray(rc[0], dims=['x'])
            ind_c = xr.DataArray(rc[1], dims=['x'])
            for key in keys:
                df[key] = X[key][ind_r, ind_c].values
            df = df.dropna().reset_index(drop=True)
            return df

        else:
            X.load()
            return X