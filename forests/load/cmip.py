import warnings

import fsspec
import xarray as xr
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
    vars=['ppt', 'tmax'],
    aggs=['sum', 'mean'],
    return_type='xarray',
):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ResourceWarning)
        warnings.simplefilter('ignore', category=FutureWarning)
        warnings.simplefilter('ignore', category=RuntimeWarning)

        if tlim is None:
            raise ValueError('must specify time limits as tlim')
        if scenario is None:
            raise ValueError('must specify scenario')
        if model is None:
            raise ValueError('must specify model')
        tlim = list(map(str, tlim))

        path = setup.loading(store)
        mapper = fsspec.get_mapper(
            (path / f'scratch/downscaling/bias-correction-annual/{model}.{scenario}.zarr').as_uri()
        )

        ds = xr.open_zarr(mapper, consolidated=True)

        X = xr.Dataset()

        keys = [var + '_' + agg for var, agg in zip(vars, aggs)]

        for key in keys:
            X[key] = ds[key][0].sel(time=slice(*tlim))
            if 'tmax' in key or 'tmin' in key:
                X[key] = X[key] - 273.15
            if mean is True:
                X[key] = X[key].mean('time')

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
            if return_type == 'xarray':
                X.load()
                return X

            if return_type == 'numpy':
                X = {key: X[key].values for key in X}
                return X
