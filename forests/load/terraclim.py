import warnings

import fsspec
import xarray as xr
from pyproj import Proj, transform
from rasterio import Affine
from rasterio.transform import rowcol

from .. import setup, utils


def terraclim(
    store='gcs',
    df=None,
    tlim=None,
    mean=True,
    coarsen=None,
    vars=['ppt', 'tmax'],
    return_type='xarray',
):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ResourceWarning)
        warnings.simplefilter('ignore', category=FutureWarning)
        warnings.simplefilter('ignore', category=RuntimeWarning)

        if tlim is None:
            raise ValueError('must specify time limits as tlim')
        tlim = list(map(str, tlim))

        path = setup.loading(store)
        mapper = fsspec.get_mapper(
            (path / 'processed/terraclimate/conus/4000m/raster.zarr').as_uri()
        )

        ds = xr.open_zarr(mapper)

        X = xr.Dataset()

        def weighted_mean(ds, *args, **kwargs):
            weights = ds.time.dt.days_in_month
            return ds.weighted(weights).mean(dim='time')

        for var in vars:
            if var == 'ppt':
                X[var] = ds[var].resample(time='AS').sum('time').sel(time=slice(*tlim))
            else:
                X[var] = (
                    ds[var]
                    .resample(time='AS')
                    .map(weighted_mean, dim='time')
                    .sel(time=slice(*tlim))
                )
            if mean is True:
                X[var] = X[var].mean('time')

        if coarsen:
            X_coarse = xr.Dataset()
            for var in vars:
                X_coarse[var] = X[var].coarsen(x=coarsen, y=coarsen, boundary='trim').mean()
            X = X_coarse

        if df is not None:
            t = Affine(*utils.albers_conus_transform(4000))
            p1 = Proj(utils.albers_conus_crs())
            p2 = Proj(proj='latlong', datum='WGS84')
            x, y = transform(p2, p1, df['lon'].values, df['lat'].values)
            rc = rowcol(t, x, y)
            ind_r = xr.DataArray(rc[0], dims=['x'])
            ind_c = xr.DataArray(rc[1], dims=['x'])
            for var in vars:
                df[var] = X[var][ind_r, ind_c].values
            df = df.dropna().reset_index(drop=True)
            return df

        else:
            if return_type == 'xarray':
                X.load()
                return X

            if return_type == 'numpy':
                X = {var: X[var].values for var in X}
                return X
