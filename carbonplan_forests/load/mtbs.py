import os
import warnings

import fsspec
import numpy as np
import xarray as xr
import zarr

from .. import setup


def mtbs(store='az', tlim=(1984, 2018), mask=None, coarsen=None):
    path = setup.loading(store)

    if store == 'az':
        prefix = 'processed/mtbs/conus/4000m/monthly.zarr'
        mapper = zarr.storage.ABSStore(
            'carbonplan-data',
            prefix=prefix,
            account_name='carbonplan',
            account_key=os.environ['BLOB_ACCOUNT_KEY'],
        )
    else:
        prefix = (path / 'carbonplan-data/processed/mtbs/conus/4000m/monthly.zarr').as_uri()
        mapper = fsspec.get_mapper(prefix)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)

        mtbs = xr.open_zarr(mapper, consolidated=True)

        if tlim:
            tlim = list(map(str, tlim))
            mtbs = mtbs.sel(time=slice(*tlim))

        if mask is not None:
            vals = mask.values
            vals[vals == 0] = np.NaN
            mtbs = mtbs * vals

        if coarsen:
            mtbs = mtbs.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()

        mtbs.load()
        return mtbs
