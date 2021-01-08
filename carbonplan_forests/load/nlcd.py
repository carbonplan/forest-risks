import numpy as np
import xarray as xr

from .. import setup


def nlcd(store='az', classes='all', year=2001, coarsen=None, mask=None):
    path = setup.loading(store)

    if classes == 'all':
        classes = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95]

    bands = xr.concat(
        [
            xr.open_rasterio(
                (path / f'carbonplan-data/processed/nlcd/conus/4000m/{year}_c{c}.tif').as_uri()
            )
            for c in classes
        ],
        dim=xr.Variable('band', classes),
    )

    if mask is not None:
        vals = mask.values
        vals[vals == 0] = np.NaN
        bands = bands * vals

    if coarsen is not None:
        bands = bands.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()

    bands.load()
    return bands
