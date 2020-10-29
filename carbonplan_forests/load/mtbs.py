import fsspec
import numpy as np
import xarray as xr

from .. import setup
from ..utils import rowcol_to_latlon


def mtbs(store='az', tlim=(1984, 2018), coarsen=None):
    path = setup.loading(store)
    mapper = fsspec.get_mapper(
        (path / 'carbonplan-data/processed/mtbs/conus/4000m/monthly_perims_raster.zarr').as_uri()
    )
    mtbs = xr.open_zarr(mapper, consolidated=True)
    mtbs['x'] = range(len(mtbs['x']))
    mtbs['y'] = range(len(mtbs['y']))
    mtbs = mtbs.drop('x')
    mtbs = mtbs.drop('y')

    rows = np.tile(mtbs['y'].values[:, np.newaxis], [1, len(mtbs['x'].values)])
    cols = np.tile(mtbs['x'].values, [len(mtbs['y'].values), 1])
    lat, lon = rowcol_to_latlon(rows.flatten(), cols.flatten(), 4000)
    mtbs['lat'] = (['y', 'x'], np.asarray(lat).reshape(len(mtbs['y']), len(mtbs['x'])))
    mtbs['lon'] = (['y', 'x'], np.asarray(lon).reshape(len(mtbs['y']), len(mtbs['x'])))

    if tlim:
        tlim = list(map(str, tlim))
        mtbs = mtbs.sel(time=slice(*tlim))

    if coarsen:
        mtbs = mtbs.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()

    mtbs.load()
    return mtbs
