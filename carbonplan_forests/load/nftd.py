import rasterio
import xarray as xr

from .. import setup


def load_rio(f):
    src = rasterio.open(f)
    return src.read(1)


def nftd(store='gcs', groups='all', coarsen=None):
    path = setup.loading(store)

    if groups == 'all':
        groups = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320,
           340, 360, 370, 380, 400, 500, 600, 700, 800, 900, 910, 920, 940,
           950, 980, 990]

    bands = xr.concat(
        [xr.open_rasterio(path / f'processed/nftd/conus/4000m/group_g{g}.tif')[0] for g in groups],
        dim=xr.Variable('band', groups),
    )

    if coarsen:
        bands = bands.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()

    bands.load()
    return bands
