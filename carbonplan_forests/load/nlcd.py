import rasterio
import xarray as xr

from .. import setup


def load_rio(f):
    src = rasterio.open(f)
    return src.read(1)


def nlcd(store='az', classes=[41, 42, 43, 90], year=2001, coarsen=None):
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
    mask = bands.sum('band', keep_attrs=True)

    if coarsen:
        mask = mask.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()

    mask.load()
    return mask
