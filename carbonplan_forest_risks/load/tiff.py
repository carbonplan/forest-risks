import xarray as xr
from carbonplan.data import cat


def tiff(url, model_ds, coarsen=1):
    target = cat.nlcd.raster.to_dask()
    source = xr.open_rasterio(url)
    source = source.where(source > -1)
    ds = source.rio.reproject_match(target)
    ds = (
        ds.where(ds > -1)
        .coarsen(x=coarsen, y=coarsen, boundary="trim")
        .mean()
        .sel(band=1)
        .drop('band')
        .drop('spatial_ref')
    )
    # make sure that the coordinates are *exactly* aligned- otherwise you'll have
    # pesky plotting peculiarities
    ds = ds.assign_coords({"x": model_ds.x, "y": model_ds.y})
    ds = ds.assign_coords({"lat": model_ds.lat, "lon": model_ds.lon})

    return ds
