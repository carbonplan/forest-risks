import xarray as xr
from carbonplan.data import cat


def tiff(url, lat_coords, lon_coords, x_coords, y_coords, coarsen=4):
    target = cat.nlcd.raster.to_dask()
    source = xr.open_rasterio(url)
    source = source.where(source > -1)
    ds = source.rio.reproject_match(target)
    ds = ds.where(ds > -1).coarsen(x=coarsen, y=coarsen, boundary="trim").mean().sel(band=1)
    # make sure that the coordinates are *exactly* aligned- otherwise you'll have
    # pesky plotting peculiarities
    ds = ds.assign_coords({"x": x_coords, "y": y_coords})
    ds = ds.assign_coords({"lat": lat_coords, "lon": lon_coords})

    return ds
