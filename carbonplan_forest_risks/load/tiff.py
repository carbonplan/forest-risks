import numpy as np
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
    # ds = ds.assign_coords({"lat": model_ds.lat, "lon": model_ds.lon})

    return ds


def impacts(url_template, spatial_template, mask=None, coarsen=1):
    start_years = np.arange(1970, 2100, 10)
    end_years = np.arange(1979, 2109, 10)
    full_ds = xr.Dataset()
    full_ds_list = []
    scenarios = ["ssp245", "ssp370", "ssp585"]

    gcms = [
        ("ACCESS-CM2", "r1i1p1f1"),
        ("ACCESS-ESM1-5", "r10i1p1f1"),
        ("MRI-ESM2-0", "r1i1p1f1"),
        ("MIROC-ES2L", "r1i1p1f2"),
        ("MPI-ESM1-2-LR", "r10i1p1f1"),
        ("CanESM5-CanOE", "r3i1p2f1"),
    ]

    for scenario in scenarios:
        impact_ds = xr.Dataset()

        for (gcm, ensemble_member) in gcms:
            year_coords, impact_ds_list = [], []
            for start_year, end_year in zip(start_years, end_years):
                #                 try:
                if start_year < 2005:
                    url = url_template.format(
                        gcm, 'historical', ensemble_member, start_year, end_year
                    )
                else:
                    url = url_template.format(gcm, scenario, ensemble_member, start_year, end_year)
                year_coords.append(start_year + 5)
                impact_ds_list.append(tiff(url, spatial_template, coarsen=coarsen).load())
            impact_ds[gcm] = xr.concat(impact_ds_list, 'year').assign_coords({'year': year_coords})
        full_ds_list.append(impact_ds)
    full_ds = xr.concat(full_ds_list, 'scenario').assign_coords({'scenario': scenarios})
    if mask is not None:
        full_ds = full_ds.where(mask > 0)
    return full_ds
