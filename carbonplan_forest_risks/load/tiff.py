import numpy as np
import xarray as xr
from carbonplan.data import cat
from rasterio.enums import Resampling


def tiff(url, model_ds, coarsen=1):
    target = cat.nlcd.raster.to_dask()
    source = xr.open_rasterio(url)
    source = source.where(source > -1)
    ds = source.rio.reproject_match(target, resampling=Resampling.bilinear)
    ds = ds.where(ds > -1).sel(band=1).drop('band').drop('spatial_ref').astype(np.float64)
    if coarsen != 1:
        ds = ds.coarsen(x=coarsen, y=coarsen, boundary="trim").mean()
    # make sure that the coordinates are *exactly* aligned- otherwise you'll have
    # pesky plotting peculiarities
    ds = ds.assign_coords({"x": model_ds.x, "y": model_ds.y})

    return ds


def impacts(
    url_template,
    spatial_template,
    mask=None,
    period_start=1970,
    period_end=2100,
    coarsen=1,
    met_data='cmip',
):
    start_years = np.arange(period_start, period_end, 10)
    end_years = np.arange(period_start + 9, period_end + 9, 10)
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
    if met_data == 'cmip':
        for scenario in scenarios:
            impact_ds = xr.Dataset()
            for (gcm, ensemble_member) in gcms:
                year_coords, impact_ds_list = [], []
                for start_year, end_year in zip(start_years, end_years):
                    if start_year < 2005:
                        url = url_template.format(
                            gcm, 'historical', ensemble_member, start_year, end_year
                        )
                    else:
                        url = url_template.format(
                            gcm, scenario, ensemble_member, start_year, end_year
                        )
                    impact_ds_list.append(tiff(url, spatial_template, coarsen=coarsen).load())
                    year_coords.append(start_year)
                impact_ds[gcm] = xr.concat(impact_ds_list, 'year').assign_coords(
                    {'year': year_coords}
                )
            full_ds_list.append(impact_ds)
        full_ds = xr.concat(full_ds_list, 'scenario').assign_coords({'scenario': scenarios})

    elif met_data == 'terraclimate':
        year_coords, impact_ds_list = [], []
        for start_year, end_year in zip(start_years, end_years):
            url = url_template.format(start_year, end_year)
            impact_ds_list.append(tiff(url, spatial_template, coarsen=coarsen).load())
            year_coords.append(start_year)
        full_ds = xr.concat(impact_ds_list, 'year').assign_coords({'year': year_coords})
    if mask is not None:
        full_ds = full_ds.where(mask > 0)
    return full_ds
