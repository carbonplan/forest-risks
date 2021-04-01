import os

import numpy as np
import xarray as xr

from carbonplan_forest_risks import load
from carbonplan_forest_risks.utils import get_store

impacts_to_consolidate = ['fire', 'insects', 'drought']
account_key = os.environ.get('BLOB_ACCOUNT_KEY')

# specify the kind of mask you want to use
website_mask = (
    load.nlcd(store="az", year=2001).sel(band=[41, 42, 43, 90]).sum("band") > 0.5
).astype("float")
gcms = [
    ("ACCESS-CM2", "r1i1p1f1"),
    ("ACCESS-ESM1-5", "r10i1p1f1"),
    ("MRI-ESM2-0", "r1i1p1f1"),
    ("MIROC-ES2L", "r1i1p1f2"),
    ("MPI-ESM1-2-LR", "r10i1p1f1"),
    ("CanESM5-CanOE", "r3i1p2f1"),
]


def initialize_empty():
    historical_fire = xr.open_zarr(
        get_store("carbonplan-scratch", "data/fire_historical_v4_high_res.zarr")
    ).load()
    ds = xr.full_like(historical_fire.isel(time=slice(None, 10)), np.nan)
    ds = ds.rename({'time': 'year'})
    ds = ds.assign_coords({"year": np.arange(2005, 2100, 10)})
    return ds


def package_impacts(url_template, mask):
    start_years = np.arange(2000, 2100, 10)
    end_years = np.arange(2009, 2109, 10)
    empty_dataset = initialize_empty()
    full_ds = xr.Dataset()
    for scenario in ["ssp245", "ssp370", "ssp585"]:
        # this initializes empty array that you'll fill
        full_ds[scenario] = empty_dataset['historical']
        ds = xr.Dataset()

        for (gcm, ensemble_member) in gcms:
            ds[gcm] = empty_dataset['historical']
            # want to def initialize_empty
            for start_year, end_year in zip(start_years, end_years):
                try:
                    if start_year == 2000:
                        url = url_template.format(
                            gcm, 'historical', ensemble_member, start_year, end_year
                        )
                    else:
                        url = url_template.format(
                            gcm, scenario, ensemble_member, start_year, end_year
                        )
                    ds[gcm].loc[dict(year=start_year + 5)] = load.tiff(url, ds).load()
                except:
                    print(gcm, scenario, start_year)
        full_ds[scenario] = ds.to_array(dim="vars").mean(dim="vars")
    full_ds = full_ds.where(mask > 0)
    return full_ds


if 'fire' in impacts_to_consolidate:
    run_name = 'v3_high_res'

    all_scenarios_ds = xr.Dataset()
    for scenario in ['ssp245', 'ssp370', 'ssp585']:
        ds = xr.Dataset()
        for (cmip_model, member) in gcms:
            path = get_store(
                'carbonplan-scratch',
                'data/fire_future_{}_{}_{}.zarr'.format(run_name, cmip_model, scenario),
                account_key=account_key,
            )
            ds[cmip_model] = (
                ['year', 'y', 'x'],
                xr.open_zarr(
                    path,  # consolidated=True
                )
                .groupby("time.year")
                .sum()
                .coarsen(year=10)
                .mean()
                .compute()['{}_{}'.format(cmip_model, scenario)],
            )

            print(cmip_model)
        # average across all variables
        ds = ds.assign_coords(
            {"x": website_mask.x, "y": website_mask.y, 'year': np.arange(1975, 2100, 10)}
        ).where(website_mask)
        all_scenarios_ds[scenario] = ds.to_array(dim="vars").mean(dim="vars")

    out_path = get_store('carbonplan-scratch', 'data/website/fire.zarr', account_key=account_key)
    all_scenarios_ds.to_zarr(out_path)

if 'insects' in impacts_to_consolidate:
    insect_url_template = "https://carbonplan.blob.core.windows.net/carbonplan-scratch/from_bill/InsectProjections_3-30/InsectModelProjection_{}.{}.{}-{}.{}-v14climate_3-30-2021.tif"
    ds = package_impacts(insect_url_template, website_mask)
    out_path = get_store('carbonplan-scratch', 'data/website/insects.zarr')
    ds.to_zarr(out_path)

if 'drought' in impacts_to_consolidate:
    drought_url_template = "https://carbonplan.blob.core.windows.net/carbonplan-scratch/from_bill/DroughtProjections_3-31/DroughtModelProjection_{}.{}.{}-{}.{}-v14climate_3-30-2021.tif"
    ds = package_impacts(drought_url_template, website_mask)
    out_path = get_store('carbonplan-scratch', 'data/website/drought.zarr', account_key=account_key)
    ds.to_zarr(out_path)


# # target:
# #for fire insects drought
# zarr

# albers conus
# 4 km
# variables: ssp245 ssp370 ssp585
# impor
# years: 2000 2010 2020 2030 2040, 2050, 2060, 2070, 2080, 2090, 2100
