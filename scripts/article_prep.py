# This script will prepare the variable timeseries for the web article.
# TODO: temperature

import json
import os
import warnings

import fsspec
import numpy as np
import rioxarray
import xarray as xr

from carbonplan_forest_risks import load, utils
from carbonplan_forest_risks.utils import get_store

# flake8: noqa


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


warnings.filterwarnings('ignore')

account_key = os.environ.get('BLOB_ACCOUNT_KEY')
gcms = [
    ('CanESM5-CanOE', 'r3i1p2f1'),
    ('MIROC-ES2L', 'r1i1p1f2'),
    ('ACCESS-CM2', 'r1i1p1f1'),
    ('ACCESS-ESM1-5', 'r10i1p1f1'),
    ('MRI-ESM2-0', 'r1i1p1f1'),
    ('MPI-ESM1-2-LR', 'r10i1p1f1'),
]
scenarios = ['ssp245', 'ssp370', 'ssp585']

website_mask = (
    load.nlcd(store="az", year=2016).sel(band=[41, 42, 43, 90]).sum("band") > 0.5
).astype("float")

region_bboxes = {
    'CONUS': {'x': slice(-3e6, 3e6), 'y': slice(4e6, 0)},
    'PNW': {'x': slice(-2.5e6, -1e6), 'y': slice(3.5e6, 2.4e6)},
    'Southwest': {'x': slice(-1.8e6, -0.9e6), 'y': slice(1.8e6, 0.9e6)},
    'California': {'x': slice(-2.3e6, -1.8e6), 'y': slice(2.5e6, 1.2e6)},
    'Southeast': {
        'x': slice(0.6e6, 1.8e6),
        'y': slice(1.6e6, 0.3e6),
    },
}


def build_climate_cube(
    tlim=(1970, 2099), variables=['tmean'], downscaling_method='quantile-mapping-v3'
):
    """"""
    gcms = [
        ('CanESM5-CanOE', 'r3i1p2f1'),
        ('MIROC-ES2L', 'r1i1p1f2'),
        ('ACCESS-CM2', 'r1i1p1f1'),
        ('ACCESS-ESM1-5', 'r10i1p1f1'),
        ('MRI-ESM2-0', 'r1i1p1f1'),
        ('MPI-ESM1-2-LR', 'r10i1p1f1'),
    ]
    scenarios = ['ssp245', 'ssp370', 'ssp585']
    all_scenarios = []
    for scenario in scenarios:
        all_gcms = []
        for (gcm, ensemble_member) in gcms:
            cmip = load.cmip(
                store='az',
                model=gcm,
                tlim=tlim,
                scenario=scenario,
                historical=True,
                member=ensemble_member,
                method=downscaling_method,
                sampling='annual',
                variables=variables,
            )
            all_gcms.append(cmip)
        concatted = xr.concat(all_gcms, dim='gcm')
        concatted = concatted.assign_coords({'gcm': [gcm[0] for gcm in gcms]})
        all_scenarios.append(concatted)
    ds = xr.concat(all_scenarios, dim='scenario')
    ds = ds.assign_coords({'scenario': scenarios})
    return ds


def repackage_drought_insects(ds):
    gcms = [
        ('CanESM5-CanOE', 'r3i1p2f1'),
        ('MIROC-ES2L', 'r1i1p1f2'),
        ('ACCESS-CM2', 'r1i1p1f1'),
        ('ACCESS-ESM1-5', 'r10i1p1f1'),
        ('MRI-ESM2-0', 'r1i1p1f1'),
        ('MPI-ESM1-2-LR', 'r10i1p1f1'),
    ]
    scenarios = ['ssp245', 'ssp370', 'ssp585']

    all_gcms = []
    for (gcm, ensemble_member) in gcms:
        all_gcms.append(
            ds[[f'{gcm}-{scenario}' for scenario in scenarios]]
            .to_array(dim='scenario', name='probability')
            .assign_coords({'scenario': scenarios})
        )
    full_ds = xr.concat(all_gcms, dim='gcm').to_dataset()
    full_ds = full_ds.assign_coords({'gcm': [gcm for (gcm, ensemble_member) in gcms]})
    return full_ds


def timeseries_dict(ds, time_period='historical'):
    gcms = [
        ('CanESM5-CanOE', 'r3i1p2f1'),
        ('MIROC-ES2L', 'r1i1p1f2'),
        ('ACCESS-CM2', 'r1i1p1f1'),
        ('ACCESS-ESM1-5', 'r10i1p1f1'),
        ('MRI-ESM2-0', 'r1i1p1f1'),
        ('MPI-ESM1-2-LR', 'r10i1p1f1'),
    ]

    mean = []
    gcm_dict = {}
    for (gcm, ensemble_member) in gcms:
        gcm_dict[gcm] = []
    if time_period == 'historical':
        years = np.arange(1980, 2020, 10)
    elif time_period == 'future':
        years = np.arange(2020, 2100, 10)
    for year in years:
        # average across scenarios
        mean.append(
            {
                'y': year,
                'r': ds.mean(dim='gcm').sel(year=year).values.item(),
            }
        )
        # then populate all the gcms
        for (gcm, ensemble_member) in gcms:
            gcm_dict[gcm].append({'y': year, 'r': ds.sel(gcm=gcm, year=year).values.item()})
    return mean, gcm_dict


# first print the bounding box coordinates

region_bboxes_lists = {}
slice_to_list = lambda x: [x.start, x.stop]
for region in region_bboxes:
    region_bboxes_lists[region] = {}
    for coord in ['x', 'y']:
        region_bboxes_lists[region][coord] = slice_to_list(region_bboxes[region][coord])
with fsspec.open(
    'az://carbonplan-forests/risks/results/web/time-series-bounding-boxes.json',
    account_name="carbonplan",
    account_key=account_key,
    mode='w',
) as f:
    json.dump(region_bboxes_lists, f, indent=2, cls=NpEncoder)

# second grab regionally-averaged reults and write them into this dictionary
# populate your results dict according to the format which will be used
# for the web article rendering (nested by impact, region, scenario, models/multi-model mean, year/risk)
results_dict = {}
# select out bounding boxes
# select out bounding boxes
for impact in ['insects', 'drought', 'fire', 'tmean']:
    results_dict[impact] = {}
    # read in the temperature data from its different sources and create a datacube
    # of the same specs as the risks
    if impact == 'tmean':
        ds = build_climate_cube()
    # grab the risks data
    else:
        store_path = f'risks/results/web/{impact}_full.zarr'
        ds = xr.open_zarr(
            get_store(
                'carbonplan-forests',
                store_path,
                account_key=account_key,
            )
        )
        ds = ds.assign_coords({'year': np.arange(1980, 2100, 10)})

    if impact in ['insects', 'drought']:
        # restructure the insects/drought ones to align with the temp/fire
        ds = repackage_drought_insects(ds)

    # assign the coords for all of the data sources (this helps make sure that
    # the masking works appropriately and coordinates aren't off by 0.00000001)
    print(ds)
    ds = ds.assign_coords(
        {
            "x": website_mask.x,
            "y": website_mask.y,
        }
    )
    # align to the annual timesteps for tmean and fire
    if impact == 'tmean':
        # calculate decadal mean
        ds = ds.coarsen(time=10).mean().compute()
        ds = ds.rename({'time': 'year'})
        ds = ds.assign_coords({'year': np.arange(1970, 2100, 10)})
        # mask according to the mask we use for the web
        ds = ds.where(website_mask > 0).compute()
        # then do rolling mean for two decades (and drop the first timestep which only
        # has info for one decade). we'll report 20 year risks at 10 year increments
        ds = ds.rolling(year=2).mean().drop_sel(year=1970)

    # loop through each of the regions of interest
    for region, bbox in region_bboxes.items():
        print(f'Calculating regional averages over the {region} for {impact}')
        # initialize the dictionary
        results_dict[impact][region] = {}
        # select out the box you want
        selected = ds.sel(**region_bboxes[region])
        # aggregate the different risks according to either 20 year integrated risk for fire
        # or just multiply by 100 to convert to percentage and then by 20 for the 20 year
        # total mortality for insects/drought
        if impact == 'fire':
            selected = selected.apply(utils.integrated_risk)
        elif impact in ['drought', 'insects']:
            selected *= 100 * 20
        # calculate regional averages (these have already been masked) and then select the
        # appropriate variable
        if impact == 'tmean':
            selected = selected.mean(dim=['x', 'y']).compute().tmean
            # create anomalies
            selected -= selected.sel(year=[1980, 1990, 2000]).mean(dim='year')
        else:
            selected = selected.mean(dim=['x', 'y']).compute().probability

        # first populate the historical values
        results_dict[impact][region]['historical'] = {}

        # initialize your dictionary with the gcm keys
        mean, models = timeseries_dict(selected.mean(dim='scenario'), time_period='historical')
        results_dict[impact][region]['historical']['mean'] = mean
        results_dict[impact][region]['historical']['models'] = models

        # then fill in each of the three different scenarios

        for scenario in scenarios:
            results_dict[impact][region][scenario] = {}

            # initialize your dictionary with the gcm keys
            mean, models = timeseries_dict(selected.sel(scenario=scenario), time_period='future')
            results_dict[impact][region][scenario]['mean'] = mean
            results_dict[impact][region][scenario]['models'] = models

# write out to dictionary to rendered within the explainer
with fsspec.open(
    'az://carbonplan-forests/risks/results/web/time-series-hybrid.json',
    account_name="carbonplan",
    account_key=account_key,
    mode='w',
) as f:
    json.dump(results_dict, f, indent=2, cls=NpEncoder)
