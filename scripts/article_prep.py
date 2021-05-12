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
            return super(NpEncoder, self).default(obj)


warnings.filterwarnings('ignore')

account_key = os.environ.get('BLOB_ACCOUNT_KEY')
gcms = ['CanESM5-CanOE', 'MIROC-ES2L', 'ACCESS-CM2', 'ACCESS-ESM1-5', 'MRI-ESM2-0', 'MPI-ESM1-2-LR']
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

results_dict = {}
# select out bounding boxes
for impact in ['fire', 'insects', 'drought']:
    results_dict[impact] = {}
    if impact == 'fire':
        store_path = 'risks/results/web/{}_cmip_high_res.zarr'.format(impact)
    else:
        store_path = 'risks/results/paper/{}_cmip.zarr'.format(impact)
    ds = xr.open_zarr(
        get_store(
            'carbonplan-forests',
            store_path,
            account_key=account_key,
        )
    )

    if impact == 'fire':
        # bring fire into same temporal scale as insects/drought
        # this takes a while but that's okay
        ds = ds.assign_coords(
            {
                "x": website_mask.x,
                "y": website_mask.y,
            }
        )
        ds = ds.groupby('time.year').sum().coarsen(year=10).mean().compute()
        ds = ds.assign_coords({'year': np.arange(1970, 2100, 10)})
    ds = ds.where(website_mask > 0)

    # then do rolling mean for two decades
    ds = ds.rolling(year=2).mean().drop_sel(year=1970)
    for region, bbox in region_bboxes.items():
        results_dict[impact][region] = {}
        selected = ds.sel(**region_bboxes[region])
        if impact == 'fire':
            selected = selected.apply(utils.integrated_risk)
        else:
            selected *= 10

        selected = selected.mean(dim=['x', 'y']).compute().probability
        # first populate the historical values
        results_dict[impact][region]['historical'] = {'mean': [], 'models': {}}
        # initialize your dictionary with the gcm keys
        for gcm in gcms:
            results_dict[impact][region]['historical']['models'][gcm] = []
        print('at historical')
        for year in np.arange(1980, 2020, 10):
            # average across scenarios
            results_dict[impact][region]['historical']['mean'].append(
                {
                    'y': year,
                    'r': selected.mean(dim='scenario').mean(dim='gcm').sel(year=year).values,
                }
            )
            # then populate all the gcms
            for gcm in gcms:
                results_dict[impact][region]['historical']['models'][gcm].append(
                    {'y': year, 'r': selected.mean(dim='scenario').sel(gcm=gcm, year=year).values}
                )

        for scenario in scenarios:
            print(scenario)
            results_dict[impact][region][scenario] = {}
            results_dict[impact][region][scenario] = {'mean': [], 'models': {}}
            # initialize your dictionary with the gcm keys
            for gcm in gcms:
                results_dict[impact][region][scenario]['models'][gcm] = []
            # loop through scenarios
            for year in np.arange(2020, 2100, 10):

                results_dict[impact][region][scenario]['mean'].append(
                    {
                        'y': year,
                        'r': selected.sel(scenario=scenario).mean(dim='gcm').sel(year=year).values,
                    }
                )
                # then populate all the gcms
                for gcm in gcms:
                    results_dict[impact][region][scenario]['models'][gcm].append(
                        {'y': year, 'r': selected.sel(gcm=gcm, scenario=scenario, year=year).values}
                    )

with fsspec.open(
    'az://carbonplan-forests/risks/results/web/time-series.json',
    account_name="carbonplan",
    account_key=account_key,
    mode='w',
) as f:
    json.dump(results_dict, f, indent=2, cls=NpEncoder)
