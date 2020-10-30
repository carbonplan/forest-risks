import sys

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from carbonplan_forests import fit, load

args = sys.argv

if len(args) < 2:
    raise ValueError('must specify dataset')
dataset = args[1]

if len(args) == 2:
    store = 'local'
else:
    store = args[2]

cmip_models = ['BCC-CSM2-MR', 'ACCESS-ESM1-5', 'CanESM5', 'MIROC6', 'MPI-ESM1-2-LR']
scenarios = ['ssp245', 'ssp370', 'ssp585']

targets = list(map(lambda x: str(x), np.arange(2020, 2120, 20)))
pf = pd.read_parquet(f'data/{dataset}.parquet')
ds = xr.Dataset()

print(f'[{dataset}] filtering values')
pf = pf.dropna().reset_index(drop=True)
if dataset in ['drought', 'insects']:
    badinds = (pf['historical'] > 1) | (np.isnan(pf['historical']))
    for key in pf.columns:
        if key not in ['lat', 'lon', 'type_code', 'r2']:
            badinds = badinds | ((pf[key] > 1) | (np.isnan(pf[key])))
    pf = pf[~badinds]

print(f'[{dataset}] computing multi model mean')
for scenario in scenarios:
    for target in targets:
        keys = list(
            filter(
                lambda x: x is not None,
                [key if ((scenario in key) & (target in key)) else None for key in pf.columns],
            )
        )
        pf[scenario + '_' + target] = pf[keys].mean(axis=1)

print(f'[{dataset}] regridding predictions')
final_mask = load.nlcd(store=store, year=2016, classes=[41, 42, 43, 90])
if dataset == 'biomass':
    final_mask.values = final_mask.values * (final_mask.values > 0.5)
else:
    final_mask.values = final_mask.values > 0.5

ds['historical'] = fit.interp(pf, final_mask, var='historical')

for scenario in tqdm(scenarios):
    results = []
    for target in targets:
        key = scenario + '_' + target
        gridded = fit.interp(pf, final_mask, var=key)
        results.append(gridded)
    da = xr.concat(results, dim=xr.Variable('year', targets))
    ds[scenario] = da

ds.to_zarr(f'data/{dataset}.zarr')
