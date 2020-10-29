import sys

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from carbonplan_forests import fit, load

args = sys.argv

if len(args) < 1:
    raise ValueError('must specify dataset')
dataset = args[1]

if len(args) == 1:
    store = 'local'
else:
    store = args[2]

cmip_model = 'BCC-CSM2-MR'
scenarios = ['ssp245', 'ssp370', 'ssp585']

targets = list(map(lambda x: str(x), np.arange(2020, 2120, 20)))
pf = pd.read_parquet(f'data/{dataset}.parquet')
ds = xr.Dataset()
pf = pf.dropna().reset_index(drop=True)

if dataset in ['drought', 'insects']:
    badinds = (pf['historical'] > 1) | (np.isnan(pf['historical']))
    for key in pf.columns:
        if key not in ['lat', 'lon', 'type_code', 'r2']:
            badinds = badinds | ((pf[key] > 1) | (np.isnan(pf[key])))
    pf = pf[~badinds]

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
        key = cmip_model + '_' + scenario + '_' + target
        gridded = fit.interp(pf, final_mask, var=key)
        results.append(gridded)
    da = xr.concat(results, dim=xr.Variable('year', targets))
    ds[cmip_model + '_' + scenario] = da

ds.to_zarr(f'data/{dataset}.zarr')
