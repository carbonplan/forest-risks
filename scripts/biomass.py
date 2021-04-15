import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from carbonplan_forest_risks import fit, load

args = sys.argv

if len(args) < 2:
    store = 'local'
else:
    store = args[1]

variables = ['tmean', 'ppt']

print('[biomass] loading data')
df = load.fia(store=store, states='conus')
df = load.terraclim(
    store=store,
    tlim=(2000, 2020),
    variables=variables,
    remove_nans=True,
    sampling='annual',
    df=df,
)
type_codes = df['type_code'].unique()

print('[biomass] fitting models')
models = {}
for code in tqdm(type_codes):
    df_type = df[df['type_code'] == code].reset_index()
    x = df_type['age']
    y = df_type['biomass']
    f = [df_type['tmean_mean'], df_type['ppt_mean']]
    model = fit.growth(x=x, y=y, f=f, noise='gamma')
    models[code] = model

print('[biomass] preparing for evaluations')
pf = pd.DataFrame()
pf['lat'] = df['lat']
pf['lon'] = df['lon']
pf['type_code'] = df['type_code']

print('[biomass] evaluating predictions on training data')
for code in type_codes:
    if code in models.keys():
        model = models[code]
        inds = df['type_code'] == code
        x = df[inds]['age']
        f = [df[inds]['tmean_mean'], df[inds]['ppt_mean']]
        pf.loc[inds, 'historical'] = model.predict(x, f)

print('[biomass] evaluating predictions on future climate models')
targets = list(map(lambda x: str(x), np.arange(2005, 2100, 10)))
cmip_models = [
    'CanESM5-CanOE',
    'MIROC-ES2L',
    'ACCESS-CM2',
    'ACCESS-ESM1-5',
    'MRI-ESM2-0',
    'MPI-ESM1-2-LR',
]
scenarios = ['ssp245', 'ssp370', 'ssp585']
for it in tqdm(range(len(targets))):
    target = targets[it]
    tlim = (str(int(target) - 5), str(int(target) + 4))
    for cmip_model in cmip_models:
        for scenario in scenarios:
            key = cmip_model + '_' + scenario + '_' + target
            df = load.cmip(
                store=store,
                tlim=(int(tlim[0]), int(tlim[1])),
                variables=variables,
                historical=True if int(tlim[0]) < 2015 else False,
                model=cmip_model,
                scenario=scenario,
                sampling='annual',
                df=df,
            )
            pf[key] = np.NaN
            for code in type_codes:
                if code in models.keys():
                    model = models[code]
                    inds = df['type_code'] == code
                    x = df[inds]['age']
                    year = df[inds]['year']
                    f = [df[inds]['tmean_mean'], df[inds]['ppt_mean']]
                    if it == 0:
                        pf.loc[inds, key] = model.predict(
                            np.maximum(x + (float(target) - year), 0), f
                        )
                    else:
                        prev_target = targets[it - 1]
                        prev_key = cmip_model + '_' + scenario + '_' + prev_target
                        diff = model.predict(x + (float(target) - year), f) - model.predict(
                            x + (float(prev_target) - year), f
                        )
                        pf.loc[inds, key] = np.maximum(pf[prev_key][inds] + diff, 0)

pf['r2'] = pf['type_code'].map(lambda k: models[k].train_r2)
pf['scale'] = pf['type_code'].map(lambda k: models[k].scale)

pf.to_parquet('data/biomass.parquet', compression='gzip', engine='fastparquet')
