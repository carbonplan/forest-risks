import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from carbonplan_forests import fit, load

args = sys.argv

if len(args) < 2:
    store = 'local'
else:
    store = args[1]

data_vars = ['tavg', 'ppt']
data_aggs = ['mean', 'sum']

print('[biomass] loading data')
df = load.fia(store=store, states='conus')
df = load.terraclim(
    store=store,
    tlim=(2000, 2020),
    data_vars=data_vars,
    data_aggs=data_aggs,
    remove_nans=True,
    df=df,
)
type_codes = df['type_code'].unique()

print('[biomass] fitting models')
models = {}
for code in tqdm(type_codes):
    df_type = df[df['type_code'] == code].reset_index()
    x = df_type['age']
    y = df_type['biomass']
    f = [df_type['tavg_mean_mean'], df_type['ppt_sum_mean']]
    model = fit.biomass(x=x, y=y, f=f, noise='gamma')
    models[code] = model

with open('data/biomass.pkl', 'wb') as f:
    pickle.dump(models, f)

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
        f = [df[inds]['tavg_mean_mean'], df[inds]['ppt_sum_mean']]
        pf.loc[inds, 'historical'] = model.predict(x, f)

print('[biomass] evaluating predictions on future climate models')
targets = list(map(lambda x: str(x), np.arange(2020, 2120, 20)))
cmip_models = ['BCC-CSM2-MR', 'ACCESS-ESM1-5', 'CanESM5', 'MIROC6', 'MPI-ESM1-2-LR']
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
                data_vars=data_vars,
                data_aggs=data_aggs,
                model=cmip_model,
                scenario=scenario,
                annual=True,
                df=df,
            )
            pf[key] = np.NaN
            for code in type_codes:
                if code in models.keys():
                    model = models[code]
                    inds = df['type_code'] == code
                    x = df[inds]['age']
                    year = df[inds]['year']
                    f = [df[inds]['tavg_mean_mean'], df[inds]['ppt_sum_mean']]
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
