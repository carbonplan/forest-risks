import sys

import numpy as np
from tqdm import tqdm

from carbonplan_forests import fit, load, prepare, utils

args = sys.argv

if len(args) < 2:
    store = 'local'
else:
    store = args[1]

data_vars = ['ppt', 'tavg']
data_aggs = ['sum', 'mean']

print('[insects] loading data')
df = load.fia(store=store, states='conus', group_repeats=True)
df = load.terraclim(
    store=store,
    tlim=(int(df['year_0'].min()), 2020),
    data_vars=data_vars,
    data_aggs=data_aggs,
    df=df,
    group_repeats=True,
)

print('[insects] prepare for fitting')
x, y, pf = prepare.insects(df)
x_z, x_mean, x_std = utils.zscore_2d(x)
codes = pf['type_code'].unique()

print('[insects] fit models')
models = {}
for code in tqdm(codes):
    inds = pf['type_code'] == code
    if (y[inds].sum() > 1) & (y[inds].sum() > 1):
        model = fit.hurdle(x=x_z[inds], y=y[inds])
        models[code] = model

print('[insects] preparing for evaluations')
df = load.fia(store=store, states='conus')
pf = df[['lat', 'lon', 'type_code']].copy().reset_index(drop=True)

print('[insects] evaluating predictions on historical data')
df = load.terraclim(
    store=store,
    tlim=(2005, 2014),
    data_vars=data_vars,
    data_aggs=data_aggs,
    df=df,
)
x, meta = prepare.insects(df, eval_only=True, duration=10)
x_z = utils.zscore_2d(x, x_mean, x_std)
for code in codes:
    if code in models.keys():
        model = models[code]
        inds = df['type_code'] == code
        pf.loc[inds, 'historical'] = model.predict(x=x_z[inds])

print('[insects] evaluating on future climate models')
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
            x, meta = prepare.insects(df, eval_only=True, duration=10)
            x_z = utils.zscore_2d(x, x_mean, x_std)
            for code in codes:
                if code in models.keys():
                    model = models[code]
                    inds = df['type_code'] == code
                    pf.loc[inds, key] = model.predict(x=x_z[inds])

pf['r2'] = pf['type_code'].map(lambda k: models[k].train_r2 if k in models.keys() else np.NaN)

pf.to_parquet('data/insects.parquet', compression='gzip', engine='fastparquet')
