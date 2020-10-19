import pickle

import numpy as np
import pandas as pd
from carbonplan.forests import load  # , fit
from tqdm import tqdm

# import xarray as xr


print('[biomass] loading data')
df = load.fia(store='local', states='conus')
df = load.terraclim(
    store='local', tlim=(2000, 2020), mean=True, df=df, vars=['tmax', 'ppt'], aggs=['mean', 'sum']
)
type_codes = df['type_code'].unique()

# print('[biomass] fitting models')
# models = {}
# for code in tqdm(type_codes):
#     df_type = df[df['type_code'] == code].reset_index()
#     if len(df_type) > 30:
#         x = df_type['age']
#         y = df_type['biomass']
#         f = [df_type['tmax_mean'], df_type['ppt_sum']]
#         model = fit.biomass(x=x, y=y, f=f, noise='gamma')
#         models[code] = model

# with open('data/biomass.pkl', 'wb') as f:
#     pickle.dump(models, f)

with open('data/biomass.pkl', 'rb') as f:
    models = pickle.load(f)

pf = pd.DataFrame()
pf['lat'] = df['lat']
pf['lon'] = df['lon']
pf['type_code'] = df['type_code']

targets = list(map(lambda x: str(x), np.arange(2020, 2120, 20)))

print('[biomass] evaluating predictions')
cmip_model = 'BCC-CSM2-MR'
scenarios = ['ssp245', 'ssp370', 'ssp585']
for it in tqdm(range(len(targets))):
    target = targets[it]
    tlim = (str(int(target) - 10), str(int(target) + 9))
    if target == '2010':
        df = load.terraclim(
            store='local', tlim=tlim, vars=['tmax', 'ppt'], aggs=['mean', 'sum'], mean=True, df=df
        )
    for scenario in scenarios:
        key = cmip_model + '_' + scenario + '_' + target
        if target != '2010':
            df = load.cmip(
                store='local',
                tlim=(int(tlim[0]), int(tlim[1])),
                vars=['tmax', 'ppt'],
                aggs=['mean', 'sum'],
                model=cmip_model,
                scenario=scenario,
                mean=True,
                df=df,
            )
        pf[key] = np.NaN
        for code in type_codes:
            if code in models.keys():
                model = models[code]
                inds = df['type_code'] == code
                x = df[inds]['age']
                year = df[inds]['year']
                f = [df[inds]['tmax_mean'], df[inds]['ppt_sum']]
                if it == 0:
                    pf.loc[inds, key] = model.predict(np.maximum(x + (float(target) - year), 0), f)
                else:
                    prev_target = targets[it - 1]
                    prev_key = cmip_model + '_' + scenario + '_' + prev_target
                    diff = model.predict(x + (float(target) - year), f) - model.predict(
                        x + (float(prev_target) - year), f
                    )
                    pf.loc[inds, key] = np.maximum(pf[prev_key][inds] + diff, 0)

pf = pf.dropna().reset_index(drop=True)
pf['r2'] = pf['type_code'].map(lambda k: models[k].train_r2)
pf['scale'] = pf['type_code'].map(lambda k: models[k].scale)

pf.to_parquet('data/biomass.parquet', compression='gzip', engine='fastparquet')
# pf = pd.read_parquet('data/biomass.parquet')

# print('[biomass] regridding predictions')
# ds = xr.Dataset()
# final_mask = load.nlcd(store='local', year=2016, classes=[41, 42, 43, 90])
# final_mask.values = final_mask.values * (final_mask.values > 0.5)
# for scenario in tqdm(scenarios):
#     results = []
#     for target in targets:
#         key = cmip_model + '_' + scenario + '_' + target
#         gridded = fit.interp(pf, final_mask, var=key)
#         results.append(gridded)
#     da = xr.concat(results, dim=xr.Variable('year', targets))
#     ds[scenario] = da

# ds.to_zarr('data/biomass.zarr')
