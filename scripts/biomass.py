import pickle

import numpy as np
import pandas as pd
import xarray as xr
from carbonplan_forests import load, fit
from tqdm import tqdm

print('[biomass] loading data')
df = load.fia(store='local', states='conus')
df = load.terraclim(
    store='local', tlim=(2000, 2020), data_vars=['tavg', 'ppt'], data_aggs=['mean', 'sum'], df=df
)
type_codes = df['type_code'].unique()

print('[biomass] fitting models')
models = {}
for code in tqdm(type_codes):
    df_type = df[df['type_code'] == code].reset_index()
    if len(df_type) > 30:
        x = df_type['age']
        y = df_type['biomass']
        f = [df_type['tavg_mean_mean'], df_type['ppt_sum_mean']]
        model = fit.biomass(x=x, y=y, f=f, noise='gamma')
        models[code] = model

with open('data/biomass.pkl', 'wb') as f:
    pickle.dump(models, f)

# with open('data/biomass.pkl', 'rb') as f:
#     models = pickle.load(f)

print('[biomass] preparing for evaluations')
pf = pd.DataFrame()
pf['lat'] = df['lat']
pf['lon'] = df['lon']
pf['type_code'] = df['type_code']
print(len(pf))

print('[biomass] evaluating predictions on training data')
for code in type_codes:
    if code in models.keys():
        model = models[code]
        inds = df['type_code'] == code
        x = df[inds]['age']
        f = [df[inds]['tavg_mean_mean'], df[inds]['ppt_sum_mean']]
        pf.loc[inds, 'fitted_biomass'] = model.predict(x, f)

print('[biomass] evaluating predictions on future climate models')
targets = list(map(lambda x: str(x), np.arange(2020, 2120, 20)))
cmip_model = 'BCC-CSM2-MR'
scenarios = ['ssp245', 'ssp370', 'ssp585']
for it in tqdm(range(len(targets))):
    target = targets[it]
    tlim = (str(int(target) - 10), str(int(target) + 9))
    for scenario in scenarios:
        key = cmip_model + '_' + scenario + '_' + target
        df = load.cmip(
            store='local',
            tlim=(int(tlim[0]), int(tlim[1])),
            data_vars=['tavg', 'ppt'],
            data_aggs=['mean', 'sum'],
            model=cmip_model,
            scenario=scenario,
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
                    pf.loc[inds, key] = model.predict(np.maximum(x + (float(target) - year), 0), f)
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

# pf = pd.read_parquet('data/biomass.parquet')

print('[biomass] regridding predictions')
ds = xr.Dataset()
pf = pf.dropna().reset_index(drop=True)
final_mask = load.nlcd(store='local', year=2016, classes=[41, 42, 43, 90])
final_mask.values = final_mask.values * (final_mask.values > 0.5)
ds['fitted_biomass'] = fit.interp(pf, final_mask, var='fitted_biomass')

for scenario in tqdm(scenarios):
    results = []
    for target in targets:
        key = cmip_model + '_' + scenario + '_' + target
        gridded = fit.interp(pf, final_mask, var=key)
        results.append(gridded)
    da = xr.concat(results, dim=xr.Variable('year', targets))
    ds[cmip_model + '_' + scenario] = da

ds.to_zarr('data/biomass.zarr')
