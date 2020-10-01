# fit biomass models and compute projections
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import pickle
from forests import load, fit

print('[biomass] loading data')
df = load.fia(store='local', states='CA')
df = load.terraclim(store='local', tlim=(2000,2020), vars=['tmax', 'ppt'], mean=True, df=df)

type_codes = df['type_code'].unique()

print('[biomass] fitting models')
models = {}
for code in tqdm(type_codes):
    df_type = df[df['type_code'] == code].reset_index()
    if len(df_type) > 30:
        x = df_type['age']
        y = df_type['biomass']
        f = [df_type['tmax'], df_type['ppt']]
        model = fit.biomass(x=x, y=y, f=f, noise='gamma')
        models[code] = model

pickle.dump(models, open('models.pkl', 'wb'))

pf = pd.DataFrame()
pf['lat'] = df['lat']
pf['lon'] = df['lon']
pf['type_code'] = df['type_code']

targets = np.arange(2000,2110,10)

print('[biomass] evaluating predictions')
for it in tqdm(range(len(targets))):
    target = targets[it]
    prev_target = targets[it-1]
    pf[target] = np.NaN
    for code in type_codes:
        if code in models.keys():
            model = models[code]
            inds = df['type_code'] == code
            x = df[inds]['age']
            year = df[inds]['year']
            f = [df[inds]['tmax'], df[inds]['ppt']]
            if it == 0:
                pf[target][inds] = model.predict(np.maximum(x + (target - year), 0), f)
            else:
                diff = model.predict(x + (target - year), f) - model.predict(x + (prev_target - year), f)
                pf[target][inds] = np.maximum(pf[prev_target][inds] + diff, 0)

pf = pf.dropna().reset_index(drop=True)

print('[biomass] regridding predictions')
mask = load.nlcd(store='local')
results = []
for target in tqdm(targets):
    gridded = fit.interp(pf, mask, var=target)
    results.append(gridded)

da = xr.concat(results, dim=xr.Variable('year', targets))
ds = xr.Dataset()
ds['biomass'] = da
ds.to_zarr('data/predictions.zarr')