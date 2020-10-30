import functools
import sys
import warnings

import numpy as np
import xarray as xr
from tqdm import tqdm

from carbonplan_forests import fit, load

warnings.simplefilter('ignore', category=RuntimeWarning)

args = sys.argv

if len(args) < 2:
    store = 'local'
else:
    store = args[1]

coarsen_fit = 4
coarsen_predict = None
coarsen_scale = 1 / coarsen_fit
tlim = (1984, 2018)
data_vars = ['ppt', 'tavg']
fit_vars = ['ppt', 'tavg']


def integrated_risk(da):
    return 1 - functools.reduce(
        lambda a, b: a * b, [1 - year for year in da.groupby('time.month').mean()]
    )


print('[fire] loading data')
mask = load.nlcd(store=store, classes='all', year=2001)
groups = load.nftd(store=store, groups='all', coarsen=coarsen_fit, mask=mask, area_threshold=1500)
climate = load.terraclim(
    store=store, tlim=tlim, coarsen=coarsen_fit, data_vars=data_vars, mask=mask
)
mtbs = load.mtbs(store=store, coarsen=coarsen_fit, tlim=tlim)
mtbs['vlf'] = mtbs['vlf'] > 0

print('[fire] fitting model')
model = fit.fire(x=climate[fit_vars], y=mtbs['vlf'], f=groups)

print('[fire] setting up evaluation')
final_mask = load.nlcd(store=store, year=2016, coarsen=coarsen_predict, classes=[41, 42, 43, 90])
final_mask.values = final_mask.values > 0.5
ds = xr.Dataset()

print('[fire] evaluating on training data')
groups = load.nftd(
    store=store, groups='all', mask=mask, coarsen=coarsen_predict, area_threshold=1500
)
climate = load.terraclim(
    store=store, tlim=(2005, 2014), coarsen=coarsen_predict, data_vars=data_vars, mask=mask
)
prediction = model.predict(x=climate[fit_vars], f=groups)
ds['historical'] = integrated_risk(prediction['prob'] * coarsen_scale) * final_mask.values

print('[fire] evaluating on future climate')
targets = list(map(lambda x: str(x), np.arange(2020, 2120, 20)))
cmip_models = ['BCC-CSM2-MR', 'ACCESS-ESM1-5', 'CanESM5', 'MIROC6', 'MPI-ESM1-2-LR']
scenarios = ['ssp245', 'ssp370', 'ssp585']
for cmip_model in cmip_models:
    for scenario in tqdm(scenarios):
        results = []
        for target in targets:
            tlim = (int(target) - 5, int(target) + 4)
            climate = load.cmip(
                store=store,
                model=cmip_model,
                coarsen=coarsen_predict,
                scenario=scenario,
                tlim=tlim,
                data_vars=data_vars,
            )
            prediction = model.predict(x=climate[fit_vars], f=groups)
            results.append(integrated_risk(prediction['prob'] * coarsen_scale) * final_mask.values)
        da = xr.concat(results, dim=xr.Variable('year', targets))
        ds[cmip_model + '_' + scenario] = da
        ds.to_zarr('data/fire.zarr', mode='w')
