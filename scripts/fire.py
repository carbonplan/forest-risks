import functools
import sys
import warnings

import numpy as np
import xarray as xr
from tqdm import tqdm

from carbonplan_forests import collect, fit, load, prepare, utils
from carbonplan_forests.utils import get_store

warnings.simplefilter('ignore', category=RuntimeWarning)

args = sys.argv

if len(args) < 2:
    store = 'local'
else:
    store = args[1]

coarsen_fit = 4
coarsen_predict = 4
coarsen_scale = 1 / coarsen_fit
tlim = (1984, 2018)
data_vars = ['tmean', "cwd", "pdsi", 'ppt']
fit_vars = ['tmean', "cwd", "pdsi", 'ppt']


def integrated_risk(da):
    return 1 - functools.reduce(
        lambda a, b: a * b, [1 - year for year in da.groupby('time.month').mean()]
    )


print('[fire] loading data')
mask = load.mask(store=store, year=2001)
nftd = load.nftd(store=store, groups='all', area_threshold=1500, coarsen=coarsen_fit, mask=mask)
climate = load.terraclim(
    store=store,
    tlim=tlim,
    coarsen=coarsen_fit,
    variables=data_vars,
    mask=mask,
    sampling="monthly",
)
mtbs = load.mtbs(store=store, coarsen=coarsen_fit, tlim=tlim)
mtbs = mtbs.assign_coords({'x': nftd.x, 'y': nftd.y})
# do we still need this line?
# mtbs['vlf'] = mtbs['vlf'] > 0

print('[fire] fitting model')
x, y = prepare.fire(climate, nftd, mtbs, add_local_climate_trends=True)
x_z, x_mean, x_std = utils.zscore_2d(x)
model = fit.hurdle(x_z, y, log=True)
yhat = model.predict(x_z)
prediction = collect.fire(yhat, mtbs)

# what does this do?
print('[fire] setting up evaluation')
final_mask = load.nlcd(store=store, year=2016, coarsen=coarsen_predict, classes=[41, 42, 43, 90])
final_mask.values = final_mask.values > 0.5
ds = xr.Dataset()

print('[fire] evaluating on training data')
groups = load.nftd(
    store=store, groups='all', mask=mask, coarsen=coarsen_predict, area_threshold=1500
)
climate = load.terraclim(
    store=store,
    tlim=(2005, 2014),
    coarsen=coarsen_predict,
    variables=data_vars,
    mask=mask,
    sampling='monthly',
)
prediction = model.predict(x_z)
ds['historical'] = (['time', 'x', 'y'], prediction['prediction'])
# Not doing integrated risk
# ds['historical'] = integrated_risk(prediction['prob'] * coarsen_scale) * final_mask.values
store = get_store('carbonplan-scratch', 'data/fire.zarr')
ds.to_zarr(store, mode='w')
print('[fire] evaluating on future climate')
targets = list(map(lambda x: str(x), np.arange(2020, 2120, 20)))
cmip_models = ['BCC-CSM2-MR', 'ACCESS-ESM1-5', 'CanESM5', 'MIROC6', 'MPI-ESM1-2-LR']
scenarios = ['ssp245', 'ssp370', 'ssp585']
for cmip_model in cmip_models:
    for scenario in tqdm(scenarios):
        results = []
        for target in targets:
            print('[fire] predicting for {} {} {}'.format(cmip_model, scenario, target))
            tlim = (int(target) - 5, int(target) + 4)
            climate = load.cmip(
                store=store,
                model=cmip_model,
                coarsen=coarsen_predict,
                scenario=scenario,
                tlim=tlim,
                data_vars=data_vars,
                sampling='monthly',
            )
            x, y = prepare.fire(climate, nftd, mtbs, add_local_climate_trends=True)
            x_z, x_mean, x_std = utils.zscore_2d(x)
            y_hat = model.predict(x_z)
            prediction = collect.fire(y_hat, climate)
            results.append(prediction)
            # results.append(integrated_risk(prediction['prob'] * coarsen_scale) * final_mask.values)
        da = xr.concat(results, dim=xr.Variable('year', targets))
        ds[cmip_model + '_' + scenario] = da
if store == 'local':
    ds.to_zarr('data/fire.zarr', mode='w')
elif store == 'az':
    store = get_store('carbonplan-scratch', 'data/fire.zarr')
    ds.to_zarr(store, mode='w')
