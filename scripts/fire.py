import functools
import os
import sys
import warnings

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
    run_historical = True
    run_future = True


coarsen_fit = 8
coarsen_predict = 8
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
x, y = prepare.fire(climate, nftd, mtbs, add_local_climate_trends=False)
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
    tlim=tlim,
    coarsen=coarsen_predict,
    variables=data_vars,
    mask=mask,
    sampling='monthly',
)
x, y = prepare.fire(climate, nftd, mtbs, add_local_climate_trends=False)
x_z = utils.zscore_2d(x, mean=x_mean, std=x_std)
yhat = model.predict(x_z)
prediction = collect.fire(yhat, climate)
ds['historical'] = (['time', 'y', 'x'], prediction['prediction'])
ds = ds.assign_coords({'x': climate.x, 'y': climate.y, 'time': climate.time})
account_key = os.environ.get('BLOB_ACCOUNT_KEY')
if store == 'local':
    ds.to_zarr('data/fire.zarr', mode='w')
elif store == 'az':
    store = get_store('carbonplan-scratch', 'data/fire.zarr', account_key=account_key)
    ds.to_zarr(store, mode='w')

print('[fire] evaluating on future climate')
cmip_models = [
    ('CanESM5', 'r10i1p1f1'),
    ('UKESM1-0-LL', 'r10i1p1f2'),
    ('MRI-ESM2-0', 'r1i1p1f1'),
    ('MIROC-ES2L', 'r1i1p1f2'),
    ('MIROC6', 'r10i1p1f1'),
    ('FGOALS-g3', 'r1i1p1f1'),
    ('HadGEM3-GC31-LL', 'r1i1p1f3'),
]
scenarios = ['ssp245', 'ssp370', 'ssp585']
ds_future = xr.Dataset()
for (cmip_model, member) in cmip_models:
    for scenario in tqdm(scenarios):
        try:
            store = 'az'
            results = []
            climate = load.cmip(
                store=store,
                model=cmip_model,
                coarsen=coarsen_predict,
                scenario=scenario,
                tlim=('2015', '2099'),
                variables=data_vars,
                sampling='monthly',
                member=member,
            )
            x, y = prepare.fire(climate, nftd, mtbs, add_local_climate_trends=False)
            x_z = utils.zscore_2d(x, mean=x_mean, std=x_std)
            y_hat = model.predict(x_z)
            prediction = collect.fire(y_hat, climate)
            ds_future[cmip_model + '_' + scenario] = prediction['prediction']

        except:
            print(
                "some things just don't work out in life, and {}-{} is one such example".format(
                    cmip_model, scenario
                )
            )
# assigning the coords as below makes it easier for follow-on analysis when binning by forest group type
ds_future = ds_future.assign_coords({'x': nftd.x, 'y': nftd.y})
if store == 'local':
    ds_future.to_zarr('data/fire_future.zarr', mode='w')
elif store == 'az':
    store = get_store('carbonplan-scratch', 'data/fire_future.zarr', account_key=account_key)
    ds_future.to_zarr(store, mode='w')
