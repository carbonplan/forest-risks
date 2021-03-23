import os
import sys
import warnings

import xarray as xr
from tqdm import tqdm

from carbonplan_forest_risks import collect, fit, load, prepare, utils
from carbonplan_forest_risks.utils import get_store

warnings.simplefilter('ignore', category=RuntimeWarning)

args = sys.argv

if len(args) < 2:
    store = 'local'
else:
    store = args[1]
    run_name = args[2]
    data_vars = args[3].strip('[]').split(',')
    coarsen_fit = int(args[4])

coarsen_predict = coarsen_fit
tlim = (1983, 2018)
analysis_tlim = slice('1984', '2018')

print('[fire] loading data')
mask = (load.nlcd(store=store, year=2001).sel(band=[41, 42, 43, 90]).sum('band') > 0.25).astype(
    'float'
)
nftd = load.nftd(store=store, groups='all', coarsen=coarsen_fit, mask=mask, area_threshold=1500)

climate = load.terraclim(
    store=store,
    tlim=tlim,
    coarsen=coarsen_fit,
    variables=data_vars,
    mask=mask,
    sampling="monthly",
)
mtbs = load.mtbs(store=store, coarsen=coarsen_fit, tlim=tlim, mask=mask)
mtbs = mtbs.assign_coords({'x': nftd.x, 'y': nftd.y})

print('[fire] fitting model')
prepend = climate.sel(time=slice('1983', '1983'))
x, y = prepare.fire(
    climate.sel(time=slice('1984', '2018')),
    nftd,
    mtbs,
    add_global_climate_trends={
        'tmean': {'climate_prepend': prepend, 'rolling_period': 12},
        'ppt': {'climate_prepend': prepend, 'rolling_period': 12},
    },
    add_local_climate_trends=None,
    analysis_tlim=slice('1984', '2018'),
)
x_z, x_mean, x_std = utils.zscore_2d(x)
model = fit.hurdle(x_z, y, log=False)
yhat = model.predict(x_z)
prediction = collect.fire(yhat, mtbs)

# what does this do?
print('[fire] setting up evaluation')

ds = xr.Dataset()

print('[fire] evaluating on training data')
climate = load.terraclim(
    store=store,
    tlim=tlim,
    coarsen=coarsen_predict,
    variables=data_vars,
    mask=mask,
    sampling='monthly',
)
prepend = climate.sel(time=slice('1983', '1983'))
x, y = prepare.fire(
    climate.sel(time=slice('1984', '2018')),
    nftd,
    mtbs,
    add_global_climate_trends={
        'tmean': {'climate_prepend': prepend, 'rolling_period': 12},
        'ppt': {'climate_prepend': prepend, 'rolling_period': 12},
    },
    add_local_climate_trends=None,
    analysis_tlim=slice('1984', '2018'),
)
x_z, x_mean, x_std = utils.zscore_2d(x)

x_z = utils.zscore_2d(x, mean=x_mean, std=x_std)
yhat = model.predict(x_z)
prediction = collect.fire(yhat, climate.sel(time=slice('1984', '2018')))
ds['historical'] = (['time', 'y', 'x'], prediction['prediction'])
ds = ds.assign_coords(
    {
        'x': climate.x,
        'y': climate.y,
        'time': climate.sel(time=slice('1984', '2018')).time,
        'lat': climate.lat,
        'lon': climate.lon,
    }
)
account_key = os.environ.get('BLOB_ACCOUNT_KEY')
if store == 'local':
    ds.to_zarr('data/fire_historical.zarr', mode='w')
elif store == 'az':
    path = get_store(
        'carbonplan-scratch',
        'data/fire_historical_{}.zarr'.format(run_name),
        account_key=account_key,
    )
    ds.to_zarr(path, mode='w')
print('[fire] evaluating on future climate')
cmip_models = [
    ('CanESM5-CanOE', 'r3i1p2f1'),
    ('MIROC-ES2L', 'r1i1p1f2'),
    ('ACCESS-CM2', 'r1i1p1f1'),
    ('ACCESS-ESM1-5', 'r10i1p1f1'),
    ('MRI-ESM2-0', 'r1i1p1f1'),
    ('MPI-ESM1-2-LR', 'r10i1p1f1'),
]
scenarios = ['ssp245', 'ssp370', 'ssp585']
ds_future = xr.Dataset()
for (cmip_model, member) in cmip_models:
    for scenario in tqdm(scenarios):
        try:
            results = []
            climate = load.cmip(
                store=store,
                model=cmip_model,
                coarsen=coarsen_predict,
                downscaling='quantile-mapping',
                scenario=scenario,
                tlim=('1969', '2099'),
                variables=data_vars,
                sampling='monthly',
                member=member,
                historical=True,
                mask=mask,
            )
            prepend = climate.sel(time=slice('1969', '1969'))
            x = prepare.fire(
                climate.sel(time=slice('1970', '2099')),
                nftd,
                mtbs,
                add_global_climate_trends={
                    'tmean': {'climate_prepend': prepend, 'rolling_period': 12},
                    'ppt': {'climate_prepend': prepend, 'rolling_period': 12},
                },
                add_local_climate_trends=None,
                eval_only=True,
                analysis_tlim=slice('1970', '2099'),
            )
            x_z = utils.zscore_2d(x, mean=x_mean, std=x_std)
            y_hat = model.predict(x_z)
            prediction = collect.fire(y_hat, climate.sel(time=slice('1970', '2099')))
            ds_future[cmip_model + '_' + scenario] = prediction['prediction']
            print('[fire] completed future run for {}-{}'.format(cmip_model, scenario))

        except:
            print(
                "some things just don't work out in life, and {}-{} is one such example".format(
                    cmip_model, scenario
                )
            )
# assigning the coords as below makes it easier for follow-on analysis when binning by forest group type
ds_future = ds_future.assign_coords(
    {
        'x': climate.x,
        'y': climate.y,
        'time': climate.sel(time=slice('1970', '2099')).time,
        'lat': climate.lat,
        'lon': climate.lon,
    }
)
if store == 'local':
    ds_future.to_zarr('data/fire_future.zarr', mode='w')
elif store == 'az':
    path = get_store(
        'carbonplan-scratch', 'data/fire_future_{}.zarr'.format(run_name), account_key=account_key
    )
    ds_future.to_zarr(path, mode='w')
