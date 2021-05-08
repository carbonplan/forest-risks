import os
import sys
import warnings

import numpy as np
import xarray as xr
from tqdm import tqdm

from carbonplan_forest_risks import collect, fit, load, prepare, utils
from carbonplan_forest_risks.utils import get_store

warnings.simplefilter('ignore', category=RuntimeWarning)
account_key = os.environ.get('BLOB_ACCOUNT_KEY')

args = sys.argv

if len(args) < 2:
    store = 'local'
else:
    store = args[1]
    run_name = args[2]
    data_vars = args[3].strip('[]').split(',')
    coarsen_fit = int(args[4])
    coarsen_predict = int(args[5])

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
print('[fire] evaluating on training data')
# reload everything at the appropriate coarsen level (in this case no coarsening)
nftd = load.nftd(store=store, groups='all', mask=mask, coarsen=coarsen_predict, area_threshold=1500)

climate = load.terraclim(
    store=store,
    tlim=tlim,
    coarsen=coarsen_predict,
    variables=data_vars,
    mask=mask,
    sampling='monthly',
)
for year in np.arange(1984, 2024, 10):
    ds = xr.Dataset()
    print('[fire] evaluating on decade beginning in {}'.format(year))
    prepend_time_slice = slice(str(year - 1), str(year - 1))
    analysis_time_slice = slice(str(year), str(year + 9))
    prepend = climate.sel(time=prepend_time_slice)
    x, y = prepare.fire(
        climate.sel(time=analysis_time_slice),
        nftd,
        mtbs,
        add_global_climate_trends={
            'tmean': {'climate_prepend': prepend, 'rolling_period': 12},
            'ppt': {'climate_prepend': prepend, 'rolling_period': 12},
        },
        add_local_climate_trends=None,
        analysis_tlim=analysis_time_slice,
    )
    x_z = utils.zscore_2d(x, mean=x_mean, std=x_std)
    yhat = model.predict(x_z)
    prediction = collect.fire(yhat, climate.sel(time=analysis_time_slice))
    ds['historical'] = (['time', 'y', 'x'], prediction['prediction'])
    ds = ds.assign_coords(
        {
            'x': climate.x,
            'y': climate.y,
            'time': climate.sel(time=analysis_time_slice).time,
            'lat': climate.lat,
            'lon': climate.lon,
        }
    )
    if store == 'local':
        ds.to_zarr('data/fire_historical.zarr', mode='w')
    elif store == 'az':
        path = get_store(
            'carbonplan-forests',
            'risks/results/paper/fire_terraclimate_{}.zarr'.format(run_name),
            account_key=account_key,
        )
        if year == 1984:
            ds.to_zarr(path, consolidated=True, mode='w')
        else:
            ds.to_zarr(path, consolidated=True, mode='a', append_dim='time')
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
for (cmip_model, member) in cmip_models:
    for scenario in tqdm(scenarios):
        results = []
        climate = load.cmip(
            store=store,
            model=cmip_model,
            coarsen=coarsen_predict,
            method='quantile-mapping-v3',
            scenario=scenario,
            tlim=('1969', '2099'),
            variables=data_vars,
            sampling='monthly',
            member=member,
            historical=True,
            mask=mask,
        )
        try:
            for year in np.arange(1970, 2100, 10):
                ds_future = xr.Dataset()

                print(
                    '[fire] conducting prediction for {} {} {}'.format(cmip_model, scenario, year)
                )
                prepend_time_slice = slice(str(year - 1), str(year - 1))
                analysis_time_slice = slice(str(year), str(year + 9))

                prepend = climate.sel(time=prepend_time_slice)
                x = prepare.fire(
                    climate.sel(time=analysis_time_slice),
                    nftd,
                    add_global_climate_trends={
                        'tmean': {'climate_prepend': prepend, 'rolling_period': 12},
                        'ppt': {'climate_prepend': prepend, 'rolling_period': 12},
                    },
                    add_local_climate_trends=None,
                    eval_only=True,
                    analysis_tlim=analysis_time_slice,
                )
                x_z = utils.zscore_2d(x, mean=x_mean, std=x_std)
                y_hat = model.predict(x_z)
                prediction = collect.fire(y_hat, climate.sel(time=analysis_time_slice))
                ds_future[cmip_model + '_' + scenario] = (
                    ['time', 'y', 'x'],
                    prediction['prediction'],
                )
                ds_future = ds_future.assign_coords(
                    {
                        'x': climate.x,
                        'y': climate.y,
                        'time': climate.sel(time=analysis_time_slice).time,
                        'lat': climate.lat,
                        'lon': climate.lon,
                    }
                )
                path = get_store(
                    'carbonplan-scratch',
                    'data/fire_future_{}_{}_{}.zarr'.format(run_name, cmip_model, scenario),
                    account_key=account_key,
                )
                # if it's the first year then make a fresh store by overwriting; if it's later, append to existing file
                if year == 1970:
                    mode = 'w'
                    append_dim = None
                else:
                    mode = 'a'
                    append_dim = 'time'
                ds_future.to_zarr(path, mode=mode, append_dim=append_dim)
                print(
                    '[fire] completed future run for {}-{} and year {}'.format(
                        cmip_model, scenario, year
                    )
                )

        except:
            print(
                "some things just don't work out in life, and {}-{} for year {} is one such example".format(
                    cmip_model, scenario, year
                )
            )

## combine all of the runs into a single zarr file for follow-on analysis
postprocess = True
gcms = ['CanESM5-CanOE', 'MIROC-ES2L', 'ACCESS-CM2', 'ACCESS-ESM1-5', 'MRI-ESM2-0', 'MPI-ESM1-2-LR']
scenarios = ['ssp245', 'ssp370', 'ssp585']
out_path = get_store(
    'carbonplan-forests',
    'risks/results/paper/fire_cmip_{}.zarr'.format(run_name),
    account_key=account_key,
)

if postprocess:
    gcm_list = []
    for gcm in gcms:
        scenario_list = []
        for scenario in scenarios:
            path = get_store(
                'carbonplan-scratch',
                'data/fire_future_{}_{}_{}.zarr'.format(run_name, gcm, scenario),
                account_key=account_key,
            )
            scenario_list.append(
                xr.open_zarr(path).rename({'{}_{}'.format(gcm, scenario): 'probability'})
            )
            print('{} {} is done!'.format(scenario, gcm))
        ds = xr.concat(scenario_list, dim='scenario')
        ds = ds.assign_coords({'scenario': scenarios})
        gcm_list.append(ds)
    full_ds = xr.concat(gcm_list, dim='gcm')
    full_ds = full_ds.assign_coords({'gcm': gcms})
    full_ds.to_zarr(out_path, consolidated=True, mode='w')
