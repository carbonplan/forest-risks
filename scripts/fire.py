import warnings

import numpy as np
import xarray as xr
from tqdm import tqdm

from carbonplan_forests import fit, load

warnings.simplefilter('ignore', category=RuntimeWarning)

coarsen = 16
tlim = (1984, 2018)
data_vars = ['ppt', 'tavg']
fit_vars = ['tavg', 'ppt']

print('[fire] loading data')
mask = load.nlcd(store='local', classes='all', year=2001)
groups = load.nftd(store='local', groups='all', coarsen=coarsen, mask=mask, area_threshold=1500)
climate = load.terraclim(store='local', tlim=tlim, coarsen=coarsen, data_vars=data_vars, mask=mask)
mtbs = load.mtbs(store='local', coarsen=coarsen, tlim=tlim)
mtbs['vlf'] = mtbs['vlf'] > 0

print('[fire] fitting model')
model = fit.fire(x=climate[fit_vars], y=mtbs['vlf'], f=groups)

ds = xr.Dataset()

print('[fire] evaluating on training data')
groups = load.nftd(store='local', groups='all', mask=mask, area_threshold=1500)
climate = load.terraclim(store='local', tlim=tlim, data_vars=data_vars, mask=mask)
prediction = model.predict(x=climate[fit_vars], f=groups)
ds['fitted_fire'] = prediction.mean('time')

print('[fire] evaluating on future climate')
targets = list(map(lambda x: str(x), np.arange(2020, 2120, 20)))
cmip_model = 'BCC-CSM2-MR'
scenarios = ['ssp245', 'ssp370', 'ssp585']
for scenario in tqdm(scenarios):
    results = []
    for target in targets:
        tlim = (int(target) - 10, int(target) + 9)
        climate = load.cmip(
            store='local', model=cmip_model, scenario=scenario, tlim=tlim, data_vars=data_vars
        )
        prediction = model.precict(x=climate[fit_vars], f=groups)
        results.append(xr.DataArray(prediction.mean('time')))
    da = xr.concat(results, dim=xr.Variable('year', targets))
    ds[cmip_model + '_' + scenario] = da

ds.to_zarr('data/fire.zarr')
