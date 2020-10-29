import warnings

import numpy as np
import xarray as xr
from tqdm import tqdm

from carbonplan_forests import fit, load

warnings.simplefilter('ignore', category=RuntimeWarning)

coarsen_fit = 16
coarsen_predict = 4
tlim = (1984, 2018)
data_vars = ['ppt', 'tavg']
fit_vars = ['ppt', 'tavg']

print('[fire] loading data')
mask = load.nlcd(store='local', classes='all', year=2001)
groups = load.nftd(store='local', groups='all', coarsen=coarsen_fit, mask=mask, area_threshold=1500)
climate = load.terraclim(
    store='local', tlim=tlim, coarsen=coarsen_fit, data_vars=data_vars, mask=mask
)
mtbs = load.mtbs(store='local', coarsen=coarsen_fit, tlim=tlim)
mtbs['vlf'] = mtbs['vlf'] > 0

print('[fire] fitting model')
model = fit.fire(x=climate[fit_vars], y=mtbs['vlf'], f=groups)

print('[fire] setting up evaluation')
final_mask = load.nlcd(store='local', year=2016, coarsen=coarsen_predict, classes=[41, 42, 43, 90])
final_mask.values = final_mask.values > 0.5
ds = xr.Dataset()

print('[fire] evaluating on training data')
groups = load.nftd(
    store='local', groups='all', mask=mask, coarsen=coarsen_predict, area_threshold=1500
)
climate = load.terraclim(
    store='local', tlim=(1984, 2018), coarsen=coarsen_predict, data_vars=data_vars, mask=mask
)
prediction = model.predict(x=climate[fit_vars], f=groups)
ds['fitted_fire'] = prediction['prob'] * final_mask.values

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
