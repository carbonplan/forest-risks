import os

import pandas as pd
import xarray as xr

from carbonplan_forest_risks import fit, load, utils

store = 'az'

df = pd.read_csv(
    'https://carbonplan.blob.core.windows.net/carbonplan-scratch/from-bill-04-14-2021/Fig1D_DroughtModel_ModeledFIAlongEnsembleHistMort_FIAlong_04-14-2021.csv'
)

pf = pd.DataFrame()

pf['lat'] = df['V3']
pf['lon'] = df['V2']
pf['mortality'] = df['V6']

pf = pf.dropna().reset_index(drop=True)

ds = xr.Dataset()

nlcd = load.nlcd(store=store, year=2016, classes=[41, 42, 43, 90])
final_mask = nlcd.sum('band')
final_mask.attrs['crs'] = nlcd.attrs['crs']
final_mask.values = final_mask.values > 0.5

gridded = fit.interp(pf, final_mask, var='mortality')

ds['historical'] = gridded

account_key = os.environ.get('BLOB_ACCOUNT_KEY')
path = utils.get_store(
    'carbonplan-forests', 'risks/results/web/drought.zarr', account_key=account_key
)
ds.to_zarr(path, mode='w')
