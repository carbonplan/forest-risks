import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import binom

from carbonplan_forest_risks import utils


def integrated_risk(p):
    return (1 - binom.cdf(0, 20, p)) * 100


args = sys.argv

if len(args) < 2:
    raise ValueError('must specify dataset')
dataset = args[1]

if len(args) > 2:
    coarsen = int(args[2])
    savename = f'{dataset}_d{coarsen}'
    res = 4000 * coarsen
else:
    coarsen = 0
    savename = dataset
    res = 4000

print(f'[{dataset}] converting to geojson')

precision = 2

store = utils.get_store('carbonplan-forests', f'risks/results/web/{dataset}.zarr')
ds = xr.open_zarr(store)

if coarsen > 0:
    ds = ds.coarsen(x=coarsen, y=coarsen, boundary='trim').mean().compute()

# if dataset == 'fire':
#     scenarios = ['ssp245', 'ssp370', 'ssp585']
#     for scenario in scenarios:
#         keys = list(
#             filter(lambda a: a is not None, [k if scenario in k else None for k in ds.data_vars])
#         )
#         ds[scenario] = functools.reduce(lambda a, b: a + b, [ds[key] for key in keys]) / len(keys)

if 'fire' in dataset or 'biomass' in dataset:
    scenarios = ['ssp245', 'ssp370', 'ssp585']
    ds = ds.sel(year=slice('2010', '2090'))
    targets = ds['year']

    a = np.concatenate([ds[scenario].values for scenario in scenarios], axis=0)

    if 'fire' in dataset:
        a = integrated_risk(a)
    if 'insects' in dataset or 'drought' in dataset:
        a = a * 20

    a[np.isnan(a)] = 0
    r, c = np.nonzero(a.max(axis=0))
    lat, lon = utils.rowcol_to_latlon(r, c, res=res)

    df = pd.DataFrame()

    for s, scenario in enumerate(scenarios):
        for y, year in enumerate(targets):
            key = str(s) + '_' + str(y)
            a = ds[scenario].sel(year=year).values

            if 'fire' in dataset:
                a = integrated_risk(a)

            if 'insects' in dataset or 'drought' in dataset:
                a = a * 20

            a[np.isnan(a)] = 0
            df[key] = np.round(a[r, c], precision)

if 'insects' in dataset or 'drought' in dataset:
    a = ds['historical'].values

    if 'insects' in dataset or 'drought' in dataset:
        a = a * 100 * 20

    a[np.isnan(a)] = 0
    r, c = np.nonzero(a)
    lat, lon = utils.rowcol_to_latlon(r, c, res=res)
    df = pd.DataFrame()
    key = '0_0'
    a[np.isnan(a)] = 0
    df = pd.DataFrame()
    df[key] = np.round(a[r, c], precision)

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lon, lat))
gdf.to_file(f'data/{savename}.geojson', driver='GeoJSON')
