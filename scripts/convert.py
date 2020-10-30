import functools
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from carbonplan_forests import utils

args = sys.argv

if len(args) < 2:
    raise ValueError('must specify dataset')
dataset = args[1]

precision = {'biomass': 2, 'fire': 3, 'drought': 3, 'insects': 3, 'biophysical': 3}

ds = xr.open_zarr(f'data/{dataset}.zarr')

if dataset == 'fire':
    scenarios = ['ssp245', 'ssp370', 'ssp585']
    for scenario in scenarios:
        keys = list(
            filter(lambda a: a is not None, [k if scenario in k else None for k in ds.data_vars])
        )
        ds[scenario] = functools.reduce(lambda a, b: a + b, [ds[key] for key in keys]) / len(keys)

if dataset in ['fire', 'biomass', 'drought', 'insects']:
    scenarios = ['ssp245', 'ssp370', 'ssp585']
    targets = ds['year'].values

    a = np.concatenate([ds[scenario].values for scenario in scenarios], axis=0)
    a[np.isnan(a)] = 0
    r, c = np.nonzero(a.max(axis=0))
    lat, lon = utils.rowcol_to_latlon(r, c, res=4000)

    df = pd.DataFrame()

    for s, scenario in enumerate(scenarios):
        for y, year in enumerate(targets):
            key = str(s) + '_' + str(y)
            a = ds[scenario].sel(year=year).values
            a[np.isnan(a)] = 0
            df[key] = np.round(a[r, c], precision[dataset])

if dataset == 'biophysical':
    a = ds['biophysical'].values
    a = -a
    a[np.isnan(a)] = 0
    a[a < 0.25 * a.max()] = 0
    a = a.astype('float64')
    r, c = np.nonzero(a)
    lat, lon = utils.rowcol_to_latlon(r, c, res=4000)

    df = pd.DataFrame()
    df['0'] = np.round(a[r, c], precision[dataset])

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lon, lat))
gdf.to_file(f'data/{dataset}.geojson', driver='GeoJSON')
