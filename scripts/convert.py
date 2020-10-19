import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from carbonplan.forests import utils

args = sys.argv

if len(args) < 1:
    raise ValueError('must specify dataset')
dataset = args[1]

precision = {'biomass': 2, 'fire': 3}

ds = xr.open_zarr(f'data/{dataset}.zarr')

cmip_model = 'BCC-CSM2-MR'
scenarios = ['ssp245', 'ssp370', 'ssp585']
years = ds['year'].values

a = np.concatenate([ds[scenario].values for scenario in scenarios], axis=0)
a[np.isnan(a)] = 0
r, c = np.nonzero(a.max(axis=0))
lat, lon = utils.rowcol_to_latlon(r, c, res=4000)

df = pd.DataFrame()

for s, scenario in enumerate(scenarios):
    for y, year in enumerate(years):
        key = '0' + '_' + str(s) + '_' + str(y)
        a = ds[scenario].sel(year=year).values
        a[np.isnan(a)] = 0
        df[key] = np.round(a[r, c], precision[dataset])

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lon, lat))
gdf.to_file(f'data/{dataset}.geojson', driver='GeoJSON')
