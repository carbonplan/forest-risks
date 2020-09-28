import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from rasterio import Affine
from rasterio.transform import rowcol
from pyproj import transform, Proj
from carbonplan_data.utils import albers_conus_crs, albers_conus_transform
from .. import setup

def biomass(store='gcs', states='all', return_type='dataframe', clean=True):
    path = setup.loading(store)

    if states == 'all':
        states = ['AL','AZ','AR','CA','CO','CT','DE','FL','GA','IA','ID','IL', 
            'IN','KS','KY','LA','ME','MA','MD','MI','MN','MO','MS','MT','NC','ND','NE','NH',
            'NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX', 
            'UT','VT','VA','WA','WV','WI','WY']

    if type(states) is str:
        return biomass_state(store, states, clean)

    if type(states) is list:
        return pd.concat([biomass_state(
            store, state, clean
        ) for state in states])

def biomass_state(store, state, clean):
    path = setup.loading(store)
    df = pd.read_parquet(path / f'processed/fia-states/long/{state.lower()}.parquet')

    if clean:
        inds = (
            (df['adj_ag_biomass'] > 0) & 
            (df['STDAGE'] < 999) & 
            (df['STDAGE'] > 0) & 
            (~np.isnan(df['FLDTYPCD'])) & 
            (df['FLDTYPCD'] != 999) &
            (df['FLDTYPCD'] != 950) & 
            (df['FLDTYPCD'] <= 983) & 
            (df['DSTRBCD1'] == 0) & 
            (df['COND_STATUS_CD'] == 1) & 
            (df['CONDPROP_UNADJ'] > 0.3) & 
            (df['INVYR'] < 9999)
        )
        df = df[inds]

    df = (
        df
        .rename(columns={
            'LAT': 'lat',
            'LON': 'lon',
            'adj_ag_biomass': 'biomass', 
            'STDAGE': 'age',
            'INVYR': 'year',
            'FLDTYPCD': 'type_code',
        })
        .filter(['lat', 'lon', 'age', 'biomass', 'year', 'type_code'])
    )

    return df

def biomass_features(store='gcs', df=None):
    path = setup.loading(store)
    mapper = fsspec.get_mapper(path / 'processed/terraclimate/conus/4000m/raster.zarr')

    t = Affine(*albers_conus_transform(4000))
    p1 = Proj(albers_conus_crs())
    p2 = Proj(proj='latlong', datum='WGS84')
    x, y = transform(p2, p1, df['lon'].values, df['lat'].values)
    rc = rowcol(t, x, y)

    ds = xr.open_zarr(mapper)

    ind_r = xr.DataArray(rc[0], dims=["x"])
    ind_c = xr.DataArray(rc[1], dims=["x"])

    def weighted_mean(ds, *args, **kwargs):
        weights = ds.time.dt.days_in_month
        return ds.weighted(weights).mean(dim='time')

    df['ppt'] = (
        ds['ppt']
        .resample(time='AS')
        .sum('time')
        .sel(time=slice('2010','2020'))
        .mean('time')
    )[ind_r, ind_c].values

    df['tmax'] = (
        ds['tmax']
        .resample(time='AS')
        .map(weighted_mean, dim='time')
        .sel(time=slice('2010','2020'))
        .mean('time')
    )[ind_r, ind_c].values

    df = df.dropna().reset_index(drop=True)

    return df