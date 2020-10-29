import numpy as np
from pyproj import Proj, transform
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.transform import rowcol, xy


def albers_conus_extent():
    return "-2493045.0 177285.0 2342655.0 3310005.0"


def albers_conus_crs():
    return (
        'PROJCS["Albers_Conical_Equal_Area",'
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
        "TOWGS84[0,0,0,-0,-0,-0,0],"
        'AUTHORITY["EPSG","6326"]],'
        'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
        'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
        'AUTHORITY["EPSG","4326"]],'
        'PROJECTION["Albers_Conic_Equal_Area"],'
        'PARAMETER["standard_parallel_1",29.5],'
        'PARAMETER["standard_parallel_2",45.5],'
        'PARAMETER["latitude_of_center",23],'
        'PARAMETER["longitude_of_center",-96],'
        'PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],'
        'UNIT["meters",1]]'
    )


def albers_conus_transform(res=4000):
    return [res, 0.0, -2493045.0, 0.0, -res, 3310005.0]


def albers_ak_extent():
    return "-2232345.0 344805.0 1494735.0 2380125.0"


def albers_ak_crs():
    return (
        'PROJCS["WGS_1984_Albers",'
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
        'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],'
        'PROJECTION["Albers_Conic_Equal_Area"],'
        'PARAMETER["standard_parallel_1",55],'
        'PARAMETER["standard_parallel_2",65],'
        'PARAMETER["latitude_of_center",50],'
        'PARAMETER["longitude_of_center",-154],'
        'PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],'
        'UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'
    )


def albers_ak_transform(res=4000):
    return [res, 0.0, -2232345.0, 0.0, -res, 2380125.0]


def rowcol_to_latlon(row, col, res=250):
    row = np.asarray(row) if type(row) is list else row
    col = np.asarray(col) if type(col) is list else col
    x, y = xy(Affine(*albers_conus_transform(res)), row, col)
    p1 = Proj(CRS.from_wkt(albers_conus_crs()))
    p2 = Proj(proj='latlong', datum='WGS84')
    lon, lat = transform(p1, p2, x, y)
    return lat, lon


def latlon_to_rowcol(lat, lon, res=250):
    lat = np.asarray(lat) if type(lat) is list else lat
    lon = np.asarray(lon) if type(lon) is list else lon
    x, y = latlon_to_xy(lat, lon)
    r, c = rowcol(albers_conus_transform(res), x, y)
    return r, c


def latlon_to_xy(lat, lon, base_crs=albers_conus_crs()):
    p1 = Proj(base_crs)
    p2 = Proj(proj='latlong', datum='WGS84')
    x, y = transform(p2, p1, np.asarray(lon), np.asarray(lat))
    return x, y


def zscore_2d(x, mean=None, std=None):
    recomputing = False
    if mean is None or std is None:
        recomputing = True
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    if recomputing:
        return (
            (x - mean) / std,
            mean,
            std,
        )
    else:
        return (x - mean) / std


def remove_nans(x, y=None, return_inds=False):
    if y is None:
        inds = np.isnan(x).sum(axis=1) == 0
        if return_inds:
            return x[inds], inds
        else:
            return x[inds]
    else:
        inds = (np.isnan(x).sum(axis=1) == 0) & (~np.isnan(y)) & (~np.isinf(y))
        if return_inds:
            return x[inds], y[inds], inds
        else:
            return x[inds], y[inds]


def weighted_mean(ds, *args, **kwargs):
    weights = ds.time.dt.days_in_month
    return ds.weighted(weights).mean(dim='time')
