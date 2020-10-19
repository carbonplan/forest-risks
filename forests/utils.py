import numpy as np
from pyproj import Proj, transform
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.transform import xy


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
