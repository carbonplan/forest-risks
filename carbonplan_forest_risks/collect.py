import numpy as np
import xarray as xr


def fire(y, src, inds=None):
    da = xr.Dataset()
    da['x'] = src.x
    da['y'] = src.y
    da['time'] = src.time
    da['lat'] = (['y', 'x'], src['lat'])
    da['lon'] = (['y', 'x'], src['lon'])
    shape = (len(src.time), len(src.y), len(src.x))

    if inds is None:
        yfull = y.reshape(shape)
    else:
        yfull = np.zeros(shape).flatten()
        yfull[inds] = y
        yfull = yfull.reshape(shape)
    da['prediction'] = (['time', 'y', 'x'], yfull)

    return da
