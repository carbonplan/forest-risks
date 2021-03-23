import warnings

import numpy as np
import xarray as xr
from astropy.convolution import Gaussian2DKernel, convolve


def smooth(da, gaussian_kernel_size=None):
    """
    Smooth in space a data array according
    to box with height and width of `spatial_smoothing_window`
    """
    if gaussian_kernel_size is not None:
        # define kernel size
        kernel = Gaussian2DKernel(x_stddev=gaussian_kernel_size)
        # blur your maps according to that kernel
        blur = convolve(da.values, kernel, boundary='extend')
        return xr.DataArray(blur, coords=da.coords)
    else:
        return da


def annualize(
    ds,
    variable,
    signal,
    climate_prepend=None,
    rolling_period=None,
    gaussian_kernel_size=None,
    analysis_tlim=slice('1984', '2018'),
):
    """
    function to aggregate your full spatial
    ds into one at annual signal. it will aggregate across space
    and time and that operation might matter, so take note!<3
    """
    if climate_prepend is not None:
        da = xr.combine_by_coords([climate_prepend, ds])[variable]
    else:
        da = ds[variable]

    if gaussian_kernel_size is not None:
        # smoooth
        for t in range(len(da['time'])):
            da[t] = smooth(da[t], gaussian_kernel_size)
    else:
        # mean
        if signal == 'global':
            da = da.mean(dim=['x', 'y'])

    if climate_prepend is not None:
        aggregated = da.rolling(
            dim={'time': rolling_period}, min_periods=rolling_period, center=False
        )
    else:
        aggregated = da.groupby('time.year')

    if variable == 'tmean':
        aggregated = aggregated.max()
    elif variable == 'ppt':
        aggregated = aggregated.max()
    elif variable == 'cwd':
        aggregated = aggregated.max()
    else:
        print('{} not implemented'.format(variable))

    # drop your first year if you were rolling
    if climate_prepend is not None:
        aggregated = aggregated.sel(time=analysis_tlim)
    return aggregated


def package_annualized(da, shape, signal='global', climate_prepend=None):
    """
    to get them into the right shapes for the model we need to do some ugly
    array reshaping
    """
    if signal == 'global':
        if climate_prepend is not None:
            tile_shape = [shape[1], shape[2]]
        else:
            tile_shape = [12, shape[1], shape[2]]
        arr = np.asarray([np.tile(a, tile_shape) for a in da]).flatten()

    if signal == 'local':
        if climate_prepend is not None:
            arr = da.values.flatten()
        else:
            tile_shape = [12, 1, 1]
            arr = np.asarray([np.tile(a, tile_shape) for a in da]).flatten()

    return arr


def fire(
    climate,
    nftd,
    mtbs=None,
    eval_only=False,
    scramble=False,
    add_global_climate_trends=False,
    add_local_climate_trends=False,
    climate_prepend=None,
    gaussian_kernel_size=None,
    analysis_tlim=('1970', '2099'),
):
    """
    Prepare x and y and group variables for fire model fitting
    given an xarray dataset
    """
    shape = (len(climate.time), len(climate.y), len(climate.x))
    x = np.asarray([climate[var].values.flatten() for var in climate.data_vars]).T
    f = np.asarray([np.tile(a, [shape[0], 1, 1]).flatten() for a in nftd.values]).T

    # after you've done your climate packaging you can tack on the earlier year for aggregations

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        if add_global_climate_trends is not None:
            arr_list = []
            for var, attrs in add_global_climate_trends.items():
                arr_list.append(
                    package_annualized(
                        annualize(
                            climate,
                            var,
                            'global',
                            climate_prepend=attrs['climate_prepend'],
                            rolling_period=attrs['rolling_period'],
                            analysis_tlim=analysis_tlim,
                        ),
                        shape,
                        'global',
                        climate_prepend=attrs['climate_prepend'],
                    )
                )
            f2 = np.asarray(arr_list).T
        if add_local_climate_trends is not None:
            arr_list = []
            for var, attrs in add_local_climate_trends.items():
                arr_list.append(
                    package_annualized(
                        annualize(
                            climate,
                            var,
                            'local',
                            climate_prepend=attrs['climate_prepend'],
                            rolling_period=attrs['rolling_period'],
                            gaussian_kernel_size=attrs['gaussian_kernel_size'],
                            analysis_tlim=analysis_tlim,
                        ),
                        shape,
                        'local',
                        climate_prepend=attrs['climate_prepend'],
                    )
                )

            f3 = np.asarray(arr_list).T
    x = np.concatenate([x, f], axis=1)
    if add_global_climate_trends:
        x = np.concatenate([x, f2], axis=1)
    if add_local_climate_trends:
        x = np.concatenate([x, f3], axis=1)

    if eval_only:
        return x

    else:
        y = mtbs['monthly'].values.flatten()
        return x, y


def drought(df, eval_only=False, duration=10):
    """
    Prepare x and y values for drought model fitting
    given a data frame
    """
    df = df.copy()

    if eval_only:
        fit_vars = ['ppt_sum_min', 'tavg_mean_max', 'age', 'age_squared', 'duration']
        df['age_squared'] = df['age'] ** 2
        df['duration'] = duration
        x = df[fit_vars]
        x = x.values
        meta = df[['lat', 'lon', 'type_code']].reset_index(drop=True)

        return x, meta

    else:
        fit_vars = ['ppt_sum_min_1', 'tavg_mean_max_1', 'age', 'age_squared', 'duration']
        # 'pdsi_mean_min_1','cwd_sum_max_1',
        # 'pet_mean_max_1', 'vpd_mean_max_1',
        inds = (
            (df['condprop'] > 0.3)
            & (not (df['disturb_human_1'] is True))
            & (not (df['disturb_fire_1'] is True))
            & (not (df['treatment_cutting_1'] is True))
        )
        df = df[inds].copy()
        df['age_squared'] = df['age'] ** 2
        df['duration'] = df['year_1'] - df['year_0']
        y = df['mort_1'] / df['balive_0']
        x = df[fit_vars]

        inds = (np.isnan(x).sum(axis=1) == 0) & (~np.isnan(y)) & (y < 1)

        meta = df[inds][['lat', 'lon', 'type_code']].reset_index(drop=True)

        x = x[inds].values
        y = y[inds].values

        return x, y, meta


def insects(df, eval_only=False, duration=10):
    """
    Prepare x and y values for insect model fitting
    given a data frame
    """
    df = df.copy()

    if eval_only:
        fit_vars = ['ppt_sum_min', 'tavg_mean_max', 'age', 'age_squared', 'duration']
        df['age_squared'] = df['age'] ** 2
        df['duration'] = duration
        x = df[fit_vars]
        x = x.values
        meta = df[['lat', 'lon', 'type_code']].reset_index(drop=True)

        return x, meta

    else:

        fit_vars = [
            'ppt_sum_min_1',
            'tavg_mean_max_1',
            'age',
            'age_squared',
            'duration',
        ]

        inds = (
            (df['condprop'] > 0.3)
            & (not (df['disturb_human_1'] is True))
            & (not (df['disturb_fire_1'] is True))
            & (not (df['treatment_cutting_1'] is True))
        )
        df = df[inds].copy()
        df['age_squared'] = df['age'] ** 2
        df['duration'] = df['year_1'] - df['year_0']
        y = df['fraction_insect_1'] * (df['mort_1'] / df['balive_0'])
        x = df[fit_vars]

        inds = (np.isnan(x).sum(axis=1) == 0) & (~np.isnan(y)) & (y < 1)

        meta = df[inds][['lat', 'lon', 'type_code']].reset_index(drop=True)

        x = x[inds].values
        y = y[inds].values

        return x, y, meta
