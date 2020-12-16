import warnings
import numpy as np

def fire(mtbs, climate, nftd, eval_only=False):
    """
    Prepare x and y and group variables for fire model fitting
    given an xarray dataset
    """
    shape = (len(mtbs.time), len(mtbs.y), len(mtbs.x))
    x = np.asarray([climate[var].values.flatten() for var in climate.data_vars]).T
    y = mtbs['monthly'].values.flatten()
    # groups = np.argmax(nftd.values, axis=0).astype('float')
    # groups[nftd.values.sum(axis=0) < 0.01] = 0
    # groups = np.tile(groups.flatten(), [shape[0], 1, 1]).flatten()
    f = np.asarray([np.tile(a, [shape[0], 1, 1]).flatten() for a in nftd.values]).T

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        f2 = np.asarray(
            [
                np.asarray(
                    [
                        np.tile(a.mean(), [12, shape[1], shape[2]])
                        for a in climate['tavg'].groupby('time.year').max()
                    ]
                ).flatten(),
                np.asarray(
                    [
                        np.tile(a.mean(), [12, shape[1], shape[2]])
                        for a in climate['ppt'].groupby('time.year').max()
                    ]
                ).flatten(),
            ]
        ).T

    x = np.concatenate([x, f, f2], axis=1)

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
        y = df['fraction_insect_1'] * (df['mort_1'] / df['balive_0'])
        x = df[fit_vars]

        inds = (np.isnan(x).sum(axis=1) == 0) & (~np.isnan(y)) & (y < 1)

        meta = df[inds][['lat', 'lon', 'type_code']].reset_index(drop=True)

        x = x[inds].values
        y = y[inds].values

        return x, y, meta
