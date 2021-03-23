import warnings

import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..utils import zscore_2d


def fire(x, y, f, crossval=False):
    """
    Fit a fire model

    Parameters
    ----------
    x : xarray dataset with primary predictor variables (vars, t, x, y)
    y : xarray dataarray with observed variable (t, x, y)
    f : xarray dataarray with auxillery features (f, x, y)
    """
    shape = (len(x.time), len(x.y), len(x.x))

    y = (y.values.flatten() > 0).astype('int')
    f = np.asarray([np.tile(a, [shape[0], 1, 1]).flatten() for a in f]).T

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        f2 = np.asarray(
            [
                np.asarray(
                    [
                        np.tile(a.mean(), [12, shape[1], shape[2]])
                        for a in x['tavg'].groupby('time.year').max()
                    ]
                ).flatten(),
                np.asarray(
                    [
                        np.tile(a.mean(), [12, shape[1], shape[2]])
                        for a in x['ppt'].groupby('time.year').max()
                    ]
                ).flatten(),
            ]
        ).T

    x = np.asarray([x[var].values.flatten() for var in x.data_vars]).T
    x = np.concatenate([x, f, f2], axis=1)

    inds = (~np.isnan(x.sum(axis=1))) & (~np.isnan(y))
    x = x[inds]
    y = y[inds]

    if crossval:
        train_inds = np.random.rand(len(y)) > 0.75
        test_inds = ~train_inds
    else:
        train_inds = np.ones(len(y)).astype('bool')

    x_z, x_mean, x_std = zscore_2d(x)

    model = LogisticRegression(fit_intercept=True, max_iter=500, solver="lbfgs")
    model.fit(x_z[train_inds], y[train_inds])

    train_prob = model.predict_proba(x_z[train_inds])[:, 1]
    train_roc = roc_auc_score(y[train_inds], train_prob)

    if crossval:
        test_prob = model.predict_proba(x_z[test_inds])[:, 1]
        test_roc = roc_auc_score(y[test_inds], test_prob)
        return FireModel(model, x_mean, x_std, train_roc, test_roc)
    else:
        return FireModel(model, x_mean, x_std, train_roc)


class FireModel:
    def __init__(self, model, x_mean, x_std, train_roc, test_roc=None):
        self.model = model
        self.x_mean = x_mean
        self.x_std = x_std
        self.train_roc = train_roc
        if test_roc:
            self.test_roc = test_roc

    def __repr__(self):
        return str(self.model)

    def predict(self, x, f):
        shape = (len(x.time), len(x.y), len(x.x))

        da = xr.Dataset()
        da['x'] = x.x
        da['y'] = x.y
        da['time'] = x.time
        da['lat'] = (['y', 'x'], x['lat'])
        da['lon'] = (['y', 'x'], x['lon'])

        f = np.asarray([np.tile(a, [shape[0], 1, 1]).flatten() for a in f]).T

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            f2 = np.asarray(
                [
                    np.asarray(
                        [
                            np.tile(a.mean(), [12, shape[1], shape[2]])
                            for a in x['tavg'].groupby('time.year').max()
                        ]
                    ).flatten(),
                    np.asarray(
                        [
                            np.tile(a.mean(), [12, shape[1], shape[2]])
                            for a in x['ppt'].groupby('time.year').max()
                        ]
                    ).flatten(),
                ]
            ).T

        x = np.asarray([x[var].values.flatten() for var in x.data_vars]).T
        x = np.concatenate([x, f, f2], axis=1)

        inds = ~np.isnan(x.sum(axis=1))

        x_z = zscore_2d(x[inds], self.x_mean, self.x_std)

        y_hat = self.model.predict_proba(x_z)[:, 1]

        y_hat_full = np.zeros(shape).flatten()
        y_hat_full[inds] = y_hat
        y_hat_full = y_hat_full.reshape(shape)

        da['prob'] = (['time', 'y', 'x'], y_hat_full)

        return da
