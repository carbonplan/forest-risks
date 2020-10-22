import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def zscore_2d(x, mean=None, std=None):
    recomputing = False
    if mean is None or std is None:
        recomputing = True
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    if recomputing:
        return (x - mean) / std, mean, std, 
    else:
        return (x - mean) / std


def fire(x, y, f):
    """
    Fit a fire model

    Parameters
    ----------
    x : xarray dataset with primary predictor variables (vars, t, x, y)
    y : xarray dataarray with observed variable (t, x, y)
    f : xarray dataarray with auxillery features (f, x, y)
    """
    shape = (len(x.time), len(x.y), len(x.x))

    x = np.asarray([x[var].values.flatten() for var in x.data_vars]).T
    y = (y.values.flatten() > 0).astype('int')
    f = np.asarray([np.tile(a, [shape[0], 1, 1]).flatten() for a in f]).T
    x = np.concatenate([x, f], axis=1)
    
    inds = (~np.isnan(x.sum(axis=1))) & (~np.isnan(y))
    x_z, x_mean, x_std = zscore_2d(x[inds])

    model = LogisticRegression(fit_intercept=True, max_iter=500, solver="lbfgs")
    model.fit(x_z, y[inds])

    return Model(model, x_mean, x_std)


class Model:
    def __init__(self, model, x_mean, x_std):
        self.model = model
        self.x_mean = x_mean
        self.x_std = x_std

    def __repr__(self):
        return str(self.model)

    def predict(self, x, f):
        shape = (len(x.time), len(x.y), len(x.x))

        da = xr.Dataset()
        da['x'] = x.x
        da['y'] = x.y
        da['time'] = x.time
        da['lat'] = (['y', 'x'], x['lat'])
        da['lon'] =(['y', 'x'], x['lon'])

        x = np.asarray([x[var].values.flatten() for var in x.data_vars]).T
        f = np.asarray([np.tile(a, [shape[0], 1, 1]).flatten() for a in f]).T
        x = np.concatenate([x, f], axis=1)

        inds = (~np.isnan(x.sum(axis=1)))

        x_z = zscore_2d(x[inds], self.x_mean, self.x_std)

        y_hat = self.model.predict_proba(x_z)[:, 1]

        y_hat_full = np.zeros(shape).flatten()
        y_hat_full[inds] = y_hat
        y_hat_full = y_hat_full.reshape(shape)

        da['prob'] = (['time', 'y', 'x'], y_hat_full)
        
        return da