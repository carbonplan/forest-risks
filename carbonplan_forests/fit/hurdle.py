import numpy as np
from sklearn.linear_model import LogisticRegression, TweedieRegressor

from ..utils import remove_nans


def hurdle(x, y, log=True):
    x, y = remove_nans(x, y)
    n_obs = len(x)

    clf = LogisticRegression(fit_intercept=True, penalty='none', max_iter=1000)
    reg = TweedieRegressor(
        fit_intercept=True, power=0, link='log', alpha=0, tol=1e-8, max_iter=1000
    )

    clf.fit(x, y > 0)
    reg.fit(x[y > 0], y[y > 0])

    return HurdleModel(clf, reg, n_obs, x=x, y=y)


class HurdleModel:
    def __init__(self, clf, reg, n_obs, x=None, y=None):
        self.clf = clf
        self.reg = reg
        self.n_obs = n_obs
        if x is not None and y is not None:
            self.train_r2 = self.r2(x, y)

    def __repr__(self):
        return str(self.clf) + str(self.reg)

    def r2(self, x, y):
        x, y = remove_nans(x, y)
        yhat = self.predict(x)
        return 1 - np.nanvar(y - yhat, ddof=1) / np.nanvar(y, ddof=1)

    def predict_binary(self, x):
        out = np.ones(len(x)) * np.NaN
        x, inds = remove_nans(x, return_inds=True)
        prediction = self.clf.predict(x) * self.reg.predict(x)
        out[inds] = prediction
        return out

    def predict(self, x):
        out = np.ones(len(x)) * np.NaN
        x, inds = remove_nans(x, return_inds=True)
        prediction = self.clf.predict_proba(x)[:, 1] * self.reg.predict(x)
        out[inds] = prediction
        return out
