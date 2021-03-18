import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, TweedieRegressor
from sklearn.metrics import roc_auc_score

from ..utils import remove_nans


def hurdle(x, y, log=True, max_iter=1000):
    x, y = remove_nans(x, y)
    n_obs = len(x)

    clf = LogisticRegression(fit_intercept=True, penalty='none', max_iter=max_iter)

    if log:
        reg = TweedieRegressor(
            fit_intercept=True, power=0, link='log', alpha=0, tol=1e-8, max_iter=max_iter
        )
    else:
        reg = LinearRegression(fit_intercept=True)

    clf.fit(x, y > 0)
    reg.fit(x[y > 0, :], y[y > 0])

    return HurdleModel(clf, reg, n_obs, log=log, x=x, y=y)


class HurdleModel:
    def __init__(self, clf, reg, n_obs, log=None, x=None, y=None):
        self.clf = clf
        self.reg = reg
        self.log = log
        self.n_obs = n_obs
        if x is not None and y is not None:
            self.train_r2 = np.corrcoef(self.predict_linear(x)[y > 0], y[y > 0])[0, 1] ** 2
            self.train_roc = roc_auc_score(y > 0, self.predict_prob(x))

    def __repr__(self):
        return f"HurdleModel(link='{self.log}', train_r2='{self.train_r2:.3f}', train_roc='{self.train_roc:.3f}')"

    def score_roc(self, x, y):
        x, y = remove_nans(x, y)
        return roc_auc_score(y > 0, self.predict_prob(x))

    def score_r2(self, x, y):
        x, y = remove_nans(x, y)
        return np.corrcoef(self.predict_linear(x)[y > 0], y[y > 0])[0, 1] ** 2

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

    def predict_prob(self, x):
        out = np.ones(len(x)) * np.NaN
        x, inds = remove_nans(x, return_inds=True)
        prediction = self.clf.predict_proba(x)[:, 1]
        out[inds] = prediction
        return out

    def predict_linear(self, x):
        out = np.ones(len(x)) * np.NaN
        x, inds = remove_nans(x, return_inds=True)
        prediction = self.reg.predict(x)
        out[inds] = prediction
        return out
