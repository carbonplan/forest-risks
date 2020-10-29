import warnings

import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression, TweedieRegressor
from sklearn.metrics import roc_auc_score

import statsmodels.api as sm

def remove_nans(x, y=None):
    if y is None:
        inds = (np.isnan(x).sum(axis=1) == 0)
        return x[inds]
    else:
        inds = (np.isnan(x).sum(axis=1) == 0) & (~np.isnan(y)) & (~np.isinf(y))
        return x[inds], y[inds]


def hurdle(x, y, log=True):
    x, y = remove_nans(x, y)
    n_obs = len(x)

    # x = sm.add_constant(x, prepend=True)
    # clf = sm.GLM(y > 0, x, family=sm.families.Binomial()).fit(maxiter=500, method='IRLS')
    # reg = sm.GLM(y[y > 0], x[y > 0], family=sm.families.Gaussian(link=sm.families.links.log())).fit(maxiter=500,  method='IRLS')

    clf = LogisticRegression(fit_intercept=True, penalty='none', max_iter=1000)
    reg = TweedieRegressor(power=0, link='log', alpha=0, tol=1e-8, max_iter=1000)

    clf.fit(x, y > 0)
    reg.fit(x[y > 0], y[y > 0])

    return Model(clf, reg, n_obs, x=x, y=y)


class Model:
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
        yhat = self.predict_expected_value(x)
        return 1 - np.nanvar(y - yhat, ddof=1) / np.nanvar(y, ddof=1)

    def predict(self, x):
        x = remove_nans(x)
        #x = sm.add_constant(x, prepend=True)
        return self.clf.predict(x) * self.reg.predict(x)

    def predict_expected_value(self, x):
        x = remove_nans(x)
        #x = sm.add_constant(x, prepend=True)
        return self.clf.predict_proba(x)[:,1] * self.reg.predict(x)
