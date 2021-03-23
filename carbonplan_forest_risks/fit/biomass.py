import warnings

import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.stats import gamma, norm


def logistic(x, f, p):
    a, b, c, w0, w1 = p
    return (
        (a + w0 * f[0] + w1 * f[1])
        * (1 / (1 + c * np.exp(-b * (x))) - (1 / (1 + c)))
        * ((c + 1) / c)
    )


def biomass(x, y, f, noise='gamma', init=None):
    def loglik(x, y, f, p):
        a, b, c, w0, w1, scale = p
        _mu = logistic(x, f, [a, b, c, w0, w1])
        if noise == 'gamma':
            return -np.sum(gamma.logpdf(y, np.maximum(_mu / scale, 1e-12), scale=scale))
        if noise == 'normal':
            return -np.sum(norm.logpdf(y, loc=_mu, scale=scale))

    fx = lambda p: loglik(x, y, f, p)

    lb = np.ones((6,)) * 0.00001
    lb[2] = 1
    lb[3] = -np.inf
    lb[4] = -np.inf
    ub = np.ones((6,)) * np.inf
    bounds = Bounds(lb, ub)
    if init is None:
        init = [np.nanmean(y), 0.1, 10, 0, 0, np.nanstd(y)]
    options_trust = {'maxiter': 5000}

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        result = minimize(fx, init, bounds=bounds, method='trust-constr', options=options_trust)

    if result.success is False:
        print('optimization failed')

    return BiomassModel(result, noise, x, y, f)


class BiomassModel:
    def __init__(self, result, noise, x=None, y=None, f=None):
        self.result = result
        self.noise = noise
        self.p = result.x[0 : len(result.x) - 1]
        self.scale = result.x[len(result.x) - 1]
        if x is not None and y is not None and f is not None:
            self.train_r2 = self.r2(x, f, y)

    def __repr__(self):
        return str(self.result)

    def r2(self, x, f, y):
        yhat = self.predict(x, f)
        return 1 - np.nanvar(y - yhat, ddof=1) / np.nanvar(y, ddof=1)

    def predict(self, x, f, percentile=None):
        if percentile is not None:
            f = [np.nanpercentile(f[0], percentile[0]), np.nanpercentile(f[1], percentile[1])]
        return logistic(x, f, self.p)

    def sample(self, x, f):
        mu = logistic(x, f, self.p)
        if self.noise == 'gamma':
            return np.random.gamma(mu / self.scale, scale=self.scale)
        if self.noise == 'normal':
            return np.random.normal(mu, scale=self.scale)
