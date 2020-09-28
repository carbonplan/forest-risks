import numpy as np
from scipy.stats import gamma, norm
from scipy.optimize import minimize, Bounds

def logistic(x, f, p):
    a, b, c, w0, w1 = p
    return (a + w0 * f[0] + w1 * f[1]) * (1 / (1 + c * np.exp(-b * (x))) - (1 / (1 + c))) * ((c + 1) / c)

def growth(x, y, f, noise='gamma', init=None):

    def loglik(x, y, f, p):
        a, b, c, w0, w1, scale = p
        _mu = logistic(x, f, [a, b, c, w0, w1])
        if noise == 'gamma':
            return -np.nansum(gamma.logpdf(y, _mu / scale, scale=scale))
        if noise == 'normal':
            return -np.nansum(norm.logpdf(y, loc=_mu, scale=scale))

    fx = lambda p : loglik(x, y, f, p)

    lb = np.ones((6,)) * 0.00001
    lb[2] = 1
    lb[3] = -np.inf
    lb[4] = -np.inf
    ub = np.ones((6,)) * np.inf
    bounds = Bounds(lb, ub)
    if init is None:
        init = [np.nanmean(y), 0.1, 10, 0, 0, np.nanstd(y)]
    options = {'eps': 1e-8, 'maxcor': 100, 'maxiter': 500, 'iprint': 99}
    result = minimize(fx, init, bounds=bounds, method='L-BFGS-B', options=options)

    if result.success is False:
        raise ValueError('optimization failed')

    return Model(result, noise)
    
class Model:
    def __init__(self, result, noise):
        self.result = result
        self.noise = noise
        self.p = result.x[0:len(result.x) - 1]
        self.scale = result.x[len(result.x) - 1]
    
    def __repr__(self):
        return self.summary()

    def summary(self):
        return None

    def r2(self, x, f, y):
        yhat = self.predict(x, f)
        return 1 - np.nanvar(y-yhat, ddof=1) / np.nanvar(y, ddof=1)

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