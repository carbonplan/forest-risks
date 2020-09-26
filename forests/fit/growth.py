import numpy as np
from scipy.stats import gamma, norm
from scipy.optimize import minimize, Bounds

def logistic(x, p):
    a, b, c = p
    return a * (1 / (1 + c * np.exp(-b * (x))) - (1 / (1 + c))) * ((c + 1) / c)

def growth(x, y, noise='gamma', init=None):

    def loglik(x, y, p):
        a, b, c, scale = p
        _mu = logistic(x, [a, b, c])
        if noise == 'gamma':
            return -np.nansum(gamma.logpdf(y, _mu / scale, scale=scale))
        if noise == 'normal':
            return -np.nansum(norm.logpdf(y, loc=_mu, scale=scale))

    f = lambda p : loglik(x, y, p)

    lb = np.ones((4,)) * 0.00001
    lb[2] = 1
    ub = np.ones((4,)) * np.inf
    bounds = Bounds(lb, ub)
    if init is None:
        init = [np.nanmean(y), 0.1, 10, np.nanstd(y)]
    result = minimize(f, init, bounds=bounds, method='L-BFGS-B')

    return Model(result, noise)
    
class Model:
    def __init__(self, result, noise):
        self.result = result
        self.noise = noise
        self.p = result.x[0:3]
        self.scale = result.x[3]
    
    def __repr__(self):
        return self.summary(printed=False)

    def summary(self):
        return None

    def r2(self, x, y):
        yhat = self.predict(x)
        return 1 - np.nanvar(y-yhat, ddof=1) / np.nanvar(y, ddof=1)

    def predict(self, x):
        return logistic(x, self.p)

    def sample(self, x):
        mu = logistic(x, self.p)
        if self.noise == 'gamma':
            return np.random.gamma(mu / self.scale, scale=self.scale)
        if self.noise == 'normal':
            return np.random.normal(mu, scale=self.scale)