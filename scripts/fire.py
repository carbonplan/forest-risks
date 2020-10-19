import warnings

import numpy as np
import xarray as xr
from carbonplan.forests import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.simplefilter('ignore', category=RuntimeWarning)


def zscore_2d(x, mean=None, std=None):
    recomputing = False
    if mean is None or std is None:
        recomputing = True
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    if recomputing:
        return mean, std, (x - mean) / std
    else:
        return (x - mean) / std


coarsen = None
threshold = 0.25
vars = ['ppt', 'tmax', 'tmax', 'tmax', 'tmin', 'tmin', 'tmin']
aggs = ['sum', 'mean', 'min', 'max', 'mean', 'min', 'max']

print('[fire] fitting model')
y, time = load.mtbs(store='local', return_type='numpy', coarsen=coarsen)

t = len(time)
n, m = y['burned_area'].shape[1:]

X = load.terraclim(
    store='local',
    return_type='numpy',
    coarsen=coarsen,
    tlim=(1984, 2018),
    mean=False,
    vars=vars,
    aggs=aggs,
)

mask = load.nlcd(
    store='local', return_type='numpy', coarsen=coarsen, classes=[41, 42, 43, 90], year=2001
)

all_mask = load.nlcd(store='local', return_type='numpy', coarsen=coarsen, classes='all', year=2001)

final_mask = load.nlcd(
    store='local', return_type='numpy', coarsen=coarsen, classes=[41, 42, 43, 90], year=2016
)
final_mask = final_mask * (final_mask > 0.5)

X['forested'] = np.tile(mask, [t, 1, 1])

for key in X.keys():
    X[key][np.tile(all_mask == 0, [t, 1, 1])] = np.NaN

X_s = np.asarray(
    [np.tile(np.nanmean(X[var], axis=0).flatten(), [1, t]).squeeze() for var in X.keys()]
)

keys = list(X.keys())
keys.remove('forested')
X_t = np.asarray(
    [
        np.tile(np.nanmean(X[var].reshape(t, n * m), axis=1), [n * m, 1]).T.flatten().squeeze()
        for var in keys
    ]
).squeeze()
X_st = np.vstack([X_s, X_t]).T

y_b = y['burned_area'].flatten() > threshold

inds = (~np.isnan(X_st.sum(axis=1))) & (~np.isnan(y_b))

train_mean, train_std, zscored = zscore_2d(X_st[inds])
model = LogisticRegression(fit_intercept=True, max_iter=500, solver='lbfgs')
model.fit(zscored, y_b[inds])

y_hat = model.predict_proba(zscored)[:, 1]

print(roc_auc_score(y_b[inds], y_hat))


print('[fire] evaluating predictions')
targets = list(map(lambda x: str(x), np.arange(2020, 2120, 20)))
scenarios = ['ssp245', 'ssp370', 'ssp585']
ds = xr.Dataset()
for scenario in tqdm(scenarios):
    results = []
    for target in targets:
        tlim = (int(target) - 10, int(target) + 9)

        Xhat = load.cmip(
            model='BCC-CSM2-MR',
            scenario=scenario,
            store='local',
            tlim=tlim,
            coarsen=coarsen,
            mean=False,
            return_type='numpy',
            vars=vars,
            aggs=aggs,
        )

        # for key in Xhat.keys():
        #     Xhat[key] = Xhat[key].reshape([1, n, m])

        t = Xhat['tmax_max'].shape[0]

        Xhat['forested'] = np.tile(mask, [t, 1, 1])

        for key in Xhat.keys():
            Xhat[key][np.tile(all_mask == 0, [t, 1, 1])] = np.NaN

        Xhat_s = np.asarray(
            [
                np.tile(np.nanmean(Xhat[var], axis=0).flatten(), [1, t]).squeeze()
                for var in Xhat.keys()
            ]
        )

        keys = list(Xhat.keys())
        keys.remove('forested')
        Xhat_t = np.asarray(
            [
                np.tile(np.nanmean(Xhat[var].reshape(t, n * m), axis=1), [n * m, 1])
                .T.flatten()
                .squeeze()
                for var in keys
            ]
        ).squeeze()

        Xhat_st = np.vstack([Xhat_s, Xhat_t]).T

        inds = ~np.isnan(Xhat_st.sum(axis=1))

        y_proj = model.predict_proba(zscore_2d(Xhat_st[inds], mean=train_mean, std=train_std))[:, 1]

        y_proj_full = np.zeros((t, n, m)).flatten()
        y_proj_full[inds] = y_proj
        y_proj_full = y_proj_full.reshape(t, n, m).squeeze()
        y_proj_mean = final_mask * np.nanmean(y_proj_full, axis=0)
        results.append(xr.DataArray(y_proj_mean))

    da = xr.concat(results, dim=xr.Variable('year', targets))
    ds[scenario] = da

ds.to_zarr('data/fire.zarr')
