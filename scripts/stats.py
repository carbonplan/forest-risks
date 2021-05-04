import numpy as np
import pandas as pd

from carbonplan_forest_risks import collect, fit, load, prepare, utils


def score(x, y, model, da, method):
    roc = model.score_roc(x, y)
    r2 = model.score_r2(x, y)
    yhat = model.predict(x)
    prediction = collect.fire(yhat, da)

    a = da.groupby("time.year").sum().mean(["x", "y"]).values
    b = prediction["prediction"].groupby("time.year").sum().mean(["x", "y"]).values
    a = a - a.mean()
    b = b - b.mean()
    annual_r2 = 1 - np.sum((a - b) ** 2) / np.sum((a - np.mean(a)) ** 2)

    a = da.groupby("time.month").mean().mean(["x", "y"]).values
    b = prediction["prediction"].groupby("time.month").mean().mean(["x", "y"]).values
    a = a - a.mean()
    b = b - b.mean()
    seasonal_r2 = 1 - np.sum((a - b) ** 2) / np.sum((a - np.mean(a)) ** 2)

    a = da.mean("time").values.flatten()
    b = prediction["prediction"].mean("time").values.flatten()
    inds = ~np.isnan(a) & ~np.isnan(b)
    a = a[inds]
    b = b[inds]
    a = a - a.mean()
    b = b - b.mean()
    spatial_r2 = 1 - np.sum((a - b) ** 2) / np.sum((a - np.mean(a)) ** 2)

    bias = (prediction['prediction'].mean().values - da.mean().values) / da.mean().values

    return {
        'method': method,
        'roc': roc,
        'r2': r2,
        'annual_r2': annual_r2,
        'seasonal_r2': seasonal_r2,
        'spatial_r2': spatial_r2,
        'bias': bias,
    }


def crossval(x, y, selection, da, method):
    inds = np.zeros(da.shape)
    inds[selection, :, :] = 1
    inds = inds.flatten()
    train_x = x[inds == 0]
    train_y = y[inds == 0]
    test_x = x[inds == 1]
    test_y = y[inds == 1]
    model = fit.hurdle(train_x, train_y, log=False)
    return score(test_x, test_y, model, da[selection], method)


def shuffle(x, y, da, method='months'):
    if 'months' in method:
        inds = np.arange(len(da['time'].values)).copy()
        np.random.shuffle(inds)
        y = y.copy().reshape(da.shape)[inds].flatten()
    elif 'years' in method:
        years = pd.to_datetime(da['time'].values).year
        scrambled = years.unique().values.copy()
        np.random.shuffle(scrambled)
        inds = np.array([np.argwhere(years == y) for y in scrambled]).flatten()
        y = y.copy().reshape(da.shape)[inds].flatten()
    elif 'all' in method:
        y = y.copy()
        np.random.shuffle(y)
    else:
        print(f"shuffling method='{method}' not implemented")

    model = fit.hurdle(x, y, log=False)
    return score(x, y, model, da, method)


def append(df, results):
    return df.append(results, ignore_index=True)


coarsen = 4
tlim = (1984, 2018)
variables = ["ppt", "tmean", "cwd"]
store = "az"

print('[stats] loading data')
mask = (load.nlcd(store=store, year=2001).sel(band=[41, 42, 43, 90]).sum('band') > 0.25).astype(
    'float'
)
nftd = load.nftd(store=store, area_threshold=1500, coarsen=coarsen, mask=mask)
climate = load.terraclim(
    store=store,
    tlim=(tlim[0] - 1, tlim[1]),
    coarsen=coarsen,
    variables=variables,
    mask=mask,
    sampling="monthly",
)
mtbs = load.mtbs(store=store, coarsen=coarsen, tlim=tlim, mask=mask)

prepend = climate.sel(time=slice('1983', '1983'))
x, y = prepare.fire(
    climate.sel(time=slice('1984', '2018')),
    nftd,
    mtbs,
    add_global_climate_trends={
        'tmean': {'climate_prepend': prepend, 'rolling_period': 12},
        'ppt': {'climate_prepend': prepend, 'rolling_period': 12},
    },
    add_local_climate_trends=None,
)
x_z, x_mean, x_std = utils.zscore_2d(x)

df = pd.DataFrame()

# same training and testing
print('[stats] same training and testing')
model = fit.hurdle(x_z, y, log=False)
results = score(x_z, y, model, mtbs['monthly'], 'training')
df = append(df, results)

# cross validation
years = pd.to_datetime(mtbs['time'].values).year
nruns = 10

print('[stats] split halves cross validation')
for index in range(nruns):
    scrambled = years.unique().values.copy()
    np.random.shuffle(scrambled)
    subset = scrambled[0 : round(len(scrambled) / 2)]
    selection = [y in subset for y in years]
    results = crossval(x_z, y, selection, mtbs['monthly'], f'split_halves_{index}')
    df = append(df, results)

print('[stats] extrapolation cross validation')
for index, threshold in enumerate([2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]):
    selection = years > threshold
    results = crossval(x_z, y, selection, mtbs['monthly'], f'extrapolate_{index}')
    df = append(df, results)

print('[stats] shuffling')
for index in range(nruns):
    results = shuffle(x_z, y, mtbs['monthly'], f'shuffle_months_{index}')
    df = append(df, results)

for index in range(nruns):
    results = shuffle(x_z, y, mtbs['monthly'], f'shuffle_years_{index}')
    df = append(df, results)

for index in range(nruns):
    results = shuffle(x_z, y, mtbs['monthly'], f'shuffle_all_{index}')
    df = append(df, results)

df = df[['method', 'roc', 'r2', 'annual_r2', 'seasonal_r2', 'spatial_r2', 'bias']]

print(df)

df.to_csv('fire_stats.csv')
