import numpy as np
import pyproj
from sklearn.metrics import r2_score

RANDOM_SEED = 1


def score(y_true, y_pred):
    m = ~np.isnan(y_pred)
    if len(y_pred) != m.sum():
        print(f'found {len(m) - m.sum()} nans in y_pred')
    return r2_score(y_true[m], y_pred[m])


def interp(df, mask, var='biomass', spacing=4000):
    """
    Grid a set of lat/lon points to a grid defined by mask

    Parameters
    ----------
    df : pd.DataFrame
        Data points to be gridded in the form of a Pandas DataFrame with
        columns ``lat``, ``lon``, and ``var``.
    mask : xr.DataArray
        Target grid defintion. Must include a pyproj parsable crs attribute
        (e.g. ``mask.attrs['crs']``). Data should be between 0 and 1.
    var : str
        Name of column in df to grid.
    spacing : float
        Grid spacing in units defined by the masks crs.

    Returns
    -------
    grid : xr.DataArray
        Gridded data from df.
    """
    import verde as vd

    # extract the projection and grid info
    region = [mask.x.data[0], mask.x.data[-1], mask.y.data[-1], mask.y.data[0]]
    projection = pyproj.Proj(mask.attrs['crs'])

    coordinates = (df.lon.values, df.lat.values)

    proj_coords = projection(*coordinates)

    # split for validation... this may belong outside of this function
    train, test = vd.train_test_split(
        projection(*coordinates),
        df[var],
        random_state=RANDOM_SEED,
    )

    # fit the gridder
    chain = vd.Chain(
        [
            ('mean', vd.BlockReduce(np.mean, spacing=spacing * 0.25, region=region)),
            ('nearest', vd.ScipyGridder(method='linear')),
        ]
    )

    chain.fit(*train)
    # y_pred = chain.predict(test[0])
    # fit_score = score(test[1][0], y_pred)

    # make the grid
    grid = chain.grid(spacing=spacing, region=region, data_names=[var], dims=('y', 'x'))
    grid = vd.distance_mask(
        proj_coords,
        maxdist=4 * spacing,
        grid=grid,
    )
    grid = np.flipud(grid[var]) * mask
    grid.name = var

    return grid
