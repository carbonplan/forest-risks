import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import regionmask as rm
from scipy.stats import binom

from . import carto, line


def monthly(data, data_var='monthly', projection='albersUsa', clim=None, cmap='reds'):
    lat = data['lat'].values.flatten()
    lon = data['lon'].values.flatten()

    shape = data['lat'].shape
    size = (170 / shape[0]) * (270 / shape[1]) * 0.9

    months = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    months_labels = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov']
    fires = data[data_var].groupby('time.month').mean().isel(month=months)

    chart = alt.vconcat()
    counter = 0
    for month in range(int(len(months) / 3)):
        row = alt.hconcat()
        for column in range(3):
            color = fires[counter].values.flatten()
            inds = color > clim[0]
            row |= carto(
                lat=lat[inds],
                lon=lon[inds],
                color=color[inds],
                clim=clim,
                cmap=cmap,
                clabel=data_var,
                size=size,
                width=270,
                height=170,
                projection=projection,
                title=str(months_labels[counter]),
            )
            counter += 1
        chart &= row

    return chart.configure_view(strokeOpacity=0)


def simple_map(data, data2=None, clabel=None, projection='albersUsa', clim=None, cmap='reds'):
    lat = data['lat'].values.flatten()
    lon = data['lon'].values.flatten()
    color = data.values.flatten()
    inds = color > clim[0]

    shape = data['lat'].shape
    size = (300 / shape[0]) * (500 / shape[1]) * 0.9

    row = alt.hconcat()

    row |= carto(
        lat=lat[inds],
        lon=lon[inds],
        color=color[inds],
        clim=clim,
        cmap=cmap,
        clabel=clabel,
        size=size,
        width=500,
        height=300,
        projection=projection,
    )
    if data2 is not None:
        color = data2.values.flatten()
        inds = color > clim[0]

        row |= carto(
            lat=lat[inds],
            lon=lon[inds],
            color=color[inds],
            clim=clim,
            cmap=cmap,
            clabel=clabel,
            size=size,
            width=500,
            height=300,
            projection=projection,
        )
    return row


def summary(data, data_var='monthly', projection='albersUsa', clim=None):
    lat = data['lat'].values.flatten()
    lon = data['lon'].values.flatten()
    color = data[data_var].groupby('time.year').sum().mean('year').values.flatten()
    inds = color > clim[0]

    shape = data['lat'].shape
    size = (300 / shape[0]) * (500 / shape[1]) * 0.9

    column = alt.vconcat()

    x = data.groupby('time.year').sum()['year'].values
    y = data.groupby('time.year').sum().mean(['x', 'y'])[data_var].values

    column &= line(x=x, y=y, width=300, height=122, strokeWidth=2, color='rgb(175,91,92)')

    x = data.groupby('time.month').mean()['month'].values
    y = data.groupby('time.month').mean().mean(['x', 'y'])[data_var].values

    column &= line(x=x, y=y, width=300, height=122, strokeWidth=2, color='rgb(175,91,92)')

    row = alt.hconcat()

    row |= column

    row |= carto(
        lat=lat[inds],
        lon=lon[inds],
        color=color[inds],
        clim=clim,
        cmap='reds',
        clabel=data_var,
        size=size,
        width=500,
        height=300,
        projection=projection,
    )

    return row


def performance(model, obs, percentage=True):
    if percentage:
        return (model - obs) / obs * 100
    else:
        return model - obs


def evaluation(
    data,
    model,
    data_var='vlf',
    model_var='prob',
    projection='albersUsa',
    clim=None,
    cmap='reds',
    percentage=True,
    comparison=True,
    add_map=True,
):
    lat = data['lat'].values.flatten()
    lon = data['lon'].values.flatten()

    color = performance(
        model[model_var].groupby('time.year').sum().mean('year').values,
        data[data_var].groupby('time.year').sum().mean('year').values,
        percentage=percentage,
    ).flatten()
    if comparison:
        # hacky!!!! WARNINGGGGGG THIS IS HORRIBLEEEE just want to make all inds active
        inds = color > -99999
    else:
        inds = color > clim[0]

    shape = data['lat'].shape
    size = (300 / shape[0]) * (500 / shape[1]) * 0.85

    column = alt.vconcat()

    x = data.groupby('time.year').sum()['year'].values
    y = data.groupby('time.year').sum().mean(['x', 'y'])[data_var].values
    yhat = model.groupby('time.year').sum().mean(['x', 'y'])[model_var].values

    column &= line(
        x=x,
        y=y,
        width=300,
        height=122,
        strokeWidth=2,
        opacity=0.5,
        color='rgb(175,91,92)',
        ylabel='Probability',
    ) + line(
        x=x,
        y=yhat,
        width=300,
        height=122,
        strokeWidth=2,
        color='rgb(175,91,92)',
        ylabel='Probability',
    )

    x = data.groupby('time.month').mean()['month'].values
    y = data.groupby('time.month').mean().mean(['x', 'y'])[data_var].values
    yhat = model.groupby('time.month').mean().mean(['x', 'y'])[model_var].values

    column &= line(
        x=x,
        y=y,
        width=300,
        height=122,
        strokeWidth=2,
        opacity=0.5,
        color='rgb(175,91,92)',
        ylabel='Probability',
    ) + line(
        x=x,
        y=yhat,
        width=300,
        height=122,
        strokeWidth=2,
        color='rgb(175,91,92)',
        ylabel='Probability',
    )

    chart = alt.hconcat()
    chart |= column

    if add_map:

        chart |= carto(
            lat=lat[inds],
            lon=lon[inds],
            color=color[inds],
            clim=clim,
            cmap=cmap,
            clabel=data_var,
            size=size,
            width=500,
            height=300,
            projection=projection,
        )

    return chart


def full_eval(
    data,
    model,
    data_var='vlf',
    model_var='prediction',
    projection='albersUsa',
    clim=None,
    cmap='reds',
    percentage=True,
    comparison=True,
):

    # a = data[data_var].mean("time").values.flatten()
    # b = model[model_var].mean("time").values.flatten()
    # inds = ~np.isnan(a) & ~np.isnan(b)
    # spatial_corr = np.corrcoef(a[inds], b[inds])[0, 1] ** 2

    eval_metrics = {
        'seasonal': np.corrcoef(
            data[data_var].groupby("time.month").mean().mean(["x", "y"]),
            model[model_var].groupby("time.month").mean().mean(["x", "y"]),
        )[0, 1]
        ** 2,
        'annual': np.corrcoef(
            data[data_var].groupby("time.year").mean().mean(["x", "y"]),
            model[model_var].groupby("time.year").mean().mean(["x", "y"]),
        )[0, 1]
        ** 2,
        # 'spatial': spatial_corr,
    }

    for metric, performance in eval_metrics.items():
        print('performance at {} scale is: {}'.format(metric, performance))

    return evaluation(
        data,
        model,
        data_var=data_var,
        model_var=model_var,
        projection=projection,
        clim=clim,
        cmap=cmap,
        percentage=percentage,
        comparison=comparison,
    )


def integrated_risk(p):
    return (1 - binom.cdf(0, 100, p)) * 100


def supersection(data, varname, store='az'):
    from palettable.colorbrewer.sequential import YlOrRd_9

    cmap = YlOrRd_9.mpl_colormap
    regions = gpd.read_file(
        "https://storage.googleapis.com/carbonplan-data/raw/ecoregions/supersections.geojson"
    )
    masks = rm.mask_3D_geopandas(regions, data)

    groupby = data.groupby('time.year').sum().mean('year')
    risks = np.asarray(
        [
            groupby[varname].where(masks.sel(region=i)).mean(['x', 'y']).values.item()
            for i in masks['region']
        ]
    )
    regions.to_crs('EPSG:5070').plot(
        integrated_risk(risks),
        figsize=[15, 8],
        cmap=cmap,
        edgecolor=[0, 0, 0],
        linewidth=0.3,
        vmin=0,
        vmax=25,
        legend=True,
    )
    plt.axis('off')
