import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import regionmask as rm
from scipy.stats import binom

from . import carto, line


def monthly(data, data_var='monthly', projection='albersUsa', clim=None, cmap='reds', clabel=None):
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
            if clim is not None:
                inds = color > clim[0]
            row |= carto(
                lat=lat[inds],
                lon=lon[inds],
                color=color[inds],
                clim=clim,
                cmap=cmap,
                clabel=clabel,
                size=size,
                width=270,
                height=170,
                projection=projection,
                title=str(months_labels[counter]),
                opacity=1,
            )
            counter += 1
        chart &= row

    return chart


def simple_map(
    data,
    data2=None,
    clabel=None,
    projection='albersUsa',
    clim=None,
    cmap='reds',
    title1=None,
    title2=None,
):
    lat = data['lat'].values.flatten()
    lon = data['lon'].values.flatten()
    color = data.values.flatten()
    inds = color >= clim[0]

    shape = data['lat'].shape
    size = (250 / shape[0]) * (400 / shape[1]) * 0.9

    row = alt.hconcat()

    row |= carto(
        lat=lat[inds],
        lon=lon[inds],
        color=color[inds],
        clim=clim,
        cmap=cmap,
        clabel=clabel,
        size=size,
        width=400,
        height=250,
        projection=projection,
        opacity=1,
        title=title1,
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
            width=160,
            height=100,
            projection=projection,
            opacity=1.0,
            title=title2,
        )
    return row


def summary(data, data_var='monthly', projection='albersUsa', clim=None, clabel=None, title=None):
    lat = data['lat'].values.flatten()
    lon = data['lon'].values.flatten()
    color = data.groupby('time.year').sum().mean('year').values.flatten()
    inds = color > clim[0]

    shape = data['lat'].shape
    size = (300 / shape[0]) * (500 / shape[1]) * 0.9

    column = alt.vconcat()

    x = data.groupby('time.year').sum()['year'].values
    y = data.groupby('time.year').sum().mean(['x', 'y']).values

    column &= line(x=x, y=y, width=300, height=122, strokeWidth=2, color='rgb(175,91,92)')

    x = data.groupby('time.month').mean()['month'].values
    y = data.groupby('time.month').mean().mean(['x', 'y']).values

    column &= line(x=x, y=y, width=300, height=122, strokeWidth=2, color='rgb(175,91,92)')

    row = alt.hconcat()

    row |= column

    row |= carto(
        lat=lat[inds],
        lon=lon[inds],
        color=color[inds],
        clim=clim,
        cmap='reds',
        clabel=clabel,
        size=size,
        width=500,
        height=300,
        projection=projection,
        title=title,
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
    mask,
    projection='albersUsa',
    clim=None,
    cmap='reds',
    percentage=True,
    comparison=True,
    add_map=True,
    clabel=None,
):
    lat = data['lat'].values.flatten()
    lon = data['lon'].values.flatten()

    color_model = model.groupby('time.year').sum().where(mask).mean('year').values.flatten()
    color_data = data.groupby('time.year').sum().where(mask).mean('year').values.flatten()

    inds_model = color_model >= clim[0]
    inds_data = color_data >= clim[0]

    shape = data['lat'].shape
    size = (300 / shape[0]) * (500 / shape[1]) * 0.85

    column = alt.vconcat()

    x = data.groupby('time.year').sum().where(mask)['year'].values
    y = data.groupby('time.year').sum().where(mask).mean(['x', 'y']).values
    yhat = model.groupby('time.year').sum().where(mask).mean(['x', 'y']).values

    column &= line(
        x=x,
        y=y,
        width=450,
        height=233,
        strokeWidth=2,
        opacity=1,
        color='darkgrey',
        ylabel='Burn area (fraction/year',
        xlabel='Time',
    ) + line(
        x=x,
        y=yhat,
        width=450,
        height=233,
        strokeWidth=2,
        opacity=1,
        color='#D77B40',
        ylabel='Burn area (fraction/year)',
        xlabel='Time',
    )

    x = data.groupby('time.month').mean()['month'].values
    y = data.groupby('time.month').mean().mean(['x', 'y']).values
    yhat = model.groupby('time.month').mean().mean(['x', 'y']).values

    column &= line(
        x=x,
        y=y,
        width=450,
        height=233,
        strokeWidth=2,
        opacity=1,
        color='darkgrey',
        ylabel='Burn area (fraction/month)',
        xlabel='Month',
    ) + line(
        x=x,
        y=yhat,
        width=450,
        height=233,
        strokeWidth=2,
        color='#D77B40',
        ylabel='Burn area (fraction/month)',
        xlabel='Month',
    )

    x = data['time'].values
    y = data.mean(['x', 'y']).values
    yhat = model.mean(['x', 'y']).values

    column &= line(
        x=x,
        y=y,
        width=450,
        height=233,
        strokeWidth=2,
        opacity=1,
        color='darkgrey',
        ylabel='Burn area (fraction/month)',
        xlabel='Time',
    ) + line(
        x=x,
        y=yhat,
        width=450,
        height=233,
        strokeWidth=2,
        color='#D77B40',
        ylabel='Burn area (fraction/month)',
        xlabel='Time',
    )

    chart = column

    if add_map:
        chart = alt.hconcat()
        chart |= column
        maps = alt.vconcat()
        maps &= carto(
            lat=lat[inds_model],
            lon=lon[inds_model],
            color=color_model[inds_model],
            clim=clim,
            cmap=cmap,
            clabel=clabel,
            size=size,
            width=450,
            height=233,
            projection=projection,
        )
        maps &= carto(
            lat=lat[inds_data],
            lon=lon[inds_data],
            color=color_data[inds_data],
            clim=clim,
            cmap=cmap,
            clabel=clabel,
            size=size,
            width=450,
            height=233,
            projection=projection,
        )
        chart |= maps

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
    clabel=None,
):

    a = data[data_var].mean("time").values.flatten()
    b = model[model_var].mean("time").values.flatten()
    inds = ~np.isnan(a) & ~np.isnan(b)
    spatial_corr = np.corrcoef(a[inds], b[inds])[0, 1] ** 2

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
        'spatial': spatial_corr,
    }

    for metric, performance in eval_metrics.items():
        print('performance at {} scale is: {}'.format(metric, performance))

    return eval_metrics, evaluation(
        data,
        model,
        data_var=data_var,
        model_var=model_var,
        projection=projection,
        clim=clim,
        cmap=cmap,
        percentage=percentage,
        comparison=comparison,
        clabel=clabel,
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


def calc_decadal_averages(simulation):
    decadal_averages = (
        simulation.sel(time=slice('2020', '2099')).coarsen(time=120).sum().load() / 10
    )
    return decadal_averages


def future_ts(decadal_averages, historical=None, domain=(0.0, 0.05)):
    df = decadal_averages.mean(dim=['x', 'y']).to_dataframe()

    df['time'] = df.index
    df_toplot = df.melt('time', var_name='gcm_scenario', value_name='probability')

    if historical is not None:
        historical['time'] = historical.index
        # the historical_historical is a hack to make the gcm/scenario splitting below not fail

        historical_df = historical.rename(columns={'historical': 'historical_historical'})
        df_toplot = df_toplot.append(
            historical_df.melt('time', var_name='gcm_scenario', value_name='probability'),
            ignore_index=True,
        )

    df_toplot['gcm'] = df_toplot.apply(lambda row: row.gcm_scenario.split('_')[0], axis=1)
    df_toplot['scenario'] = df_toplot.apply(lambda row: row.gcm_scenario.split('_')[1], axis=1)
    scenarios = ['historical', 'ssp245', 'ssp370', 'ssp585']
    colors_ = ['black', '#7eb36a', '#ea9755', '#f07071']

    base = alt.Chart(df_toplot).properties(width=550)
    line = base.mark_line().encode(
        alt.Y('probability:Q', scale=alt.Scale(domain=domain)),
        x='time',
        color=alt.Color('scenario', scale=alt.Scale(domain=scenarios, range=colors_)),
        strokeDash='gcm',
    )
    return line
