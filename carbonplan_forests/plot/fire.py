import altair as alt
import pandas as pd
import scipy as sp

from . import carto, line


def monthly(data, data_var='monthly', projection='albersUsa', clim=None):
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
                cmap='reds',
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


def simple_map(data, clabel=None, projection='albersUsa', clim=None, cmap='reds'):
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


def evaluation(data, model, data_var='vlf', model_var='prob', projection='albersUsa', clim=None):
    lat = data['lat'].values.flatten()
    lon = data['lon'].values.flatten()
    color = model[model_var].groupby('time.year').sum().mean('year').values.flatten()
    inds = color > clim[0]

    shape = data['lat'].shape
    size = (300 / shape[0]) * (500 / shape[1]) * 0.85

    column = alt.vconcat()

    x = data.groupby('time.year').sum()['year'].values
    y = data.groupby('time.year').sum().mean(['x', 'y'])[data_var].values
    yhat = model.groupby('time.year').sum().mean(['x', 'y'])[model_var].values

    column &= line(
        x=x, y=y, width=300, height=122, strokeWidth=2, opacity=0.5, color='rgb(175,91,92)'
    ) + line(x=x, y=yhat, width=300, height=122, strokeWidth=2, color='rgb(175,91,92)')

    x = data.groupby('time.month').mean()['month'].values
    y = data.groupby('time.month').mean().mean(['x', 'y'])[data_var].values
    yhat = model.groupby('time.month').mean().mean(['x', 'y'])[model_var].values

    column &= line(
        x=x, y=y, width=300, height=122, strokeWidth=2, opacity=0.5, color='rgb(175,91,92)'
    ) + line(x=x, y=yhat, width=300, height=122, strokeWidth=2, color='rgb(175,91,92)')

    chart = alt.hconcat()

    chart |= column

    chart |= carto(
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

    return chart


def package_for_altair(fire, climate, label='value'):
    data = fire.mean(dim=['x', 'y']).monthly.to_dataframe().rename(columns={'monthly': 'fire'})
    data['temp'] = climate['tmax'].mean(dim=['x', 'y']).values
    data['ppt'] = climate['ppt'].mean(dim=['x', 'y']).values
    for variable in data.columns:
        data[variable] = sp.stats.zscore(data[variable])
    data['year'] = data.index.year
    data['month'] = data.index.month
    data = data.reset_index().drop(['time'], axis=1)
    data = pd.melt(
        data,
        id_vars=['year', 'month'],
        value_vars=['fire', 'temp', 'ppt'],
        value_name=label,
        ignore_index=False,
    )
    return data


def multipanel_slider(data, year_limits, region_labels):

    if len(region_labels) != 4:
        raise Exception('SO SORRY! Four is the magic number- we only accept four regions right now')
    slider = alt.binding_range(min=year_limits[0], max=year_limits[0], step=1)
    select_year = alt.selection_single(
        name="year", fields=['year'], bind=slider, init={'year': year_limits[0]}
    )

    base = (
        alt.Chart(data)
        .mark_line()
        .encode(x='month:O', color='variable:N')
        .add_selection(select_year)
        .properties(width=200, height=100)
        .transform_filter(select_year)
    )
    charts = {}
    for region in region_labels:
        charts[region] = base.encode(y='{}:Q'.format(region))
    full_chart = (
        charts[region_labels[0]]
        | charts[region_labels[1]]
        | charts[region_labels[2]]
        | charts[region_labels[3]]
    )
    return full_chart
