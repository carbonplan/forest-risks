import altair as alt

from . import carto, line


def monthly(data, data_var='vlf', projection='albersUsa', clim=None):
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


def summary(data, data_var='vlf', projection='albersUsa', clim=None):
    lat = data['lat'].values.flatten()
    lon = data['lon'].values.flatten()
    color = data[data_var].mean('time').values.flatten()
    inds = color > clim[0]

    shape = data['lat'].shape
    size = (300 / shape[0]) * (500 / shape[1]) * 0.9

    column = alt.vconcat()

    x = data.groupby('time.year').mean()['year'].values
    y = data.groupby('time.year').mean().mean(['x', 'y'])[data_var].values

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
    color = model[model_var].mean('time').values.flatten()
    inds = color > clim[0]

    shape = data['lat'].shape
    size = (300 / shape[0]) * (500 / shape[1]) * 0.85

    column = alt.vconcat()

    x = data.groupby('time.year').mean()['year'].values
    y = data.groupby('time.year').mean().mean(['x', 'y'])[data_var].values
    yhat = model.groupby('time.year').mean().mean(['x', 'y'])[model_var].values

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
