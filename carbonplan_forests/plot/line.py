import altair as alt
import pandas as pd


def line(
    x=None,
    y=None,
    color=None,
    data=None,
    cmap=None,
    clim=None,
    xlim=None,
    ylim=None,
    xlabel=None,
    ylabel=None,
    width=350,
    height=200,
    strokeWidth=2,
    opacity=1,
):
    """
    plot two variables optionally colored by some feature
    """
    if data is None:
        if hasattr(x, 'name') and xlabel is None:
            xlabel = x.name
        if hasattr(y, 'name') and ylabel is None:
            ylabel = y.name
        df = pd.DataFrame({'x': x, 'y': y})
        _x = 'x'
        _y = 'y'
        if color is not None:
            df['color'] = color
            _color = 'color'
    else:
        if hasattr(data[x], 'name') and xlabel is None:
            xlabel = data[x].name
        if hasattr(data[y], 'name') and ylabel is None:
            ylabel = data[y].name
        df = data
        _x = x
        _y = y
        _color = color

    if xlabel is None:
        xlabel = 'x'

    if ylabel is None:
        ylabel = 'y'

    xaxis = alt.Axis(title=xlabel)
    yaxis = alt.Axis(title=ylabel)

    def x_scaled(x):
        if xlim is None:
            return alt.X(x, axis=xaxis)
        else:
            return alt.X(x, axis=xaxis, scale=alt.Scale(domain=xlim, clamp=True))

    def y_scaled(y):
        if ylim is None:
            return alt.Y(y, axis=yaxis)
        else:
            return alt.Y(y, axis=yaxis, scale=alt.Scale(domain=ylim, clamp=True))

    def color_scaled(color):
        if clim is None and cmap is None:
            return alt.Color(color, scale=alt.Scale(scheme='viridis'))
        elif clim is None and cmap is not None:
            return alt.Color(color, scale=alt.Scale(scheme=cmap))
        elif clim is not None and cmap is None:
            return alt.Color(color, scale=alt.Scale(domain=clim))
        elif clim is not None and cmap is not None:
            return alt.Color(color, scale=alt.Scale(domain=clim, scheme=cmap, clamp=True))

    if color is None:
        color = 'rgb(250,100,150)'

    if type(color) is str:
        line = (
            alt.Chart(df)
            .mark_line(strokeWidth=strokeWidth, color=color, opacity=opacity)
            .encode(
                x=x_scaled(_x),
                y=y_scaled(_y),
            )
            .properties(width=width, height=height)
        )
    else:
        line = (
            alt.Chart(df)
            .mark_line(strokeWidth=strokeWidth, opacity=opacity)
            .encode(x=x_scaled(_x), y=y_scaled(_y), color=color_scaled(_color))
            .properties(width=width, height=height)
        )

    return line
