import altair as alt
import pandas as pd


def line(data=None, x=None, y=None, color=None, cmap=None, clim=None, xlim=None, ylim=None):
    """
    plot two variables optionally colored by some feature
    """
    if data is None:
        df = pd.DataFrame({'x': x, 'y': y})
        _x = 'x'
        _y = 'y'
        if color is not None:
            df['color'] = color
            _color = 'color'
    else:
        df = data
        _x = x
        _y = y
        _color = color

    def x_scaled(x):
        return alt.X(x) if xlim is None else alt.X(x, scale=alt.Scale(domain=xlim, clamp=True))

    def y_scaled(y):
        return alt.Y(y) if ylim is None else alt.Y(y, scale=alt.Scale(domain=ylim, clamp=True))

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
        line = (
            alt.Chart(df)
            .mark_line(strokeWidth=3, color='rgb(250,100,150)')
            .encode(
                x=x_scaled(_x),
                y=y_scaled(_y),
            )
            .properties(width=350, height=300)
        )
    else:
        line = (
            alt.Chart(df)
            .mark_line(strokeWidth=3)
            .encode(x=x_scaled(_x), y=y_scaled(_y), color=color_scaled(_color))
            .properties(width=350, height=300)
        )

    return line
