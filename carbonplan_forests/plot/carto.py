import altair as alt
import pandas as pd


def carto(
    data=None, lon='lon', lat='lat', projection='albersUsa', color=None, cmap=None, clim=None
):
    if data is None:
        df = pd.DataFrame({'lon': lon, 'lat': lat})
        if color is not None:
            df['color'] = color
            _color = 'color'
    else:
        df = data
        _color = color

    def color_scaled(color):
        if clim is None and cmap is None:
            return alt.Color(color)
        elif clim is None and cmap is not None:
            return alt.Color(color, scale=alt.Scale(scheme=cmap))
        elif clim is not None and cmap is None:
            return alt.Color(color, scale=alt.Scale(domain=clim, clamp=True))
        elif clim is not None and cmap is not None:
            return alt.Color(color, scale=alt.Scale(domain=clim, scheme=cmap, clamp=True))

    if color is None:
        geomap = (
            alt.Chart(df)
            .mark_circle(size=5, color='rgb(150,150,150)')
            .encode(longitude='lon:Q', latitude='lat:Q')
            .project(type=projection)
            .properties(width=650, height=400)
        )
    else:
        geomap = (
            alt.Chart(df)
            .mark_circle(size=3)
            .encode(longitude='lon:Q', latitude='lat:Q', color=color_scaled(_color))
            .project(type=projection)
            .properties(width=650, height=400)
        )

    return geomap
