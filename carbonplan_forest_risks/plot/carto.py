import altair as alt
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
from carbonplan_data import utils
from cartopy.io import shapereader
from vega_datasets import data as vega_data


def carto(
    data=None,
    lon='lon',
    lat='lat',
    projection='albersUsa',
    color=None,
    cmap=None,
    clabel=None,
    clim=None,
    size=5,
    width=650,
    height=400,
    title=None,
    opacity=1,
):
    if data is None:
        df = pd.DataFrame({'lon': lon, 'lat': lat})
        if color is not None and hasattr(color, 'name') and clabel is None:
            clabel = color.name
        if color is not None:
            df['color'] = color
            _color = 'color'
    else:
        df = data
        _color = color
    if clabel is None:
        clabel = 'color'

    clegend = alt.Legend(title=clabel)

    def color_scaled(color):
        if clim is None and cmap is None:
            return alt.Color(color)
        elif clim is None and cmap is not None:
            return alt.Color(color, legend=clegend, scale=alt.Scale(scheme=cmap))
        elif clim is not None and cmap is None:
            return alt.Color(color, legend=clegend, scale=alt.Scale(domain=clim, clamp=True))
        elif clim is not None and cmap is not None:
            if type(cmap) == str:
                return alt.Color(
                    color, legend=clegend, scale=alt.Scale(domain=clim, scheme=cmap, clamp=True)
                )
            else:
                return alt.Color(
                    color, legend=clegend, scale=alt.Scale(domain=clim, range=cmap, clamp=True)
                )

    states = alt.topo_feature(vega_data.us_10m.url, 'states')

    background = (
        alt.Chart(states)
        .mark_geoshape(
            fill='white',
            stroke='black',
            strokeWidth=0.3,
        )
        .project(projection)
    )

    if title:
        background = background.properties(width=width, height=height, title=title)
    else:
        background = background.properties(width=width, height=height)

    if color is None:
        geomap = (
            alt.Chart(df)
            .mark_square(size=size, color='rgb(150,150,150)', opacity=opacity)
            .encode(longitude='lon:Q', latitude='lat:Q')
            .project(type=projection)
            .properties(width=width, height=height)
        )
    else:
        geomap = (
            alt.Chart(df)
            .mark_square(size=size, opacity=opacity)
            .encode(longitude='lon:Q', latitude='lat:Q', color=color_scaled(_color))
            .project(type=projection)
            .properties(width=width, height=height)
        )

    return background + geomap


def cartopy_proj_albers(region='conus'):
    return ccrs.AlbersEqualArea(
        central_longitude=-96, central_latitude=23, standard_parallels=(29.5, 45.5)
    )


def cartopy_borders():
    states_df = gpd.read_file(
        shapereader.natural_earth('50m', 'cultural', 'admin_1_states_provinces')
    )
    states = (
        states_df.loc[states_df['iso_a2'] == 'US']
        .set_crs(epsg=4326)
        .to_crs(utils.projections('albers', 'conus'))
        .drop([49, 60])['geometry']
        .values
    )

    countries_df = gpd.read_file(shapereader.natural_earth('50m', 'cultural', 'admin_0_countries'))
    countries = (
        countries_df[countries_df['ADMIN'] == 'United States of America']
        .set_crs(epsg=4326)
        .to_crs(utils.projections('albers', 'conus'))['geometry']
        .values
    )
    return states, countries
