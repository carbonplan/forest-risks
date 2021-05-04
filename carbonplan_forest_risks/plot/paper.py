import matplotlib as mpl

from . import cartopy_borders, cartopy_proj_albers


def map_pretty(ax, title=''):
    state_borders, us_border = cartopy_borders()
    ax.add_geometries(
        state_borders,
        facecolor='none',
        edgecolor='k',
        crs=cartopy_proj_albers(),
        linewidth=0.3,
        zorder=0,
    )
    ax.add_geometries(
        us_border,
        facecolor='none',
        edgecolor='k',
        crs=cartopy_proj_albers(),
        linewidth=0.3,
        zorder=0,
    )
    ax.axis('off')
    ax.set_extent([-125, -70, 20, 50])
    ax.text(0.77, 0.96, title, transform=ax.transAxes)


# def
def add_colorbar(
    fig,
    to_plot=None,
    x_location=1.08,
    y_location=0.76,
    height=0.12,
    width=0.018,
    vmin=None,
    vmax=None,
    cbar_label='',
    cmap='viridis',
):
    cax = fig.add_axes([x_location, y_location, width, height])
    cax.text(
        0.5,
        -0.08,
        vmin,
        transform=cax.transAxes,
        horizontalalignment='center',
        verticalalignment='center',
    )
    cax.text(
        0.5,
        1.08,
        vmax,
        transform=cax.transAxes,
        horizontalalignment='center',
        verticalalignment='center',
    )
    cax.text(
        1.8,
        0.5,
        cbar_label,
        transform=cax.transAxes,
        verticalalignment='center',
        multialignment='center',
        rotation=-90,
    )
    if to_plot is not None:
        cbar = fig.colorbar(to_plot, cax=cax, orientation='vertical')
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical'
        )
    cbar.outline.set_visible(False)
    cbar.set_ticks([])
    return cbar


# def model_comparisons():
