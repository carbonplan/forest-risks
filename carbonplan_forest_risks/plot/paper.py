import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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


def ts_pretty(ax, impact, ylims):
    ax.set_xlim(1970, 2100)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(ylims)
    ax.set_title("")


def multipanel_ts(results_dict, region_bboxes, fig_path):

    gcms = [
        ("MRI-ESM2-0", (0, 0)),
        ("MIROC-ES2L", (1, 0)),
        ("MPI-ESM1-2-LR", (2, 0)),
        ("ACCESS-ESM1-5", (3, 0)),
        ("ACCESS-CM2", (4, 0)),
        ("CanESM5-CanOE", (5, 0)),
    ]

    titles = {
        "fire": "Burn area\n(fraction/year)",
        "drought": "Drought-related\nmortality (%/year)",
        "insects": "Insect-related\nmortality (%/year)",
    }
    ylims = {
        "fire": (0, 0.05),
        "drought": (0, 2.5),
        "insects": (0, 1),
    }

    fig = plt.figure(figsize=(12, 10))
    full_gridspec = gridspec.GridSpec(4, 4, wspace=0.2, hspace=0.2)

    for col, region in enumerate(region_bboxes.keys()):
        ax = fig.add_subplot(full_gridspec[0, col], projection=cartopy_proj_albers())
        map_pretty(ax, title=region)

        ax.add_patch(
            mpatches.Rectangle(
                xy=[region_bboxes[region]['x'].start, region_bboxes[region]['y'].start],
                width=region_bboxes[region]['x'].stop - region_bboxes[region]['x'].start,
                height=region_bboxes[region]['y'].stop - region_bboxes[region]['y'].start,
                facecolor='grey',
                alpha=0.5,
            )
        )

    for impact_num, impact in enumerate(["fire", "drought", "insects"]):
        impact_axes = []
        for j, region in enumerate(region_bboxes.keys()):
            ax = fig.add_subplot(full_gridspec[1 + impact_num, j])

            results_dict[impact][region]["historical"].plot(ax=ax, color="k", zorder=60)
            for scenario in ["ssp245", "ssp370", "ssp585"]:
                impact_axes.append(ax)
                plot_future_ts_traces(ax, results_dict[impact][region]["future"], scenario, gcms)
            ts_pretty(ax, impact, ylims[impact])
            if impact != 'insects':
                ax.set_xticks([])
            if len(impact_axes) > 3:
                ax.set_yticks([])
            else:
                ax.set_ylabel(titles[impact])

    for format_string in ["svg", "png"]:
        plt.savefig(fig_path + format_string, format=format_string)


def plot_future_ts_traces(ax, ds, scenario, gcms):

    scenario_colors = {
        "ssp245": "#59A82F",
        "ssp370": "#D8B525",
        "ssp585": "#D83232",
    }
    scenario_colors_light = {
        "ssp245": "#DEEED5",
        "ssp370": "#F7F0D3",
        "ssp585": "#F7D6D6",
    }

    ssp_rename = {"ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}

    for (gcm, location) in gcms:
        ds.sel(gcm=gcm, scenario=scenario).plot(ax=ax, color=scenario_colors_light[scenario])

    ds.sel(scenario=scenario).mean(dim="gcm").plot(
        ax=ax, color=scenario_colors[scenario], label=ssp_rename[scenario], zorder=30
    )
