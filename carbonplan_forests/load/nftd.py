import numpy as np
import rasterio
import xarray as xr

from .. import setup


def load_rio(f):
    src = rasterio.open(f)
    return src.read(1)


def nftd(store='az', groups='all', coarsen=None, append_all=False, mask=None, area_threshold=None):
    path = setup.loading(store)

    if groups == 'all':
        groups = [
            100,
            120,
            140,
            160,
            180,
            200,
            220,
            240,
            260,
            280,
            300,
            320,
            340,
            360,
            370,
            400,
            500,
            600,
            700,
            800,
            900,
            910,
            920,
            940,
            950,
        ]

    bands = xr.concat(
        [
            xr.open_rasterio(
                (path / f'carbonplan-data/processed/nftd/conus/4000m/group_g{g}.tif').as_uri()
            )[0]
            for g in groups
        ],
        dim=xr.Variable('band', groups),
    )

    if area_threshold is not None:
        band_inds = bands['band'].values
        areas = np.asarray([bands.sel(band=band).sum(['x', 'y']).item() for band in band_inds])
        small_inds = areas < area_threshold
        large_inds = areas >= area_threshold
        matches = {}

        for band in band_inds[small_inds]:
            corrs = []
            for other in band_inds[~small_inds]:
                x = bands.sel(band=band).values.flatten()
                y = bands.sel(band=other).values.flatten()
                notnan = ~np.isnan(x) & ~np.isnan(y)
                corrs.append(np.corrcoef(x[notnan], y[notnan])[0, 1])
            ind = np.argmax(corrs)
            matches[band] = band_inds[~small_inds][ind]

        for source, target in matches.items():
            bands.loc[target] += bands.loc[source]

        bands = bands[large_inds]

    if append_all:
        total = bands.sum('band').values[np.newaxis, :, :]
        total = xr.DataArray(total, dims=['band', 'y', 'x'], coords={'band': [0]})
        bands = xr.concat([bands, total], dim='band')

    if mask is not None:
        vals = mask.values
        vals[vals == 0] = np.NaN
        bands = bands * vals

    if coarsen:
        bands = bands.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()

    bands.load()
    return bands
