#!/usr/bin/env python
import os

import dask
import fsspec
import numpy as np
from cmip6_downscaling.workflows.share import get_cmip_runs
from dask.diagnostics import ProgressBar

from carbonplan_forests import load

# parameters
variables = ['ppt', 'tmean', 'pdsi', 'cwd', 'pet', 'vpd']
targets_future = list(map(lambda x: str(x), np.arange(2015, 2105, 10)))
targets_historical = list(map(lambda x: str(x), np.arange(1955, 2015, 10)))
prefix = 'az://carbonplan-scratch/forests'
states = 'conus'
version = 'v14'
date = '03-17-2021'


def write_df(df, name):
    fname = f'{prefix}/{name}-{version}-{date}.csv'
    print('writing', fname)
    with fsspec.open(
        fname,
        'w',
        account_name='carbonplan',
        account_key=os.environ['BLOB_ACCOUNT_KEY'],
    ) as f:
        df.to_csv(f, index=False)


def terraclimate_fia_wide():
    # generate wide data w/ terraclim
    df = load.fia(store='az', states=states, group_repeats=True)
    df = load.terraclim(
        store='az',
        tlim=(int(df['year_0'].min()), 2020),
        variables=variables,
        df=df,
        group_repeats=True,
        sampling='annual',
    )
    write_df(df, 'FIA-TerraClim-Wide')


def terraclimate_fia_long():
    # generate long data w/ terraclim

    df = load.fia(store='az', states=states, clean=False)
    df = load.terraclim(
        store='az',
        tlim=(int(df['year'].min()), 2020),
        variables=variables,
        df=df,
        sampling='annual',
    )

    write_df(df, 'FIA-TerraClim-Long')


def cmip_fia_long(cmip_table, downscaling):
    # generate long data w/ cmip

    df = load.fia(store='az', states=states, clean=False)
    keep_vars = (
        ['lat', 'lon', 'plot_cn']
        + [var + '_min' for var in variables]
        + [var + '_mean' for var in variables]
        + [var + '_max' for var in variables]
    )

    for i, row in cmip_table.iterrows():
        if 'hist' in row.scenario:
            targets = targets_historical
        else:
            targets = targets_future

        for target in targets:
            historical = target == '2015'
            tlim = (str(int(target) - 5), str(int(target) + 4))

            print(tlim, row)
            df = load.cmip(
                store='az',
                tlim=(int(tlim[0]), int(tlim[1])),
                variables=variables,
                df=df,
                model=row.model,
                scenario=row.scenario,
                member=row.member,
                historical=historical,
                downscaling=downscaling,
                sampling='annual',
            )
            df = df[keep_vars]
            write_df(
                df,
                f'{downscaling}/FIA-CMIP6-Long-{row.model}.{row.scenario}.{row.member}.-{tlim[0]}.{tlim[1]}',
            )


if __name__ == '__main__':

    terraclimate_fia_long()
    terraclimate_fia_wide()
    with dask.config.set(scheduler='processes'):
        with ProgressBar():
            df = get_cmip_runs()
            for downscaling in ['quantile-mapping']:  # , 'bias-corrected']:
                cmip_fia_long(df, downscaling=downscaling)
