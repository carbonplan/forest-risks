import numpy as np

from carbonplan_forests import load

# parameters

variables = ['ppt', 'tmean', 'pdsi', 'cwd', 'pet', 'vpd']
targets = list(map(lambda x: str(x), np.arange(2015, 2105, 10)))
targets_historical = list(map(lambda x: str(x), np.arange(1955, 2015, 10)))
models = ['CanESM5','MIROC-ES2L','FGOALS-g3']
states = 'conus'
scenarios = ['ssp245', 'ssp370', 'ssp585']
version = 'v11'
date = '02-07-21'

# generate wide data w/ terraclim

df = load.fia(store='local', states=states, group_repeats=True)
df = load.terraclim(
    store='local',
    tlim=(int(df['year_0'].min()), 2020),
    variables=variables,
    df=df,
    group_repeats=True,
    sampling='annual',
)
df.to_csv(f'FIA-TerraClim-Wide-{version}-{date}.csv', index=False)

# generate long data w/ terraclim

df = load.fia(store='local', states=states, clean=False)
df = load.terraclim(
    store='local', tlim=(int(df['year'].min()), 2020), variables=variables, df=df, sampling='annual'
)
df.to_csv(f'FIA-TerraClim-Long-{version}-{date}.csv', index=False)

# generate long data w/ cmip

df = load.fia(store='local', states=states, clean=False)
keep_vars = (
    ['lat', 'lon', 'plot_cn']
    + [var + '_min' for var in variables]
    + [var + '_mean' for var in variables]
    + [var + '_max' for var in variables]
)

for target in targets:
    historical = True if target == '2015' else False
    tlim = (str(int(target) - 5), str(int(target) + 4))
    for model in models:
        for scenario in scenarios:
            df = load.cmip(
                store='local',
                tlim=(int(tlim[0]), int(tlim[1])),
                variables=variables,
                df=df,
                model=model,
                scenario=scenario,
                historical=historical,
                sampling='annual',
            )
            df = df[keep_vars]
            df.to_csv(
                f'FIA-CMIP6-Long-{model}.{scenario}-{tlim[0]}.{tlim[1]}-{version}-{date}.csv',
                index=False,
            )

for target in targets_historical:
    tlim = (str(int(target) - 5), str(int(target) + 4))
    for model in models:
        df = load.cmip(
            store='local',
            tlim=(int(tlim[0]), int(tlim[1])),
            variables=variables,
            df=df,
            model=model,
            scenario='historical',
            historical=False,
            sampling='annual',
        )
        df = df[keep_vars]
        df.to_csv(
            f'FIA-CMIP6-Long-{model}.{scenario}-{tlim[0]}.{tlim[1]}-{version}-{date}.csv',
            index=False,
        )
