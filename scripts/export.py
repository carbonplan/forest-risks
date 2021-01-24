import numpy as np

from carbonplan_forests import load

# parameters

variables = ['ppt', 'tmean', 'pdsi', 'cwd', 'pet', 'vpd']
targets = list(map(lambda x: str(x), np.arange(2020, 2120, 20)))
models = ['CanESM5']
states = 'conus'
scenarios = ['ssp245', 'ssp370']
version = 'v10'
date = '01-19-20'

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
                sampling='annual',
            )
            df = df[keep_vars]
            df.to_csv(
                f'FIA-CMIP6-Long-{model}.{scenario}-{tlim[0]}.{tlim[1]}-{version}-{date}.csv',
                index=False,
            )
