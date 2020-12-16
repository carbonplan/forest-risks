from carbonplan_forests import load

data_vars = ['ppt','tavg','pdsi','cwd','pet','vpd']
data_aggs = ['sum','mean','mean','mean','mean','mean']

fia = load.fia(store='local', group_repeats=True)

climate = load.terraclim(
    store='local',
    tlim=(int(df['year_0'].min()), 2020),
    data_vars=data_vars,
    data_aggs=data_aggs,
    df=df,
    group_repeats=True,
)

df = load.join(fia, climate)

df.to_csv('FIA-TerraClim-Grouped-v6-12-03-20.csv', index=False)

df = load.fia(store='local', clean=False)

df = load.terraclim(
    store='local',
    tlim=(int(df['year'].min()), 2020),
    data_vars=data_vars,
    data_aggs=data_aggs,
    df=df
)