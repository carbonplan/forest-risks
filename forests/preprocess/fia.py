from dask.dataframe import read_parquet
from pandas import read_csv, merge, Series
from numpy import full, isnan, nan, nanmean, nansum
from networkx import Graph, connected_components

def fia(states):
    if type(states) is str:
        if states == 'all':
            df = read_csv('gs://carbonplan-data/raw/fia/REF_RESEARCH_STATION.csv')
            [process_by_state(state) for state in df['STATE_ABBR']]
        else:
            process_by_state(states)
    else:
        [process_by_state(state) for state in states]
    return


def generate_uids(data, prev_cn_var="PREV_PLT_CN"):
    """
    Generate dict mapping ever CN to a unique group, allows tracking single plot/tree through time
    Can change `prev_cn_var` to apply to tree (etc)
    """
    g = Graph()
    for row in data[["CN", prev_cn_var]].itertuples():
        if ~isnan(row.PREV_PLT_CN):  # has ancestor, add nodes + edge
            g.add_edge(row.CN, row.PREV_PLT_CN)
        else:
            g.add_node(row.CN)  # no ancestor, potentially not resampled

    uids = {}
    for uid, subgraph in enumerate(connected_components(g)):
        for node in subgraph:
            uids[node] = uid
    return uids


def tree_stats(df):
    idx = (df["STATUSCD"] == 1) & (df["DIA"] > 1.0) & (df["DIACHECK"] == 0)
    out = Series(
        full(6, nan), index=["DIA", "HT", "TOTAGE", "SITREE", "BIOMASS", "BALIVE"]
    )
    if sum(idx) > 0:
        out["DIA"] = nanmean(df["DIA"][idx])
        out["HT"] = nanmean(df["HT"][idx])
        out["TOTAGE"] = nanmean(df["TOTAGE"][idx])
        out["SITREE"] = nanmean(df["SITREE"][idx])
        out["BIOMASS"] = nansum((df["CARBON_AG"][idx] * 2) * df["TPA_UNADJ"][idx])
        out["BALIVE"] = nansum(
            ((df["DIA"][idx] ** 2) * 0.005454) * df["TPA_UNADJ"][idx]
        )
    return out


def process_by_state(state):
    state = state.lower()
    print(f'loading tables for {state}')
    tree_df = read_parquet(f'gs://carbonplan-scratch/fia/states/tree_{state}.parquet').compute()
    plot_df = read_parquet(f'gs://carbonplan-scratch/fia/states/plot_{state}.parquet').compute()
    cond_df = read_parquet(f'gs://carbonplan-scratch/fia/states/cond_{state}.parquet').compute()

    plot_usevars = [
        "CN", 
        "PREV_PLT_CN", 
        "LAT", 
        "LON", 
        "ELEV", 
        "INVYR"
    ]

    tree_usevars = [
        "DIA",
        "PLT_CN",
        "CONDID",
        "HT",
        "TOTAGE",
        "SITREE",
        "TPA_UNADJ",
        "CARBON_AG",
        "STATUSCD",
        "DIACHECK",
    ]

    cond_vars = [
        "STDAGE",
        "BALIVE",
        "SICOND",
        "SISP",
        "FORTYPCD",
        "DSTRBCD1",
        "DSTRBYR1",
        "CONDPROP_UNADJ",
    ]

    plot_df = plot_df.loc[:, plot_usevars]
    tree_df = tree_df.loc[:, tree_usevars]
    cond_df = cond_df.loc[:, ["PLT_CN", "CONDID"] + cond_vars]

    if len(plot_df) > 0:
        print(f'doing aggregation for {state}')
        cond_agg = cond_df.groupby(["PLT_CN", "CONDID"])[cond_vars].max()
        cond_agg = cond_agg.rename(
            columns={"BALIVE": "BALIVE_COND", "STDAGE": "STDAGE_COND"}
        )
        tree_agg = tree_df.groupby(["PLT_CN", "CONDID"]).apply(tree_stats)

        plt_uids = generate_uids(plot_df)

        agg = merge(
            cond_agg, tree_agg, left_index=True, right_index=True
        ).reset_index(level=1)
        agg = agg.join(plot_df.set_index("CN")[["LAT", "LON", "ELEV", "INVYR"]],)
        agg["plt_uid"] = agg.index.map(plt_uids)
        agg.to_parquet(
            f"gs://carbonplan-scratch/fia/states/agg_{state}.parquet",
            compression="gzip",
            engine="fastparquet",
        )
    return  # not sure if needed. Dask documentation tends to include
