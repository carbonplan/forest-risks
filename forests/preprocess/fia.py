from dask.dataframe import read_parquet

import networkx as nx
import numpy as np
import pandas as pd

def fia(states, save=True):
    if type(states) is str:
        if states == 'all':
            df = pd.read_csv('gs://carbonplan-data/raw/fia/REF_RESEARCH_STATION.csv')
            [preprocess_state(state, save=save) for state in df['STATE_ABBR']]
        else:
            preprocess_state(states, save=save)
    else:
        [preprocess_state(state, save=save) for state in states]
    return


def preprocess_state(state, save=True):
    state = state.lower()
    tree_df = read_parquet(f'gs://carbonplan-scratch/fia/states/tree_{state}.parquet').compute()
    plot_df = read_parquet(f'gs://carbonplan-scratch/fia/states/plot_{state}.parquet').compute()
    cond_df = read_parquet(f'gs://carbonplan-scratch/fia/states/cond_{state}.parquet').compute()
    
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

    # should be unique, but pandas was complaining, so group to ensure.
    cond_agg = cond_df.groupby(['PLT_CN', 'CONDID'])[cond_vars].max()
    cond_agg = cond_agg.rename(
        columns={"BALIVE": "BALIVE_COND", "STDAGE": "STDAGE_COND"}
    )
    
    tree_agg = tree_df.groupby(["PLT_CN", "CONDID"]).apply(tree_stats)

    plt_uids = generate_uids(plot_df)

    agg = pd.merge(
        cond_agg, tree_agg, left_index=True, right_index=True
    ).reset_index(level=1)
    agg = agg.join(plot_df.set_index("CN")[["LAT", "LON", "ELEV", "INVYR"]],)
    agg["plt_uid"] = agg.index.map(plt_uids)
    if save:
        agg.to_parquet(
            f"gs://carbonplan-scratch/fia/preprocessed/{state}.parquet",
            compression="gzip",
            engine="fastparquet",
        )
    return  # not sure if needed. Dask documentation tends to include

def generate_uids(data, prev_cn_var="PREV_PLT_CN"):
    """
    Generate dict mapping ever CN to a unique group, allows tracking single plot/tree through time
    Can change `prev_cn_var` to apply to tree (etc)
    """
    g = nx.Graph()
    for row in data[["CN", prev_cn_var]].itertuples():
        if ~np.isnan(row.PREV_PLT_CN):  # has ancestor, add nodes + edge
            g.add_edge(row.CN, row.PREV_PLT_CN)
        else:
            g.add_node(row.CN)  # no ancestor, potentially not resampled

    uids = {}
    for uid, subgraph in enumerate(nx.connected_components(g)):
        for node in subgraph:
            uids[node] = uid
    return uids


def tree_stats(df):
    idx = (df["STATUSCD"] == 1) & (df["DIA"] > 1.0) & (df["DIACHECK"] == 0)
    out = pd.Series(
        np.full(6, np.nan), index=["DIA", "HT", "TOTAGE", "SITREE", "BIOMASS", "BALIVE"]
    )
    if sum(idx) > 0:
        out["DIA"] = np.nanmean(df["DIA"][idx])
        out["HT"] = np.nanmean(df["HT"][idx])
        out["TOTAGE"] = np.nanmean(df["TOTAGE"][idx])
        out["SITREE"] = np.nanmean(df["SITREE"][idx])
        out["BIOMASS"] = np.nansum((df["CARBON_AG"][idx] * 2) * df["TPA_UNADJ"][idx])
        out["BALIVE"] = np.nansum(
            ((df["DIA"][idx] ** 2) * 0.005454) * df["TPA_UNADJ"][idx]
        )
    return out
