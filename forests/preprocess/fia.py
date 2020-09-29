import math

import networkx as nx
import numpy as np
import pandas as pd


def fia(states, save=True):
    if type(states) is str:
        if states == "all":
            df = pd.read_csv("gs://carbonplan-data/raw/fia/REF_RESEARCH_STATION.csv")
            return [
                preprocess_state_long(state, save=save) for state in df["STATE_ABBR"]
            ]
        else:
            return preprocess_state_long(states, save=save)
    else:
        return [preprocess_state_long(state, save=save) for state in states]


def preprocess_state_long(state_abbr, save=True):
    print(f"preprocessing state {state_abbr}")
    tree_df = pd.read_parquet(
        f"gs://carbonplan-data/raw/fia-states/tree_{state_abbr}.parquet",
        columns=[
            "CN",
            "PLT_CN",
            "DIA",
            "STATUSCD",
            "CONDID",
            "TPA_UNADJ",
            "TPAMORT_UNADJ",
            "DIACHECK",
            "CARBON_AG",
            "CARBON_BG",
        ],
    )
    # calculate tree-level statistics that will sum later.
    tree_df["unadj_basal_area"] = (
        math.pi * (tree_df["DIA"] / (2 * 12)) ** 2 * tree_df["TPA_UNADJ"]
    )

    # 892.179 converts lbs/acre to t/ha
    tree_df["unadj_ag_biomass"] = (
        tree_df["CARBON_AG"] * tree_df["TPA_UNADJ"] * 2 / 892.1791216197013
    )
    tree_df["unadj_bg_biomass"] = (
        tree_df["CARBON_BG"] * tree_df["TPA_UNADJ"] * 2 / 892.1791216197013
    )

    plot_df = pd.read_parquet(
        f"gs://carbonplan-data/raw/fia-states/plot_{state_abbr}.parquet"
    )
    cond_df = pd.read_parquet(
        f"gs://carbonplan-data/raw/fia-states/cond_{state_abbr}.parquet"
    )

    cond_vars = [
        "STDAGE",
        "BALIVE",
        "SICOND",
        "SISP",
        "OWNCD",
        "FORTYPCD",
        "FLDTYPCD",
        "DSTRBCD1",
        "DSTRBYR1",
        "TRTCD1",
        "CONDPROP_UNADJ",
        "COND_STATUS_CD",
        "SLOPE",
        "ASPECT",
        "INVYR",
        "CN",
    ]

    cond_agg = cond_df.groupby(["PLT_CN", "CONDID"])[cond_vars].max()
    cond_agg = cond_agg.join(
        plot_df.set_index("CN")[["LAT", "LON", "ELEV"]], on="PLT_CN"
    )

    # per-tree variables that need to sum per condition
    alive_vars = ["unadj_ag_biomass", "unadj_bg_biomass", "unadj_basal_area"]
    condition_alive_stats = (
        tree_df.loc[tree_df["STATUSCD"] == 1]
        .groupby(["PLT_CN", "CONDID"])[alive_vars]
        .sum()
    )

    condition_alive_stats = condition_alive_stats.rename(
        columns={"unadj_basal_area": "balive"}
    )

    condition_mortality = (
        tree_df.loc[tree_df["TPAMORT_UNADJ"] > 0]
        .groupby(["PLT_CN", "CONDID"])["unadj_basal_area"]
        .sum()
    )

    condition_mortality = condition_mortality.to_frame().rename(
        columns={"unadj_basal_area": "bamort"}
    )

    full = cond_agg.join(condition_alive_stats)
    full = full.join(condition_mortality)
    full = full.reset_index()

    full.loc[:, "adj_mort"] = full.bamort / full.CONDPROP_UNADJ
    full.loc[:, "adj_balive"] = full.balive / full.CONDPROP_UNADJ
    full.loc[:, "adj_bg_biomass"] = full.unadj_bg_biomass / full.CONDPROP_UNADJ
    full.loc[:, "adj_ag_biomass"] = full.unadj_ag_biomass / full.CONDPROP_UNADJ

    plt_uids = generate_uids(plot_df)
    full["plt_uid"] = full["PLT_CN"].map(plt_uids)

    if save:
        full.to_parquet(
            f"gs://carbonplan-data/processed/fia-states/long/{state_abbr}.parquet",
            compression="gzip",
            engine="fastparquet",
        )
    return full


def to_wide(state_abbr):
    state_long = pd.read_parquet(
        f"gs://carbonplan-data/processed/fia-states/long/{state_abbr}.parquet"
    )
    # sort by plt_uid-cond pairs by INVYR, so can give idx by cumcount. 
    state_long = state_long.sort_values(["plt_uid", "CONDID", "INVYR"])
    state_long["wide_idx"] = state_long.groupby(["plt_uid", "CONDID"]).cumcount()

    tmp = []
    for var in ["INVYR", "adj_balive", "adj_mort"]:
        state_long["tmp_idx"] = var + "_" + state_long["wide_idx"].astype(str)
        tmp.append(
            state_long.pivot(index=["plt_uid", "CONDID"], columns="tmp_idx", values=var)
        )

    wide = pd.concat(tmp, axis=1)
    attrs = state_long.groupby(["plt_uid", "CONDID"])[
        ["LAT", "LON", "FORTYPCD", "FLDTYPCD", "ELEV", "SLOPE", "ASPECT"]
    ].max()
    return attrs.join(wide).dropna(subset=["INVYR_1"])  # only repeat plots go wide


def generate_uids(data, prev_cn_var="PREV_PLT_CN"):
    """
    Generate dict mapping ever CN to a unique group, allows tracking single plot/tree through time
    Can change `prev_cn_var` to apply to tree (etc)
    """
    g = nx.Graph()
    for _, row in data[["CN", prev_cn_var]].iterrows():
        if ~np.isnan(row[prev_cn_var]):  # has ancestor, add nodes + edge
            g.add_edge(row.CN, row[prev_cn_var])
        else:
            g.add_node(row.CN)  # no ancestor, potentially not resampled

    uids = {}
    for uid, subgraph in enumerate(nx.connected_components(g)):
        for node in subgraph:
            uids[node] = uid
    return uids
