# flake8: noqa
import math

import networkx as nx
import numpy as np
import pandas as pd


def fia(states, save=True):
    if type(states) is str:
        if states == 'all':
            df = pd.read_csv('gs://carbonplan-data/raw/fia/REF_RESEARCH_STATION.csv')
            return [preprocess_state(state, save=save) for state in df['STATE_ABBR']]
        else:
            return preprocess_state(states, save=save)
    else:
        return [preprocess_state(state, save=save) for state in states]


def tree_based_mortality(tree_df):
    """
    Roll tree-based mortality codes up to condition level.
    TODO: raw agent codes contain per-state info that we might want to use
    """
    tree_df['bulk_agentcd'] = tree_df['AGENTCD'] // 10
    per_agent_mort = (
        tree_df[~np.isnan(tree_df['bulk_agentcd'])]
        .groupby(['PLT_CN', 'CONDID', 'bulk_agentcd'])
        .TPAMORT_UNADJ.sum()
    )
    total_mort = tree_df.groupby(['PLT_CN', 'CONDID']).TPAMORT_UNADJ.sum()

    per_agent_mort = per_agent_mort.reset_index(level=2)

    all_mort = per_agent_mort.join(total_mort, lsuffix='_agent', rsuffix='_total')
    all_mort['fraction'] = all_mort['TPAMORT_UNADJ_agent'] / all_mort['TPAMORT_UNADJ_total']

    bulk_agent_map = {
        1: 'fraction_insect',
        2: 'fraction_disease',
        3: 'fraction_fire',
        8: 'fraction_human',
    }
    fraction_dist = all_mort[['bulk_agentcd', 'fraction']].pivot(columns='bulk_agentcd')['fraction']
    fraction_dist = fraction_dist.rename(columns=bulk_agent_map)[bulk_agent_map.values()]
    return fraction_dist.replace(np.nan, 0).round(2)


def preprocess_state(state_abbr, save=True):
    state_abbr = state_abbr.lower()
    tree_df = pd.read_parquet(
        f'gs://carbonplan-data/raw/fia-states/tree_{state_abbr}.parquet',
        columns=[
            'CN',
            'PLT_CN',
            'DIA',
            'STATUSCD',
            'CONDID',
            'TPA_UNADJ',
            'AGENTCD',
            'TPAMORT_UNADJ',
            'DIACHECK',
            'CARBON_AG',
            'CARBON_BG',
        ],
    )
    # calculate tree-level statistics that will sum later.
    tree_df['unadj_basal_area'] = math.pi * (tree_df['DIA'] / (2 * 12)) ** 2 * tree_df['TPA_UNADJ']
    tree_df['unadj_mort_basal_area'] = tree_df['unadj_basal_area'] * (
        tree_df['TPAMORT_UNADJ'] > 0
    )  # evals 1 if TPAMORT > 0, 0 otherwise

    # 892.179 converts lbs/acre to t/ha
    tree_df['unadj_ag_biomass'] = (
        tree_df['CARBON_AG'] * tree_df['TPA_UNADJ'] * 2 / 892.1791216197013
    )
    tree_df['unadj_bg_biomass'] = (
        tree_df['CARBON_BG'] * tree_df['TPA_UNADJ'] * 2 / 892.1791216197013
    )

    plot_df = pd.read_parquet(f'gs://carbonplan-data/raw/fia-states/plot_{state_abbr}.parquet')
    cond_df = pd.read_parquet(f'gs://carbonplan-data/raw/fia-states/cond_{state_abbr}.parquet')

    cond_vars = [
        'STDAGE',
        'BALIVE',
        'SICOND',
        'SISP',
        'OWNCD',
        'FORTYPCD',
        'FLDTYPCD',
        'DSTRBCD1',
        'DSTRBCD2',
        'DSTRBCD3',
        'TRTCD1',
        'TRTCD2',
        'TRTCD3',
        'CONDPROP_UNADJ',
        'COND_STATUS_CD',
        'SLOPE',
        'ASPECT',
        'INVYR',
        'CN',
    ]

    cond_agg = cond_df.groupby(['PLT_CN', 'CONDID'])[cond_vars].max()
    cond_agg = cond_agg.join(plot_df.set_index('CN')[['LAT', 'LON', 'ELEV']], on='PLT_CN')

    def dstrbcd_to_disturb_class(dstrbcd):
        """
        Transforms dstrbcd (int 0-90) to bulk disturbance class (bugs, fires, weather, etc)
        """
        disturb_class_map = {
            10: 'insect',
            12: 'insect',
            20: 'insect',
            22: 'insect',
            30: 'fire',
            32: 'fire',
            50: 'weather',
            51: 'weather',
            52: 'weather',
            53: 'weather',
            54: 'drought',
            80: 'human',
        }
        return (dstrbcd).map(disturb_class_map)

    dstrb_hot_encodings = [
        pd.get_dummies(dstrbcd_to_disturb_class(cond_agg[k]), prefix='disturb')
        for k in ['DSTRBCD1', 'DSTRBCD2', 'DSTRBCD3']
    ]
    # sum all disturb codes, then cast to bool so we know if 0/1 disturbance type occurred
    # https://stackoverflow.com/questions/13078751/combine-duplicated-columns-within-a-dataframe
    disturb_flags = (
        (pd.concat(dstrb_hot_encodings, axis=1)).groupby(level=0, axis=1).sum().astype(bool)
    )

    def trtcd_to_treatment_class(trtcd):
        """
        Transforms trtcd (int 00-30) to bulk treatment class (bugs, fires, weather, etc)
        """
        treatment_class_map = {
            10: 'cutting',
            20: 'preparation',
            30: 'regeneration',
            40: 'regeneration',
            50: 'other',
        }
        return (trtcd).map(treatment_class_map)

    trt_hot_encodings = [
        pd.get_dummies(trtcd_to_treatment_class(cond_agg[k]), prefix='treatment')
        for k in ['TRTCD1', 'TRTCD2', 'TRTCD3']
    ]
    # sum all treatment codes, then cast to bool so we know if 0/1 treatment type occurred
    # https://stackoverflow.com/questions/13078751/combine-duplicated-columns-within-a-dataframe
    treatment_flags = (
        (pd.concat(trt_hot_encodings, axis=1)).groupby(level=0, axis=1).sum().astype(bool)
    )

    # per-tree variables that need to sum per condition
    alive_vars = ['unadj_ag_biomass', 'unadj_bg_biomass', 'unadj_basal_area']
    condition_alive_stats = (
        tree_df.loc[tree_df['STATUSCD'] == 1].groupby(['PLT_CN', 'CONDID'])[alive_vars].sum()
    )

    condition_mortality = tree_df.groupby(['PLT_CN', 'CONDID'])['unadj_mort_basal_area'].sum()

    tree_mortality_fractions = tree_based_mortality(tree_df)

    full = cond_agg.join(condition_alive_stats)
    full = full.join(disturb_flags)
    full = full.join(treatment_flags)
    full = full.join(condition_mortality)
    full = full.join(tree_mortality_fractions)
    full = full.reset_index()

    full.loc[:, 'adj_mort'] = full.unadj_mort_basal_area / full.CONDPROP_UNADJ
    full.loc[:, 'adj_balive'] = full.unadj_basal_area / full.CONDPROP_UNADJ
    full.loc[:, 'adj_bg_biomass'] = full.unadj_bg_biomass / full.CONDPROP_UNADJ
    full.loc[:, 'adj_ag_biomass'] = full.unadj_ag_biomass / full.CONDPROP_UNADJ

    plt_uids = generate_uids(plot_df)
    full['plt_uid'] = full['PLT_CN'].map(plt_uids)

    if save:
        full.to_parquet(
            f'gs://carbonplan-data/processed/fia-states/long/{state_abbr}.parquet',
            compression='gzip',
            engine='fastparquet',
        )
    return full


def generate_uids(data, prev_cn_var='PREV_PLT_CN'):
    """
    Generate dict mapping ever CN to a unique group, allows tracking single plot/tree through time
    Can change `prev_cn_var` to apply to tree (etc)
    """
    g = nx.Graph()
    for _, row in data[['CN', prev_cn_var]].iterrows():
        if ~np.isnan(row[prev_cn_var]):  # has ancestor, add nodes + edge
            g.add_edge(row.CN, row[prev_cn_var])
        else:
            g.add_node(row.CN)  # no ancestor, potentially not resampled

    uids = {}
    for uid, subgraph in enumerate(nx.connected_components(g)):
        for node in subgraph:
            uids[node] = uid
    return uids
