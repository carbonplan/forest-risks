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


def get_mort_removal_df(tree_df):
    """Getting mortality requires a few extra steps -- we just generate a copy of the data here
    and impute as needed!
    """
    mort_df = tree_df.copy()
    mort_df = mort_df.reset_index(drop=True)

    # 10 is minimum code - 0s are legacy and we assume anything < 10 is legacy as well.
    mort_df = mort_df[(mort_df['AGENTCD'] >= 10) & (mort_df['AGENTCD'] < 90)]

    # fill in missing TPA_UNADJ with TPAGROW_UNADJ - this is totally off list but JS approved
    mort_df.loc[np.isnan(mort_df['TPA_UNADJ']), 'TPA_UNADJ'] = mort_df['TPAGROW_UNADJ']
    mort_df['unadj_basal_area'] = math.pi * (mort_df['DIA'] / (2 * 12)) ** 2 * mort_df['TPA_UNADJ']

    # drop trees where TPA_UNADJ == 0 -- we have no way of using these data
    # This is rare, but I interpret this to mean that tree cannot be reliably scaled to acre-1 measurement -- for whatever reason
    mort_df = mort_df[mort_df['TPA_UNADJ'] > 0]

    # smaller-ish trees with agent code somtimes lack a DIA, but have a DIACALC -- back-fill. JS approved!
    mort_df.loc[np.isnan(mort_df['DIA']), 'unadj_basal_area'] = (
        math.pi
        * (mort_df[np.isnan(mort_df['DIA'])]['DIACALC'] / (2 * 12)) ** 2
        * mort_df[np.isnan(mort_df['DIA'])]['TPA_UNADJ']
    )

    # TODO: There is another pot of trees we can access if we impute old diameters
    # if a tree is dead (['STATUSCD'] == 2)
    # it has no DIA or DIACAKCA (np.isnan(tree_df['DIA']) & (np.isnan(tree_df['DIACALC'])
    # it has a PREV_TRE_CN (~np.isnan(tree_df['PREV_TRE_CN']))
    # The current record either has a TPA_UNADJ or TPAGROW_UNADJ ((tree_df['TPA_UNADJ'] > 0 ) | (tree_df['TPAGROW_UNADJ'] > 0))
    # These would allow recovery a handful of conditions, primarily in region 8 (i think)

    # do not return records without unadj_basal_area -- see above TODO.
    return mort_df[mort_df['unadj_basal_area'] > 0]


def preprocess_state(state_abbr, save=True):
    state_abbr = state_abbr.lower()
    tree_df = pd.read_parquet(
        f'gs://carbonplan-data/raw/fia-states/tree_{state_abbr}.parquet',
        columns=[
            'CN',
            'PLT_CN',
            'DIA',
            'DIACALC',
            'HT',
            'ACTUALHT',
            'STATUSCD',
            'CONDID',
            'TPA_UNADJ',
            'AGENTCD',
            'TPAGROW_UNADJ',
            'TPAMORT_UNADJ',
            'TPAREMV_UNADJ',
            'DIACHECK',
            'CARBON_AG',
            'CARBON_BG',
        ],
    )
    # calculate tree-level statistics that will sum later.
    tree_df['unadj_basal_area'] = math.pi * (tree_df['DIA'] / (2 * 12)) ** 2 * tree_df['TPA_UNADJ']

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
        'SITECLCD',
        'PHYSCLCD',
        'ALSTK',
        'ALSTKCD',
        'GSSTK',
        'GSSTKCD',
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
        'COND_NONSAMPLE_REASN_CD',
        'SLOPE',
        'ASPECT',
        'INVYR',
        'CN',
    ]

    cond_agg = cond_df.groupby(['PLT_CN', 'CONDID'])[cond_vars].max()
    cond_agg = cond_agg.join(
        plot_df[plot_df['PLOT_STATUS_CD'] != 2].set_index('CN')[
            ['LAT', 'LON', 'ELEV', 'KINDCD', 'MEASYEAR', 'REMPER', 'RDDISTCD', 'ECOSUBCD']
        ],
        on='PLT_CN',
    )

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
    alive_sum_vars = ['unadj_ag_biomass', 'unadj_bg_biomass', 'unadj_basal_area']
    condition_alive_sums = (
        tree_df.loc[tree_df['STATUSCD'] == 1].groupby(['PLT_CN', 'CONDID'])[alive_sum_vars].sum()
    )

    alive_mean_vars = ['HT', 'ACTUALHT']
    condition_alive_means = (
        tree_df.loc[tree_df['STATUSCD'] == 1].groupby(['PLT_CN', 'CONDID'])[alive_mean_vars].mean()
    )

    mort_removal_trees = get_mort_removal_df(tree_df)

    # define queries -- we then subset mort_removal_trees and aggregate separately to prevent zeros from sneaking in
    mort_removal_queries = {
        'unadj_full_mort': (mort_removal_trees['AGENTCD'] < 80),
        'unadj_pop_mort': (mort_removal_trees['TPAMORT_UNADJ'] > 0)
        & (mort_removal_trees['AGENTCD'] >= 10)
        & (mort_removal_trees['AGENTCD'] < 80),
        'unadj_removal': (mort_removal_trees['AGENTCD'] == 80)
        # & (mort_removal_trees['TPAREMV_UNADJ'] > 0),
    }

    condition_mort_removal = pd.concat(
        [
            mort_removal_trees[idx]
            .groupby(['PLT_CN', 'CONDID'])['unadj_basal_area']
            .sum()
            .rename(k)
            for k, idx in mort_removal_queries.items()
        ],
        axis=1,
    )

    # rerun aggregation with AGENTCD to get fraction mortality on pop estimates
    pop_mort_trees = mort_removal_trees[mort_removal_queries['unadj_pop_mort']]
    pop_mort_by_agent = (
        pop_mort_trees.groupby(['PLT_CN', 'CONDID', pop_mort_trees['AGENTCD'] // 10])[
            'unadj_basal_area'
        ]
        .sum()
        .unstack(2)  # agents to col
    )

    BULK_AGENT_MAP = {
        1: 'frac_pop_mort_insect',
        2: 'frac_pop_mort_disease',
        3: 'frac_pop_mort_fire',
        4: 'frac_pop_mort_animal',
        5: 'frac_pop_mort_weather',
        6: 'frac_pop_mort_vegetation',
        7: 'frac_pop_mort_unknown',
    }

    # convert to fraction
    fraction_pop_mort = (
        pop_mort_by_agent.div(
            pop_mort_by_agent.sum(axis=1), axis=0
        )  # invokes numpy broadcasting along each row.
        .fillna(0)
        .round(3)
        .rename(columns=BULK_AGENT_MAP)
    )

    full = cond_agg.join(condition_alive_sums)
    full = full.join(condition_alive_means)
    full = full.join(disturb_flags)
    full = full.join(treatment_flags)
    full = full.join(condition_mort_removal)
    full = full.join(fraction_pop_mort)
    full = full.reset_index()

    full.loc[:, 'adj_full_mort'] = full.unadj_full_mort / full.CONDPROP_UNADJ
    full.loc[:, 'adj_pop_mort'] = full.unadj_pop_mort / full.CONDPROP_UNADJ

    full.loc[:, 'adj_removal'] = full.unadj_removal / full.CONDPROP_UNADJ
    full.loc[:, 'adj_balive'] = full.unadj_basal_area / full.CONDPROP_UNADJ
    full.loc[:, 'adj_bg_biomass'] = full.unadj_bg_biomass / full.CONDPROP_UNADJ
    full.loc[:, 'adj_ag_biomass'] = full.unadj_ag_biomass / full.CONDPROP_UNADJ

    full['adj_sapling_mort'] = (
        full['adj_full_mort'] - full['adj_pop_mort']
    )  # diff out mort due to saplings

    plt_uids = generate_uids(plot_df)
    full['plt_uid'] = full['PLT_CN'].map(plt_uids)

    if save:
        full.to_parquet(
            f'gs://carbonplan-data/processed/fia-states/long/{state_abbr}.parquet',
            compression='gzip',
            engine='fastparquet',
        )
    return full
