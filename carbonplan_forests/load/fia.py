import numpy as np
import pandas as pd

from .. import setup

forest_type_remap = {
    101: 101,
    102: 102,
    103: 103,
    104: 104,
    105: 105,
    121: 121,
    122: 122,
    123: 123,
    124: 124,
    125: 125,
    126: 126,
    127: 127,
    129: 124,
    141: 141,
    142: 142,
    151: 141,
    161: 161,
    162: 162,
    163: 163,
    164: 164,
    165: 163,
    166: 166,
    167: 167,
    168: 161,
    171: 171,
    182: 182,
    184: 184,
    185: 185,
    201: 201,
    202: 271,
    203: 201,
    221: 221,
    222: 221,
    224: 225,
    225: 225,
    226: 225,
    241: 221,
    261: 261,
    262: 262,
    263: 263,
    264: 264,
    265: 265,
    266: 266,
    267: 267,
    268: 268,
    269: 265,
    270: 270,
    271: 271,
    281: 281,
    301: 301,
    304: 304,
    305: 305,
    321: 321,
    341: 341,
    342: 341,
    361: 368,
    362: 368,
    363: 368,
    365: 368,
    366: 366,
    367: 367,
    368: 368,
    369: 369,
    371: 371,
    381: 381,
    383: 368,
    384: 266,
    385: 368,
    391: 368,
    401: 401,
    402: 402,
    403: 403,
    404: 404,
    405: 405,
    406: 406,
    407: 407,
    409: 409,
    501: 501,
    502: 502,
    503: 503,
    504: 504,
    505: 505,
    506: 506,
    507: 507,
    508: 508,
    509: 509,
    510: 510,
    511: 511,
    512: 512,
    513: 513,
    514: 514,
    515: 515,
    516: 516,
    517: 517,
    519: 519,
    520: 520,
    601: 601,
    602: 602,
    605: 605,
    606: 609,
    607: 607,
    608: 608,
    609: 609,
    701: 701,
    702: 702,
    703: 703,
    704: 704,
    705: 705,
    706: 706,
    707: 707,
    708: 708,
    709: 709,
    722: 701,
    801: 801,
    802: 802,
    805: 805,
    809: 809,
    901: 901,
    902: 902,
    903: 902,
    904: 904,
    905: 802,
    911: 911,
    912: 801,
    921: 221,
    922: 922,
    923: 923,
    924: 924,
    931: 933,
    933: 933,
    934: 934,
    935: 933,
    941: 941,
    942: 962,
    943: 962,
    961: 962,
    962: 962,
    971: 971,
    972: 972,
    973: 973,
    974: 974,
    975: 976,
    976: 976,
    982: 983,
    983: 983,
    984: 984,
    985: 985,
    987: 987,
    989: 989,
    992: 992,
    993: 993,
    995: 995,
    999: 999,
    280: 281,
    910: 911,
    364: 368,
    500: 503,
    800: 801,
    600: 602,
    400: 401,
    380: 368,
    700: 701,
    900: 901,
    128: 124,
    200: 201,
    360: 368,
    300: 301,
    396: 368,
    220: 221,
    260: 261,
    320: 321,
}

conus_states = [
    'AL',
    'AZ',
    'AR',
    'CA',
    'CO',
    'CT',
    'DE',
    'FL',
    'GA',
    'IA',
    'ID',
    'IL',
    'IN',
    'KS',
    'KY',
    'LA',
    'ME',
    'MA',
    'MD',
    'MI',
    'MN',
    'MO',
    'MS',
    'MT',
    'NC',
    'ND',
    'NE',
    'NH',
    'NJ',
    'NM',
    'NV',
    'NY',
    'OH',
    'OK',
    'OR',
    'PA',
    'RI',
    'SC',
    'SD',
    'TN',
    'TX',
    'UT',
    'VT',
    'VA',
    'WA',
    'WV',
    'WI',
    'WY',
]


def fia(store='az', states='conus', clean=True, group_repeats=False):
    if states == 'conus':
        states = conus_states

    load_state = fia_state_grouped if group_repeats is True else fia_state

    if type(states) is str:
        df = load_state(store, states, clean)

    if type(states) is list:
        df = pd.concat([load_state(store, state, clean) for state in states])

    if group_repeats:
        # TODO this is to drop columns added due to missing data
        # should maybe handle more cleanly
        remove = ['disturb_animal', 'disturb_fire', 'disturb_disease']
        for var in remove:
            if var in df.columns:
                df = df.drop(columns=[var])

    return df


def fia_state(store, state, clean):
    path = setup.loading(store)
    df = pd.read_parquet(
        path / f'carbonplan-data/processed/fia-states/long/{state.lower()}.parquet'
    )

    if clean:
        inds = (
            (df['adj_ag_biomass'] > 0)
            & (df['STDAGE'] < 999)
            & (df['STDAGE'] > 0)
            & (~np.isnan(df['FLDTYPCD']))
            & (df['FLDTYPCD'] != 999)
            & (df['FLDTYPCD'] != 950)
            & (df['FLDTYPCD'] <= 983)
            & (df['DSTRBCD1'] == 0)
            & (df['COND_STATUS_CD'] == 1)
            & (df['CONDPROP_UNADJ'] > 0.3)
            & (df['INVYR'] < 9999)
            & (df['INVYR'] > 2000)
        )
        df = df[inds]

    df = df.rename(
        columns={
            'LAT': 'lat',
            'LON': 'lon',
            'adj_ag_biomass': 'biomass',
            'adj_mort': 'mort',
            'STDAGE': 'age',
            'INVYR': 'year',
            'FLDTYPCD': 'type_code',
            'ELEV': 'elevation',
            'SLOPE': 'slope',
            'ASPECT': 'aspect',
            'OWNCD': 'owner',
            'PLT_CN': 'plot_id',
        }
    ).filter(
        [
            'lat',
            'lon',
            'age',
            'biomass',
            'year',
            'type_code',
            'elevation',
            'slope',
            'aspect',
            'mort',
            'owner',
        ]
    )

    df['type_code'] = df['type_code'].map(forest_type_remap)

    df['state'] = state.upper()

    return df.reset_index(drop=True)


def fia_state_grouped(store, state, clean):
    """
    Pivot long (plot-condition-invyr per row) to wide by grouping
    related plot-condition invyrs into a single row
    """
    path = setup.loading(store)
    state_long = pd.read_parquet(
        path / f'carbonplan-data/processed/fia-states/long/{state.lower()}.parquet'
    )

    state_long = state_long.rename(
        columns={
            'LAT': 'lat',
            'LON': 'lon',
            'FLDTYPCD': 'type_code',
            'INVYR': 'year',
            'STDAGE': 'age',
            'ELEV': 'elevation',
            'SLOPE': 'slope',
            'ASPECT': 'aspect',
            'OWNCD': 'owner',
            'CONDPROP_UNADJ': 'condprop',
            'mort': 'unadj_mort',
            'balive': 'unadj_balive',
            'adj_mort': 'mort',
            'adj_balive': 'balive',
        }
    )

    state_long = state_long.sort_values(['plt_uid', 'CONDID', 'year'])
    state_long['wide_idx'] = state_long.groupby(['plt_uid', 'CONDID']).cumcount()
    tmp = []
    missing_vars = []  # append missing vars, then will fill in nan cols after concat the wide df
    for var in [
        'year',
        'balive',
        'mort',
        'fraction_insect',
        'fraction_disease',
        'fraction_fire',
        'fraction_human',
        'disturb_animal',
        'disturb_insect',
        'disturb_disease',
        'disturb_fire',
        'disturb_human',
        'disturb_weather',
        'treatment_cutting',
        'treatment_regeneration',
        'treatment_preparation',
        'treatment_other',
    ]:
        state_long['tmp_idx'] = var + '_' + state_long['wide_idx'].astype(str)
        if var in state_long.columns:
            tmp.append(state_long.pivot(index=['plt_uid', 'CONDID'], columns='tmp_idx', values=var))
        else:
            missing_vars.append(var)
    wide = pd.concat(tmp, axis=1)

    if missing_vars:
        for missing_var in missing_vars:
            wide[missing_var] = np.nan

    attrs = state_long.groupby(['plt_uid', 'CONDID'])[
        ['lat', 'lon', 'age', 'type_code', 'elevation', 'slope', 'aspect', 'condprop', 'owner']
    ].max()

    if 'year_1' not in wide.columns:
        return None

    df = attrs.join(wide).dropna(subset=['year_1'])

    if clean:
        inds = (
            (~np.isnan(df['type_code']))
            & (df['type_code'] != 999)
            & (df['type_code'] != 950)
            & (df['type_code'] <= 983)
        )
        df = df[inds]

    for year in range(6):
        key = f'year_{year}'
        if key in df.columns:
            if clean:
                df = df[(df[key] < 9999) | np.isnan((df[key]))]
            if sum(np.isnan(df[key])) == len(df):
                del df[key]

    df['type_code'] = df['type_code'].map(forest_type_remap)

    df['state'] = state.upper()

    return df.reset_index(drop=True)
