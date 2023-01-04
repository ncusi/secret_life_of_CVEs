#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName)s [OPTIONS] <cve_survival_input_df> ...

Based on 'notebooks/surv_ana.ipynb' and 'notebooks/surv_clean_programming_classes.ipynb'
Jupyter Notebooks
"""
import pathlib     # file and path handling
import sys
#from collections import defaultdict

import numpy as np
import scipy.stats
import pandas as pd
from joblib import Parallel, delayed
from lifelines.utils import concordance_index
from tqdm import tqdm
import yaml        # for simple YAML format handling
import click       # command line parsing


def mean_confidence_interval(data, confidence=0.95):
    """Calculate mean, and lower and upper bound of confidence interval"""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def Dxy(event_observed, cve_lifetime, predicted_scores):
    """Calculate Dxy

    Rescale values from 0<=x<=1 range, where 0 is perfect anti-concordance,
    and 0.5 is the expected result from random predictions,
    to the -1<=x<=1 range
    """
    #return 2 * concordance_index_censored(event_observed, cve_lifetime, predicted_scores)[0] - 1
    return 2 * concordance_index(cve_lifetime, predicted_scores, event_observed) - 1


def bootstrap_dxy_inner(df):
    """Bootstrapped Dxy calculation, randomly sampled with replacement

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to sample to compute Dxy from concordance index.
        Assumes that first 3 columns in this dataframe are:

        1. boolean column denoting which events were observed (un-censored)
        2. column with event times, in this case CVE lifetime (time to fix)
        3. column with predicted score, assumed to be category number,
           ordered in such way that larger values predict shorter lifetime

        Other columns are not used.

    Returns
    -------
    float
        Coefficient of concordance correlation, a number between –1 and 1 that
        measures the strength and direction of the relationship between
        predicted score (risk factor) and event time (CVE survival lifetime).
    """
    e, y, x = df.columns
    sample = df.sample(n=df.shape[0], replace=True)

    # calculate Dxy from sample
    return Dxy(sample[e], sample[y], sample[x])


def bootstrap_dxy(df, n=5):
    """Boostrap and calculate Dxy, resampling `n` times"""
    # resample n times
    result = Parallel(n_jobs=-1)(delayed(bootstrap_dxy_inner)(df) for i in range(n))

    return result


def apply_stats_for_each_value(params, df, fmap, condition_names=None):
    """Apply stats to each value in column"""

    all_count = df.shape[0]

    dff = pd.DataFrame({'E': df['E'], 'Y': df['Y'], 'agg': df.apply(fmap, axis=1)}).dropna()
    selected_count = dff.shape[0]
    click.echo(f"all = {all_count}, selected = {selected_count}, uncensored = {dff['E'].sum()}",
               file=sys.stderr)

    # DEBUG
    #print(dff.head())

    stats = dff['Y'].aggregate(['count', 'median'])

    click.echo(f"Computing {params['bootstrap_samples']} × bootstrap Dxy "+
               f"for {dff.shape[0]} elements...", file=sys.stderr)
    dxy_bootstrapped = bootstrap_dxy(dff[['E', 'Y', 'agg']], params['bootstrap_samples'])
    # confidence interval
    click.echo(f"Computing confidence interval from {len(dxy_bootstrapped)} samples...", file=sys.stderr)
    dxy, ci_low, ci_high = mean_confidence_interval(dxy_bootstrapped, confidence=params['confidence'])

    ret = {
        'Number of patients': stats['count'],
        '% of cohort': np.round(100.0*selected_count / all_count, 2),
        'Survival days, median': stats['median'],
        'Survival years, median': stats['median'] / 365,
        'Dxy': np.round(dxy, 3),
        'Confidence interval 95% low': np.round(ci_low, 2),
        'Confidence interval 95% high': np.round(ci_high, 2),
    }
    result = pd.DataFrame(ret, index=(0,))

    click.echo("Computing descriptive statistics like mean, median, etc....", file=sys.stderr)
    groups = dff.groupby(by=['agg'])['Y'].aggregate(['count', 'median', 'min', 'max', 'mean', 'std'])

    if condition_names:
        groups.index = groups.index.map(condition_names)
    # groups.columns = ['Number of patiets', 'Survival days, median', 'min', 'max', 'std']

    # for value in dff["agg"].unique():
    #     mask = (dff["agg"] == value)
    #     time_cell, survival_prob_cell = kaplan_meier_estimator(dff["E"][mask], dff["Y"][mask])
    #     plt.step(time_cell, survival_prob_cell, where="post",
    #              label="%s (n = %d)" % (condition_names[value], mask.sum()))
    #
    # plt.ylabel("est. probability of survival $\hat{S}(t)$")
    # plt.xlabel("time $t$")
    # plt.legend(loc="best")
    # plt.show()
    # plt.clf()
    return result, groups


def create_values_ranking_list(column_s, column_dtype):
    # if column_dtype is ordered category, we can use the order;
    # or column_dtype is unordered category with two values, any order is good;
    # or column_dtype is some kind of integer, we can use values

    # or column_dtype == 'category'
    if isinstance(column_dtype, pd.CategoricalDtype):
        if column_dtype.ordered:
            # we can use category ordering as ranking
            return column_dtype.categories.to_list()
        if column_dtype.categories.shape[0] == 2:
            # any order is good, we can get correlation or anti-correlation
            return column_dtype.categories.to_list()

    if pd.api.types.is_string_dtype(column_dtype) \
            and column_s.nunique(dropna=True) == 2:
        return column_s.unique().tolist()

    # we can't create ranking list of values
    return None


def values_ranking_hashes(values_ranking_list):
    values_ranking_hash = { value: idx
                            for (idx, value) in enumerate(values_ranking_list) }
    rankings_condition_names = { idx: value
                                 for (idx, value) in enumerate(values_ranking_list) }

    return values_ranking_hash, rankings_condition_names


def f_map_int(row, column_name, min_value=None, max_value=None):
    value = int(row[column_name])
    if min_value is not None and value < min_value:
        return None
    if max_value is not None and value > max_value:
        return None

    return value


def f_map_generic(row, column_name, values_ranking_hash):
    value = row[column_name]
    if value in values_ranking_hash:
        return values_ranking_hash[value]

    return None


@click.command()
# parameters file (for DVC use), and whether it is auto-saved
# see e.g. https://dvc.org/doc/command-reference/params
@click.option('--params-file',
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              help='Path to YAML/JSON/TOML/Python file defining stage parameters',
              default='params.yaml', show_default=True)
@click.option('--save-params',
              is_flag=True, type=bool, default=False,
              help='Save parameters provided as options in the parameters file (see previous option)')
# TODO: --save-everything, including default values
# parameters that can also be found in parameters file
@click.option('--confidence',
              # could use `max_open=True`, but then `clamp=True` cannot be used
              type=click.FloatRange(min=0.0, max=1.0, clamp=True),
              help="Confidence level for mean confidence interval [default: 0.95]")
@click.option('--bootstrap-samples', '-n',
              type=click.IntRange(min=1),
              help="Number of bootstrap samples to compute Dxy, and its confidence intervals [default: 5]")
@click.option('--lifetime-column',
              help="Name of column with event time (with CVE lifetime) [default: 'cve_lifespan_commiter_time']")
@click.option('--risk-column',
              help="Name of column with risk factor [default: 'CVSS v3.1 Ratings']")
@click.option('--limit',
              type=click.IntRange(min=1),
              help="Use only first few rows for analysis [default: no limit]")
# Parquet file containing dataframe with data to do CVE survival analysis on
@click.argument('input_df',
                type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
def main(params_file, save_params,
         confidence, bootstrap_samples,
         lifetime_column, risk_column, limit,
         input_df):
    """CVE survival analysis script"""
    # processing options and arguments
    params = {}
    click.echo(f"Reading parameters from '{params_file}'...", file=sys.stderr)
    with click.open_file(params_file, mode='r') as yaml_file:
        all_params = yaml.safe_load(yaml_file)
        if (all_params is not None) and ("eval" in all_params):
            params = all_params["eval"]
    params_changed = False

    # autovivification
    if 'cve_survival_analysis' not in params:
        # maybe https://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
        # to avoid `'foo' in d and d['foo'] is not None` code
        params['cve_survival_analysis'] = {}

    # override values in parameters file with options
    if confidence is not None:
        params['confidence'] = confidence
        params_changed = True
    if bootstrap_samples is not None:
        params['bootstrap_samples'] = bootstrap_samples
        params_changed = True
    if lifetime_column is not None:
        params['cve_survival_analysis']['lifetime_column_name'] = lifetime_column
        params_changed = True
    if risk_column is not None:
        params['cve_survival_analysis']['risk_column_name'] = risk_column
        params_changed = True
    if limit is not None:
        params['cve_survival_analysis']['limit'] = limit
        params_changed = True

    # possibly save parameters to parameters file
    if params_changed:
        click.echo("some parameters changed with options from the command line")
    if params_changed and save_params:
        click.echo(f"Saving all parameters back to '{params_file}'...", file=sys.stderr)
        with click.open_file(params_file, mode='w', atomic=True) as yaml_file:
            yaml.safe_dump(params, yaml_file, default_flow_style=False)

    # set default values for unset parameters
    if 'confidence' not in params:
        params['confidence'] = 0.95
    if 'bootstrap_samples' not in params:
        params['bootstrap_samples'] = 5
    if 'lifetime_column_name' not in params['cve_survival_analysis']:
        params['cve_survival_analysis']['lifetime_column_name'] = 'cve_lifespan_commiter_time'
    if 'risk_column_name' not in params['cve_survival_analysis']:
        params['cve_survival_analysis']['risk_column_name'] = 'CVSS v3.1 Ratings'

    # print parameters
    click.echo("Parameters:", file=sys.stderr)
    click.echo(f"- confidence: {params['confidence']}", file=sys.stderr)
    click.echo(f"- bootstrap samples (n): {params['bootstrap_samples']}", file=sys.stderr)
    click.echo("- CVE survival analysis:", file=sys.stderr)
    click.echo(f"    - CVE lifetime column name: '{params['cve_survival_analysis']['lifetime_column_name']}'",
               file=sys.stderr)
    click.echo(f"    - risk factor column name:  '{params['cve_survival_analysis']['risk_column_name']}'",
               file=sys.stderr)
    if 'limit' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['limit'] is not None:
        click.echo(f"    - use only first rows: {params['cve_survival_analysis']['limit']}",
                   file=sys.stderr)
    click.echo("", file=sys.stderr)

    # reading input data
    tqdm.pandas()
    click.echo(f"Reading input dataframe from '{input_df}'...", file=sys.stderr, nl=False)
    df = pd.read_parquet(input_df)
    click.echo(f" {df.shape[0]} elems", file=sys.stderr, nl=False)
    df = df.drop_duplicates()
    click.echo(f", {df.shape[0]} unique", file=sys.stderr)
    # assuming everything is merged in

    # sanity checking
    if params['cve_survival_analysis']['lifetime_column_name'] not in df.columns:
        click.echo('\033[31m'+"ERROR"+'\033[39m'+": "+
                   f"No '{params['cve_survival_analysis']['lifetime_column_name']}' column in dataframe!",
                   err=True)
        sys.exit(1)
    if params['cve_survival_analysis']['risk_column_name'] not in df.columns:
        click.echo('\033[31m'+"ERROR"+'\033[39m'+": "+
                   f"No '{params['risk_survival_analysis']['lifetime_column_name']}' column in dataframe!",
                   err=True)
        sys.exit(1)

    # censoring and lifetime
    click.echo(f"Computing or extracting CVE lifetime "+
               f"('{params['cve_survival_analysis']['lifetime_column_name']}') in days...", file=sys.stderr)
    params['cve_survival_analysis']['lifetime_column_name [days]'] =\
        f"{params['cve_survival_analysis']['lifetime_column_name']} [days]"

    # censoring
    #censoring_mask = df[params['cve_survival_analysis']['risk_column_name']].isna()

    df['E'] = True
    #df.loc[censoring_mask,'E'] = False
    #click.echo(f"censored = {censoring_mask.sum()}; uncensored = {df['E'].sum()}; total = {df['E'].count()}",
    #           file=sys.stderr)

    # TODO: pass names of columns, instead of creating new columns
    if params['cve_survival_analysis']['lifetime_column_name [days]'] in df.columns:
        df['Y'] = df[params['cve_survival_analysis']['lifetime_column_name [days]']]
    else:
        df['Y'] = df[params['cve_survival_analysis']['lifetime_column_name']].dt.days

    # f_map and ranking names
    values_list = create_values_ranking_list(df[params['cve_survival_analysis']['risk_column_name']],
                                             df.dtypes[params['cve_survival_analysis']['risk_column_name']])
    click.echo(f"risk column values, ordered   = {values_list}", file=sys.stderr)
    click.echo(f"risk column values, unordered = " +
               f"{df[params['cve_survival_analysis']['risk_column_name']].unique()};",
               file=sys.stderr)
    values_hash, condition_names_hash = values_ranking_hashes(values_list)
    click.echo(f"risk column values hash = {values_hash}", file=sys.stderr)
    click.echo(f"condition names hash    = {condition_names_hash}", file=sys.stderr)

    if pd.api.types.is_integer_dtype(df.dtypes[params['cve_survival_analysis']['risk_column_name']]):
        f_map = lambda row: f_map_int(row, params['cve_survival_analysis']['risk_column_name'])
    else:
        f_map = lambda row: f_map_generic(row, params['cve_survival_analysis']['risk_column_name'],
                                          values_hash)

    if 'limit' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['limit'] is not None:
        click.echo(f"Computing stats for first {params['cve_survival_analysis']['limit']} elements,"+
                   f" for '{params['cve_survival_analysis']['risk_column_name']}' risk factor...",
                   file=sys.stderr)
        S1, S2 = apply_stats_for_each_value(params, df[:params['cve_survival_analysis']['limit']],
                                            f_map,
                                            condition_names=condition_names_hash)
    else:
        click.echo(f"Computing stats for all {df.shape[0]} elements," +
                   f" for '{params['cve_survival_analysis']['risk_column_name']}' risk factor...",
                   file=sys.stderr)
        S1, S2 = apply_stats_for_each_value(params, df,
                                            f_map,
                                            condition_names=condition_names_hash)
    click.echo("", file=sys.stderr)
    print(S1.T)
    print(S2)

if __name__ == '__main__':
    # does not match signature because of @click decorations
    main()

# end of cve_surv_analysis.py
