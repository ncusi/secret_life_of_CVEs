#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName)s [OPTIONS] <cve_survival_input_df> ...

Based on 'notebooks/surv_ana.ipynb' and 'notebooks/surv_clean_programming_classes.ipynb'
Jupyter Notebooks
"""
import json
import pathlib     # file and path handling
import sys
from functools import reduce

import numpy as np
import scipy.stats
import pandas as pd
from joblib import Parallel, delayed
from lifelines.utils import concordance_index
from matplotlib import pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
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


def apply_stats_for_each_value(params, df, fmap, condition_names=None, df_mask=None):
    """Apply stats to each value in column"""

    all_count = df.shape[0]

    if df_mask is not None:
        df = df[df_mask]

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
    # DEBUG
    #print(dxy_bootstrapped)
    # confidence interval
    click.echo(f"Computing confidence interval from {len(dxy_bootstrapped)} samples...", file=sys.stderr)
    dxy, ci_low, ci_high = mean_confidence_interval(dxy_bootstrapped, confidence=params['confidence'])

    ret = {
        'Cohort': all_count,
        'Number of patients': stats['count'],
        '% of cohort': 100.0*selected_count / all_count,
        'Survival days, median': stats['median'],
        'Survival years, median': stats['median'] / 365,
        'Dxy (full)': Dxy(dff['E'], dff['Y'], dff['agg']),
        'bootstrap': {
            'Dxy': dxy,
            'Confidence interval low': ci_low,
            'Confidence interval high': ci_high,
            'confidence threshold %': 100.0*params['confidence'],
            'bootstrap samples': params['bootstrap_samples'],
        },
    }

    click.echo("Computing descriptive statistics like mean, median, etc....", file=sys.stderr)
    dff_groupby_y =dff.groupby(by=['agg'])['Y']
    groups = dff_groupby_y\
        .agg(['count', 'median', 'min',
              lambda x: np.percentile(x, q=25), lambda x: np.percentile(x, q=75),
              'max', 'mean', 'std', 'skew'])\
        .rename(columns={'<lambda_0>': '25%', '<lambda_1>': '75%'})

    # DEBUG
    #print(dff_groupby_y.describe())

    groups.index.names = [params['cve_survival_analysis']['risk_column_name']]
    if condition_names:
        groups.index = groups.index.map(condition_names)

    return ret, groups, dff


def plot_survival_function(params, plot_path, dff, condition_names=None):
    for value in dff["agg"].unique():
        mask = (dff["agg"] == value)
        time_cell, survival_prob_cell = kaplan_meier_estimator(dff["E"][mask], dff["Y"][mask])
        plt.step(time_cell, survival_prob_cell, where="post",
                 label="%s (n = %d)" % (condition_names[value], mask.sum()) \
                     if condition_names else \
                       "%d (n = %d)" % (value, mask.sum())
                 )

    plt.ylabel("est. probability of survival $\\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.legend(loc="best")
    # plt.show()
    plt.savefig(plot_path)
    plt.clf()
    pass


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


def uniquify(param):
    seen = set()
    uniq = []
    for elem in param:
        if elem not in seen:
            uniq.append(elem)
            seen.add(elem)

    return uniq


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
@click.option('--save-every-param',
              is_flag=True, type=bool, default=False,
              help='Save every parameter value (including default values) in the parameters file')
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
@click.option('--drop-if-true', '+d', multiple=True,
              help="Drop rows where this boolean column evaluates to true")
@click.option('--value', multiple=True,
              help="Provide risk column values of interest, in ranking order")
@click.option('--limit',
              type=click.IntRange(min=1),
              help="Use only first few rows for analysis [default: no limit]")
@click.option('--lifespan-max',
              type=click.FloatRange(min=0.0),
              help="Use only events where CVE lifespan in days is smaller than provided value")
# where to put output files
# NOTE: for no value to mean no output, there cannot be default value
@click.option('--eval-path',
              type=click.Path(dir_okay=True, file_okay=False, path_type=pathlib.Path),
              help='Directory where to save plots, metrics, and other output files')
# Parquet file containing dataframe with data to do CVE survival analysis on
@click.argument('input_df',
                type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
def main(params_file, save_params, save_every_param,
         confidence, bootstrap_samples,
         lifetime_column, risk_column,
         drop_if_true, value,
         limit, lifespan_max,
         eval_path,
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
    if drop_if_true:
        params['cve_survival_analysis']['drop_if_true_column_names'] = drop_if_true
        params_changed = True
    if value:
        params['cve_survival_analysis']['values_ranking'] = value
        params_changed = True
    if limit is not None:
        params['cve_survival_analysis']['limit'] = limit
        params_changed = True
    if lifespan_max is not None:
        params['cve_survival_analysis']['lifespan_max'] = lifespan_max
        params_changed = True
    if eval_path is not None:
        params['eval_path'] = eval_path
        params_changed = True

    # possibly save parameters to parameters file
    if params_changed:
        click.echo("some parameters changed with options from the command line")
    if params_changed and save_params and not save_every_param:
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
    if 'drop_if_true_column_names' not in params['cve_survival_analysis']:
        params['cve_survival_analysis']['drop_if_true_column_names'] = []

    # save all parameter values, if requested
    if save_every_param:
        click.echo(f"Saving every parameter back to '{params_file}'...", file=sys.stderr)
        with click.open_file(params_file, mode='w', atomic=True) as yaml_file:
            yaml.safe_dump(params, yaml_file, default_flow_style=False)

    # TODO?: use local variables in place of `params` hash for shorter code

    # print parameters
    click.echo("Parameters:", file=sys.stderr)
    if 'eval_path' in params:
        click.echo(f"- eval path: '{params['eval_path']}' = '{params['eval_path'].absolute()}'")
    click.echo(f"- confidence: {params['confidence']}", file=sys.stderr)
    click.echo(f"- bootstrap samples (n): {params['bootstrap_samples']}", file=sys.stderr)
    click.echo("- CVE survival analysis:", file=sys.stderr)
    click.echo(f"    - CVE lifetime column name: '{params['cve_survival_analysis']['lifetime_column_name']}'",
               file=sys.stderr)
    click.echo(f"    - risk factor column name:  '{params['cve_survival_analysis']['risk_column_name']}'",
               file=sys.stderr)
    if 'values_ranking' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['values_ranking']:
        click.echo(f"    - ranking of values ({len(params['cve_survival_analysis']['values_ranking'])} values):",
                   file=sys.stderr)
        for idx, val in enumerate(params['cve_survival_analysis']['values_ranking']):
            click.echo(f"        {idx:2d}: {val}", file=sys.stderr)
    if params['cve_survival_analysis']['drop_if_true_column_names']:
        click.echo(f"    - drop rows from dataframe if any of the following "+
                   f"{len(params['cve_survival_analysis']['drop_if_true_column_names'])} columns are true",
                   file=sys.stderr)
        for column_name in params['cve_survival_analysis']['drop_if_true_column_names']:
            click.echo(f"        - '{column_name}'", file=sys.stderr)
    if 'limit' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['limit'] is not None:
        click.echo(f"    - use only first rows: {params['cve_survival_analysis']['limit']}",
                   file=sys.stderr)
    if 'lifespan_max' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['lifespan_max'] is not None:
        click.echo(f"    - use only data with CVE lifespan"+
                   f" < {params['cve_survival_analysis']['lifespan_max']} days",
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
    if params['cve_survival_analysis']['drop_if_true_column_names']:
        columns_error = False
        for column_name in params['cve_survival_analysis']['drop_if_true_column_names']:
            if column_name not in df.columns:
                click.echo('\033[31m'+"ERROR"+'\033[39m'+": "+
                           f"No '{column_name}' column in dataframe!",
                           err=True)
                columns_error = True
            elif not pd.api.types.is_bool_dtype(df[column_name]):
                click.echo("ERROR" + ": " +
                           f"'{column_name}' column is not boolean, it is of {df.dtypes[column_name]} type",
                           err=True)
                columns_error = True
        if columns_error:
            sys.exit(2)

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

    # dropping
    df_mask = None
    if params['cve_survival_analysis']['drop_if_true_column_names']:
        df_mask = reduce(lambda col_a, col_b: col_a | col_b,
                         [df[col] for col in params['cve_survival_analysis']['drop_if_true_column_names']])
        to_drop = df_mask.sum()
        df_mask = ~df_mask
        click.echo(f"Kept {df_mask.sum()} elements, dropped {to_drop} out of {df_mask.count()} after drop-if-true",
                   file=sys.stderr)
        click.echo(f"* drop-if-true: {params['cve_survival_analysis']['drop_if_true_column_names']}",
                   file=sys.stderr)
    if 'lifespan_max' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['lifespan_max'] is not None:
        lifespan_mask = df['Y'] < params['cve_survival_analysis']['lifespan_max']
        click.echo(f"Limiting lifespan to < {params['cve_survival_analysis']['lifespan_max']} days"+
                   f" kept {lifespan_mask.sum()} out of {lifespan_mask.count()} rows", file=sys.stderr)
        if df_mask is not None:
            df_mask &= lifespan_mask
        else:
            df_mask = lifespan_mask

    # f_map and ranking names
    condition_names_hash = None
    f_map = None
    values_list = None
    # TODO: move handling integer-valued risk column first, and use `--value`s as descriptions
    if 'values_ranking' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['values_ranking']:
        values_list = uniquify(params['cve_survival_analysis']['values_ranking'])
        values_hash, condition_names_hash = values_ranking_hashes(values_list)

        f_map = lambda row: f_map_generic(row, params['cve_survival_analysis']['risk_column_name'],
                                          values_hash)

        if len(values_list) < len(params['cve_survival_analysis']['values_ranking']):
            click.echo("WARNING: some of risk values were provided more than once!", err=True)
    elif pd.api.types.is_integer_dtype(df.dtypes[params['cve_survival_analysis']['risk_column_name']]):
        f_map = lambda row: f_map_int(row, params['cve_survival_analysis']['risk_column_name'])
        click.echo(f"risk column is integer valued: {df.dtypes[params['cve_survival_analysis']['risk_column_name']]}",
                   file=sys.stderr)
    else:
        values_list = create_values_ranking_list(df[params['cve_survival_analysis']['risk_column_name']],
                                                 df.dtypes[params['cve_survival_analysis']['risk_column_name']])
        click.echo(f"risk column values, ordered = {values_list}", file=sys.stderr)
        if values_list:
            values_hash, condition_names_hash = values_ranking_hashes(values_list)
            click.echo(f"risk column values hash = {values_hash}", file=sys.stderr)
            click.echo(f"condition names hash    = {condition_names_hash}", file=sys.stderr)

            f_map = lambda row: f_map_generic(row, params['cve_survival_analysis']['risk_column_name'],
                                              values_hash)
        else:
            click.echo(f"No good ordering for '{params['cve_survival_analysis']['risk_column_name']}' column "+
                       f"of '{df.dtypes[params['cve_survival_analysis']['risk_column_name']]}' type",
                       file=sys.stderr)

    unique_values = df[params['cve_survival_analysis']['risk_column_name']].unique()
    click.echo(f"risk column values, unordered = {unique_values};",
               file=sys.stderr)
    # sanity check, with warning
    if values_list:
        unique_values_set = set(unique_values.tolist())
        for val in values_list:
            if val not in unique_values_set:
                click.echo(f"- {val} not in risk column values!", err=True)

    # do we have ranking?
    if f_map is None:
        click.echo('\033[31m'+"ERROR"+'\033[39m'+": "+
                   "No ranking provided or found for risk column",
                   err=True)
        sys.exit(3)

    if 'limit' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['limit'] is not None:
        click.echo(f"Computing stats for first {params['cve_survival_analysis']['limit']} elements,"+
                   f" for '{params['cve_survival_analysis']['risk_column_name']}' risk factor...",
                   file=sys.stderr)
        measures, groups_df, dff = \
            apply_stats_for_each_value(params, df[:params['cve_survival_analysis']['limit']],
                                       f_map, condition_names=condition_names_hash,
                                       df_mask=df_mask)
    else:
        click.echo(f"Computing stats for all {df.shape[0]} elements," +
                   f" for '{params['cve_survival_analysis']['risk_column_name']}' risk factor...",
                   file=sys.stderr)
        measures, groups_df, dff = \
            apply_stats_for_each_value(params, df,
                                       f_map, condition_names=condition_names_hash,
                                       df_mask=df_mask)
    click.echo("", file=sys.stderr)
    print(json.dumps(measures, indent=4))
    print(groups_df)
    if 'eval_path' in params:
        eval_path = params['eval_path']
        # ensure that directory exists
        eval_path.mkdir(parents=True, exist_ok=True)

        # save metrics, statistics, and plots
        click.echo(f"Saving output files to '{eval_path}/' directory...", file=sys.stderr)
        params['eval_path'] = str(params['eval_path']) # Path to string
        with eval_path.joinpath('cve_surv_params.yaml').open('w') as yaml_file:
            yaml.safe_dump(params, yaml_file, default_flow_style=False)
        with eval_path.joinpath('cve_surv_metrics.json').open('w') as json_file:
            json.dump(measures, json_file, indent=4)
        pd.DataFrame({
            'Number of patients': measures['Number of patients'],
            '% of cohort': measures['% of cohort'],
            'Dxy': measures['bootstrap']['Dxy'],
            f"Confidence interval {measures['bootstrap']['confidence threshold %']}% low":
                measures['bootstrap']['Confidence interval low'],
            f"Confidence interval {measures['bootstrap']['confidence threshold %']}% high":
                measures['bootstrap']['Confidence interval high'],
        }, index=[ params['cve_survival_analysis']['risk_column_name'] ])\
            .to_csv(eval_path / 'cve_surv_group_metrics.csv', index=True)
        groups_df.to_csv(eval_path / 'cve_surv_statistics.csv', index=True)
        plot_survival_function(params, eval_path / 'cve_survival_function.png',
                               dff, condition_names=condition_names_hash)

if __name__ == '__main__':
    # does not match signature because of @click decorations
    main()

# end of cve_surv_analysis.py
