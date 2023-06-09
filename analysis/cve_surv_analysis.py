#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName)s [OPTIONS] <cve_survival_input_df> ...

Based on 'notebooks/surv_ana.ipynb' and 'notebooks/surv_clean_programming_classes.ipynb'
Jupyter Notebooks.

See also Jupyter notebooks in 'notebooks/risk_factors/' directory.

TODO: make use of apply_stats_for_each_value() returning results_summary
TODO: docstrings with typing, add and update
TODO: save plots and metrics into subdirectories (for run_cve_surv_analysis.py)
TODO: support for binning for categories (see Jupyter notebooks)
TODO: use 'dvc' module for reading params: `dvc.api.params_show()` - maybe
TODO: use mlflow / dvclive / dagshub API for saving metrics - as an option
TODO: plot output as set of values (for dvc)
TODO: other plots, and other ways of evaluating survival models
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
from sksurv.nonparametric import kaplan_meier_estimator
from matplotlib import pyplot as plt
from tqdm import tqdm
import yaml        # for simple YAML format handling
import click       # command line parsing


def mean_confidence_interval(data, confidence=0.95):
    """Calculate mean, and lower and upper bound of confidence interval

    Parameters
    ----------
    data : array_like
        Data to compute mean, and lower and upper bound of confidence
        interval from
    confidence : float
        Confidence interval, 0 <= confidence <= 1

    Returns
    -------
    (float, float, float)
        Tuple of mean, lower bound, upper bound of given confidence interval;
        the lower bound and upper bound limits are symmetric around mean.
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def Dxy(event_observed, cve_lifetime, predicted_scores):
    """Calculate Dxy

    Calculates Somers’ D, a measure of ordinal association between two possibly
    dependent random variables: cve_lifetime and predicted_scores, that is
    between a series of event times and a predicted score.  It also knows
    how to handle censored values (denoted via event_observed).

    Calling convention the same as for `sksurv.metrics.concordance_index_censored()`.
    Uses `lifelines.utils.concordance_index()` instead for better performance.

    Computed from the concordance index (C-index), with the values
    rescaled from 0<=x<=1 range, where 0 is perfect anti-concordance,
    and 0.5 is the expected result from random predictions,
    to the -1<=x<=1 range.

    Parameters
    ----------
    event_observed : iterable
        a length-n iterable censoring flags, 1 if observed, 0 if not
    cve_lifetime : iterable
        a length-n iterable of observed survival times
    predicted_scores : iterable
        a length-n iterable of predicted scores - these could be survival times,
        or hazards, etc.

    Returns
    -------
    float
        Values between −1 when all pairs of the variables disagree
        and 1 when all pairs of the variables agree.
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
    """Boostrap and calculate Dxy, resampling `n` times

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
    n : int, optional
        Number of samples used for bootstrap [default: 5]

    Returns
    -------
    Sequence[float]
        List of coefficient of concordance correlation (see `Dxy`) for each
        bootstrap sample.
    """
    # resample n times
    result = Parallel(n_jobs=-1)(delayed(bootstrap_dxy_inner)(df) for _ in range(n))

    return result


def apply_stats_for_each_value(params, df, fmap, condition_names=None, df_mask=None):
    """Apply stats to each value in column

    Find median lifetime for each ordinal category of the risk factor selected
    through `fmap`, and Sommers' D correlation index for risk column as
    a whole.

    Parameters
    ----------
    params : dict
        Holds various parameters that control the behavior of this function.
    df : pandas.DataFrame
        Holds data to analyze.  This function assumes that this dataframe has
        the following columns:
          - df.Y : holding event lifetime
          - df.E : holding event mask
        The risk factor is computed over the rows of this dataframe via `fmap`
        function (with `df.apply(fmap, axis=1)`).
    fmap : function
        Maps rows with risk column values to ordinal risk score.
        Function to apply to each row of `df` dataframe.
    condition_names : dict, optional
        Mapping from risk score to category name, "reversing" `fmap`.
        Changes the index of `groups` part of function result.
    df_mask : bool pandas.Series/pandas.DataFrame, optional
        Filter to apply to `df` before doing any calculations:
        `df = df[df_mask]`.

    Returns
    -------
    (dict, pandas.DataFrame, pandas.DataFrame, dict)
        First element of returned tuple is the `results` nested dict, containing
        information about the risk factor / risk column as a whole, including
        Dxy (Sommers' D measure, a correlation coefficient, see `Dxy`) and
        its confidence interval (CI) found through bootstrapping.

        Second element of returned tuple is the `groups` DataFrame, containing
        information about each category in the risk columns, including median,
        percent of cohort, etc.

        Third element of returned tuple is the `dff` DataFrame containing
        relevant parts used to compute everything else, namely:
          - 'E' column with event mask
          - 'Y' column with event lifetime
          - 'agg' column with categories mapped to ordinal numbers
            possibly with concatenated or skipped categories

        Fourth element of returned tuple is the `results_summary` flat dict,
        containing the information about risk column as a whole, to be
        used when comparing different risk factors.
    """
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

    click.echo(f"Computing {params['bootstrap_samples']} × bootstrap Dxy " +
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
    dff_groupby_y = dff.groupby(by=['agg'])['Y']
    groups = dff_groupby_y\
        .agg(['count', 'median', 'min',
              lambda x: np.percentile(x, q=25), lambda x: np.percentile(x, q=75),
              'max', 'mean', 'std', 'skew'])\
        .rename(columns={'<lambda_0>': '25%', '<lambda_1>': '75%'})
    # add columns
    groups['% of cohort'] = 100.0 * groups['count'] / all_count
    groups['% of selected'] = 100.0 * groups['count'] / selected_count
    groups['% of patients'] = 100.0 * groups['count'] / stats['count']
    groups['interquartile range'] = groups['75%'] - groups['25%']

    # DEBUG
    #print(dff_groupby_y.describe())

    groups.index.names = [ params['cve_survival_analysis']['risk_column_name'] ]
    if condition_names:
        groups.index = groups.index.map(condition_names)

    results_summary = {
        'cohort': all_count,
        'count': stats['count'],
        '% of cohort': ret['% of cohort'],
        'median': stats['median'],
        'Dxy': dxy,
        'Dxy_low': ci_low,
        'Dxy_high': ci_high,
    }

    return ret, groups, dff, results_summary


def plot_survival_function(params, plot_path, dff, condition_names=None):
    """Create plot of survival function and save it as an image

    Parameters
    ----------
    params : dict
        Uses params['cve_survival_analysis']['risk_column_name'] and
        params['description'] to create plot titles.
    plot_path : str | pathlib.Path
        Where to save the plot.  Determines image type via extension (PNG, PDF, SVG,...).
    dff : pandas.DataFrame
        The dataframe with data to compute survival function from.
    condition_names : dict | None
        Mapping from risk factor score to risk factor name.
    """
    for value in dff["agg"].unique():
        mask = (dff["agg"] == value)  # it's a boolean-valued pd.Series
        time_cell, survival_prob_cell = kaplan_meier_estimator(dff["E"][mask], dff["Y"][mask])
        plt.step(time_cell, survival_prob_cell, where="post",
                 label=f"{condition_names[value]} (n = {mask.sum():d}, median = {dff['Y'][mask].median():.0f})"
                       if condition_names else
                       f"{value:d} (n = {mask.sum():d}, median = {dff['Y'][mask].median():.0f})"
                 )

    plt.suptitle(f"Risk factor: '{params['cve_survival_analysis']['risk_column_name']}'")
    if 'description' in params:
        plt.title(params['description'])
    plt.ylabel("est. probability of survival $\\hat{S}(t)$")
    plt.xlabel("time $t$ [days]")
    plt.legend(loc="best")
    # plt.show()
    plt.savefig(plot_path)
    plt.clf()


def create_values_ranking_list(column_s, column_dtype):
    """Create ordered ranking list of values for a given column, if possible

    Parameters
    ----------
    column_s : pandas.Series
        Column of DataFrame, as Series (e.g. df['cvss'])
    column_dtype : numpy.dtype
        Data type of given column

    Returns
    -------
    list | None
        Ordered unique values, based on categories (if categorical type
        is ordered), or random order of values if there are only two values.
        Currently, it supports only categorical and string typed columns.

        Returns None if it can't create ordered list of values.
    """
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
    """Map list of values to { value: idx } and { idx: value } hashes

    Parameters
    ----------
    values_ranking_list : list
        Ordered list of values

    Returns
    -------
    (dict, dict)
        Mapping from value to index, and from index to value.

    Examples
    --------
    >>> values_ranking_hashes(['a', 'b', 'c'])
    ({'a': 0, 'b': 1, 'c': 2},
     {0: 'a', 1: 'b', 2: 'c'})
    """
    values_ranking_hash = { value: idx
                            for (idx, value) in enumerate(values_ranking_list) }
    rankings_condition_names = { idx: value
                                 for (idx, value) in enumerate(values_ranking_list) }

    return values_ranking_hash, rankings_condition_names


def f_map_int(row, column_name, min_value=None, max_value=None):
    """Extract risk score from integer valued column

    To be used to compose the `fmap` parameter of `apply_stat_for_each_value()`
    function via a partial evaluation of all but first parameter, for example:
    >>> fmap = lambda x: f_map_int(x, 'integer-valued-column')

    Parameters
    ----------
    row : pandas.Series
        DataFrame row passed, via `pandas.DataFrame.apply()`
    column_name : str
        Which column to extract, assumed to be integer valued
    min_value : int, optional
        Values smaller than min_value would be converted to None,
        (skipped) if min_value is set.
    max_value :  int, optional
        Values larger than max_value would be converted to None,
        (skipped) if max_value is set.

    Returns
    -------
    int
        Value to be used as risk score.
    """
    value = int(row[column_name])
    if min_value is not None and value < min_value:
        return None
    if max_value is not None and value > max_value:
        return None

    return value


def f_map_bool(row, column_name):
    """Extract risk score from boolean valued column

    To be used to compose the `fmap` parameter of `apply_stat_for_each_value()`
    function via a partial evaluation of all but first parameter, for example:
    >>> fmap = lambda x: f_map_bool(x, 'boolean-valued-column')

    Parameters
    ----------
    row : pandas.Series
        DataFrame row passed, via `pandas.DataFrame.apply()`
    column_name : str
        Which column to extract, assumed to be boolean valued.
        There can be only two non-N/A values, `true` and `false`,
        so any ordering is good (it would result in correlation
        or anti-correlation).

    Returns
    -------
    bool
        Value to be used as risk score.
    """
    value = bool(row[column_name])
    return value


def f_map_generic(row, column_name, values_ranking_hash):
    """Extract categorical-like column

    To be used to compose the `fmap` parameter of `apply_stat_for_each_value()`
    function via a partial evaluation of all but first parameter, for example:
    >>> fmap = lambda x: f_map_generic(x, 'categorical-like-column',
    >>>                                {'value1': 0, 'value2': 1, 'n/a': None})

    Parameters
    ----------
    row : pandas.Series
        DataFrame row passed, via `pandas.DataFrame.apply()`
    column_name : str
        Which column to extract, assumed to have limited number of unique
        values, like for example a categorical column.
    values_ranking_hash : dict
        Maps values to their risk score.

    Returns
    -------
    int
        Value to be used as risk score.
    """
    value = row[column_name]
    if value in values_ranking_hash:
        return values_ranking_hash[value]

    return None


def uniquify(param):
    """Return unique values in the sequence

    Parameters
    ----------
    param : Sequence
        List of values to iterate over

    Returns
    -------
    list
        Unique values in `params`, in order of occurrence

    Examples
    --------
    >>> uniquify([1, 3, 2, 1, 2, 2, 4])
    [1, 3, 2, 4]
    """
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
@click.option('--description', type=str,
              help='Description of this particular set of parameters (used as subtitle for plots)')
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
@click.option('--lifespan-min',
              type=click.FloatRange(min=0.0),
              help="Use only events where CVE lifespan in days is larger than provided value" +
                   " (ignored if larger than lifespan_max, if the latter is provided)")
# where to put output files
# NOTE: for no value to mean no output, there cannot be default value
@click.option('--eval-path',
              type=click.Path(dir_okay=True, file_okay=False, path_type=pathlib.Path),
              help='Directory where to save plots, metrics, and other output files; ' +
                   'note that if this option is not set, no output file is produced')
@click.option('--path-prefix',
              type=str,
              help="Prefix of every output file name, can contain relative path e.g. 'dir/' or 'dir/prefix-'")
@click.option('--append', '--append-to-group-metrics', '-a',
              is_flag=True, type=bool, default=False,
              help="Append to CSV file with group metrics 'cve_surv_statistics.csv', " +
                   "also ignore path prefix for this file")
# Parquet file containing dataframe with data to do CVE survival analysis on
@click.argument('input_df',
                type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
def main(params_file, save_params, save_every_param,
         description,
         confidence, bootstrap_samples,
         lifetime_column, risk_column,
         drop_if_true, value,
         limit, lifespan_max, lifespan_min,
         eval_path, path_prefix,
         append,
         input_df):
    """CVE survival analysis script"""
    # processing options and arguments
    all_params = {}
    params = {}
    click.echo(f"Reading parameters from '{params_file}'...", file=sys.stderr)
    with click.open_file(params_file, mode='r') as yaml_file:
        all_params = yaml.safe_load(yaml_file)
        if all_params is None:
            click.echo(f"- no documents in parameters file ", file=sys.stderr)
            all_params = {}
        else:
            click.echo(f"- read {len(all_params)} sections from first document in parameters file",
                       file=sys.stderr)
        if (all_params is not None) and ("eval" in all_params):
            params = all_params["eval"]
            click.echo(f"- read {len(params)} parameters from 'eval' section", file=sys.stderr)
    params_changed = False

    # autovivification
    if 'cve_survival_analysis' not in params:
        # maybe https://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
        # to avoid `'foo' in d and d['foo'] is not None` code
        params['cve_survival_analysis'] = {}

    # sanity check of --lifespan-min
    if lifespan_max is not None and lifespan_min is not None and lifespan_min >= lifespan_max:
        click.echo(f"Value of --lifespan-min {lifespan_min} >= {lifespan_max} --lifespan-max! Ignoring.",
                   err=True)
        lifespan_min = None

    # override values in parameters file with options
    if description is not None:
        params['description'] = description
        params_changed = True
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
    if lifespan_min is not None:
        params['cve_survival_analysis']['lifespan_min'] = lifespan_min
        params_changed = True
    if eval_path is not None:
        params['eval_path'] = str(eval_path)
        params_changed = True
    if path_prefix is not None:
        params['path_prefix'] = str(path_prefix)
        params_changed = True

    # sanity check of lifespan_min (possibly from parameters file)
    if 'lifespan_min' in params['cve_survival_analysis'] \
            and 'lifespan_max' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['lifespan_min'] >= params['cve_survival_analysis']['lifespan_max']:
        click.echo(f"Value of lifespan_min {params['cve_survival_analysis']['lifespan_min']} >= " +
                   f"{params['cve_survival_analysis']['lifespan_max']} lifespan_max! Deleting.",
                   err=True)
        del params['cve_survival_analysis']['lifespan_min']

    # possibly save parameters to parameters file
    if params_changed:
        click.echo("some parameters changed with options from the command line")
    if params_changed and save_params and not save_every_param:
        click.echo(f"Saving parameters back to '{params_file}' after changes...", file=sys.stderr)
        with click.open_file(params_file, mode='w', atomic=True) as yaml_file:
            all_params['eval'] = params
            yaml.safe_dump(all_params, yaml_file, default_flow_style=False)

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
            all_params['eval'] = params
            yaml.safe_dump(all_params, yaml_file, default_flow_style=False)

    # TODO?: use local variables in place of `params` hash for shorter code

    # convert paths to pathlib.Path
    if eval_path is not None:
        params['eval_path'] = eval_path
    elif 'eval_path' in params:
        params['eval_path'] = pathlib.Path(params['eval_path'])

    # print parameters
    click.echo("Parameters:", file=sys.stderr)
    if 'description' in params:
        click.echo(f"- description: {params['description']}", file=sys.stderr)
    if 'eval_path' in params:
        click.echo(f"- eval path: '{params['eval_path']}' = '{params['eval_path'].absolute()}'", file=sys.stderr)
    if 'path_prefix' in params:
        click.echo(f"- path_prefix: '{params['path_prefix']}'", file=sys.stderr)
    if append:
        click.echo("- appending to 'cve_surv_group_metrics.csv'", file=sys.stderr)
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
        click.echo(f"    - drop rows from dataframe if any of the following " +
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
        click.echo(f"    - use only data with CVE lifespan" +
                   f" < {params['cve_survival_analysis']['lifespan_max']} days",
                   file=sys.stderr)
    if 'lifespan_min' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['lifespan_min'] is not None:
        click.echo(f"    - use only data with CVE lifespan" +
                   f" > {params['cve_survival_analysis']['lifespan_min']} days",
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
        click.echo('\033[31m'+"ERROR"+'\033[39m'+": " +
                   f"No '{params['cve_survival_analysis']['lifetime_column_name']}' column in dataframe!",
                   err=True)
        sys.exit(1)
    if params['cve_survival_analysis']['risk_column_name'] not in df.columns:
        click.echo('\033[31m'+"ERROR"+'\033[39m'+": " +
                   f"No '{params['risk_survival_analysis']['lifetime_column_name']}' column in dataframe!",
                   err=True)
        sys.exit(1)
    if params['cve_survival_analysis']['drop_if_true_column_names']:
        columns_error = False
        for column_name in params['cve_survival_analysis']['drop_if_true_column_names']:
            if column_name not in df.columns:
                click.echo('\033[31m'+"ERROR"+'\033[39m'+": " +
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
    click.echo(f"Computing or extracting CVE lifetime " +
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
        click.echo(f"Limiting lifespan to < {params['cve_survival_analysis']['lifespan_max']} days" +
                   f" kept {lifespan_mask.sum()} out of {lifespan_mask.count()} rows", file=sys.stderr)
        if df_mask is not None:
            df_mask &= lifespan_mask
        else:
            df_mask = lifespan_mask
    if 'lifespan_min' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['lifespan_min'] is not None:
        lifespan_mask = df['Y'] > params['cve_survival_analysis']['lifespan_min']
        click.echo(f"Limiting lifespan to > {params['cve_survival_analysis']['lifespan_min']} days" +
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
    elif pd.api.types.is_bool_dtype(df.dtypes[params['cve_survival_analysis']['risk_column_name']]):
        f_map = lambda row: f_map_bool(row, params['cve_survival_analysis']['risk_column_name'])
        click.echo(f"risk column is bool valued: {df.dtypes[params['cve_survival_analysis']['risk_column_name']]}",
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
            click.echo(f"No good ordering for '{params['cve_survival_analysis']['risk_column_name']}' column " +
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
        click.echo('\033[31m'+"ERROR"+'\033[39m'+": " +
                   "No ranking provided or found for risk column",
                   err=True)
        sys.exit(3)

    if 'limit' in params['cve_survival_analysis'] \
            and params['cve_survival_analysis']['limit'] is not None:
        click.echo(f"Computing stats for first {params['cve_survival_analysis']['limit']} elements," +
                   f" for '{params['cve_survival_analysis']['risk_column_name']}' risk factor...",
                   file=sys.stderr)
        measures, groups_df, dff, _ = \
            apply_stats_for_each_value(params, df[:params['cve_survival_analysis']['limit']],
                                       f_map, condition_names=condition_names_hash,
                                       df_mask=df_mask)
    else:
        click.echo(f"Computing stats for all {df.shape[0]} elements," +
                   f" for '{params['cve_survival_analysis']['risk_column_name']}' risk factor...",
                   file=sys.stderr)
        measures, groups_df, dff, _ = \
            apply_stats_for_each_value(params, df,
                                       f_map, condition_names=condition_names_hash,
                                       df_mask=df_mask)
    click.echo("", file=sys.stderr)
    print(json.dumps(measures, indent=4))
    print(groups_df)
    if 'eval_path' in params:
        eval_path = pathlib.Path(params['eval_path'])
        path_prefix = params['path_prefix'] if 'path_prefix' in params else ""

        # ignore --path-prefix for group metrics CSV if appending
        # will be adjusted for non --append-to-group-metrics
        group_metrics_path = pathlib.Path(eval_path / 'cve_surv_group_metrics.csv')

        # split path prefix into directory part and prefix part, if exists
        if 'path_prefix' in params:
            orig_path_prefix = path_prefix
            if path_prefix.endswith('/'):
                maybe_dir = pathlib.Path(path_prefix)
                path_prefix = ""
            else:
                maybe_dir = pathlib.Path(path_prefix).parent
                path_prefix = str(pathlib.Path(path_prefix).name)
            if maybe_dir != pathlib.Path('.'):
                click.echo(f"Found '{maybe_dir}/' in '{orig_path_prefix}' path prefix; ",
                           nl=False, file=sys.stderr)
                eval_path = eval_path.joinpath(maybe_dir)
                click.echo(f"path prefix stripped to '{path_prefix}'", file=sys.stderr)

        # ensure that directory exists
        eval_path.mkdir(parents=True, exist_ok=True)

        if not append:
            # take into account --path-prefix
            group_metrics_path = pathlib.Path(eval_path / f'{path_prefix}cve_surv_group_metrics.csv')
        group_metrics_kwargs = {}

        # save metrics, statistics, and plots
        click.echo(f"Saving output files to '{eval_path}/' directory...", file=sys.stderr)
        if append:
            group_metrics_kwargs['mode'] = 'a'  # default is 'w'
            if group_metrics_path.exists():
                group_metrics_kwargs['header'] = False

        params['eval_path'] = str(params['eval_path'])  # Path to string, for saving params in YAML file
        with eval_path.joinpath(f'{path_prefix}cve_surv_params.yaml').open('w') as yaml_file:
            yaml.safe_dump(params, yaml_file, default_flow_style=False)
        with eval_path.joinpath(f'{path_prefix}cve_surv_metrics.json').open('w') as json_file:
            json.dump(measures, json_file, indent=4)
        pd.DataFrame({
            'Number of patients': measures['Number of patients'],
            '% of cohort': measures['% of cohort'],
            'Dxy': measures['bootstrap']['Dxy'],
            f"Confidence interval {measures['bootstrap']['confidence threshold %']}% low":
                measures['bootstrap']['Confidence interval low'],
            f"Confidence interval {measures['bootstrap']['confidence threshold %']}% high":
                measures['bootstrap']['Confidence interval high'],
            'Parameters file': eval_path.joinpath(f'{path_prefix}cve_surv_params.yaml'),
        }, index=[ params['cve_survival_analysis']['risk_column_name'] ])\
            .to_csv(group_metrics_path, index=True, **group_metrics_kwargs)
        groups_df.to_csv(eval_path / f'{path_prefix}cve_surv_statistics.csv', index=True)
        plot_survival_function(params, eval_path / f'{path_prefix}cve_survival_function.png',
                               dff, condition_names=condition_names_hash)


if __name__ == '__main__':
    # does not match signature because of @click decorations
    main()

# end of cve_surv_analysis.py
