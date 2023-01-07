#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName)s <unique_cve_info-cvss_cwe_etc.parquet> <unique_cve_info-cvss_cwe_ranking_etc.parquet>

Adds CVSS Ratings for the Common Vulnerability Scoring System (CVSS) score 0..10,
that should be present as 'cvss' column in the input file.

See https://nvd.nist.gov/vuln-metrics/cvss
See https://www.first.org/cvss/specification-document#Qualitative-Severity-Rating-Scale

In the future we might want to add categories based on CWE (Common Weakness Enumeration)
hierarchies, like e.g. CWE VIEW (1000): Research Concepts.  However, those hierarchies
form a graph, not a tree.  Each CWE can be included in multiple top-level CWE categories
("pillars"), which is a problem.

Based on notebooks/risk_factors/CVE_metadata.ipynb
"""
import pandas as pd
import sys
import os   # MAYBE: replace with `from pathlib import Path`

from tqdm import tqdm


def usage():
    """print script usage and exit program
    """
    # %(prog)s is argparse specifier for the program name
    print(__doc__ % {'scriptName': os.path.basename(sys.argv[0])})
    exit()


def add_cvss_rankings(df):
    """Add 'CVSS v2.0 Ratings' and 'CVSS v3.1 Ratings' columns to dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to process.  Must include 'cvss' column.

    Returns
    -------
    pandas.DataFrame

    References
    ----------
    [1] "NVD - Vulnerability Metrics"
        https://nvd.nist.gov/vuln-metrics/cvss
    [2] "CVSS 3.1: Specification Document"
        https://www.first.org/cvss/specification-document
    """
    # CVSS v2.0 Ratings
    #
    # https://nvd.nist.gov/vuln-metrics/cvss
    #
    # - Low    : 0.0-3.9
    # - Medium : 4.0-6.9
    # - High   : 7.0-10.0
    cvss2r_s = pd.cut(df['cvss'],
                      bins=[0, 4, 7, 10.1], right=False, include_lowest=True,
                      labels=['Low', 'Medium', 'High'])
    df['CVSS v2.0 Ratings'] = cvss2r_s

    # CVSS v3.1 Ratings
    #
    # https://nvd.nist.gov/vuln-metrics/cvss
    # https://www.first.org/cvss/specification-document#Qualitative-Severity-Rating-Scale
    #
    # - None      : 0.0-0.0999
    # - Low	      : 0.1-3.9
    # - Medium	  : 4.0-6.9
    # - High	  : 7.0-8.9
    # - Critical  : 9.0-10.0
    cvss3r_s = pd.cut(df['cvss'],
                      bins=[0, 0.1, 4, 7, 9, 10.1], right=False, include_lowest=True,
                      labels=['None', 'Low', 'Medium', 'High', 'Critical'])
    df['CVSS v3.1 Ratings'] = cvss3r_s

    return df


def make_columns_categorical_sorted(df):
    df['access.authentication'] = df['access.authentication'].astype(
        pd.CategoricalDtype(categories=['NONE', 'SINGLE', 'MULTIPLE'], ordered=True)
    )
    df['access.complexity'] = df['access.complexity'].astype(
        pd.CategoricalDtype(categories=['LOW', 'MEDIUM', 'HIGH'], ordered=True)
    )
    df['access.vector'] = df['access.vector'].astype(
        pd.CategoricalDtype(categories=['LOCAL', 'ADJACENT_NETWORK', 'NETWORK'], ordered=True)
    )

    impact_dtype = \
        pd.CategoricalDtype(categories=['NONE', 'PARTIAL', 'COMPLETE'], ordered=True)
    df['impact.availability'] = df['impact.availability'].astype(impact_dtype)
    df['impact.confidentiality'] = df['impact.confidentiality'].astype(impact_dtype)
    df['impact.integrity'] = df['impact.integrity'].astype(impact_dtype)

    return df


# MAYBE: move to argparse or click library for parsing command line parameters
# https://docs.python.org/3/library/argparse.html
# https://click.palletsprojects.com/
def main():
    if len(sys.argv) <= 2:
        usage()

    # progress reports for some pandas operations
    tqdm.pandas()

    in_df_filename = sys.argv[1]
    out_df_filename = sys.argv[2]

    print(f"Reading DataFrame from '{in_df_filename}'...", file=sys.stderr)
    df = pd.read_parquet(in_df_filename)

    # error checking
    if 'cvss' not in df.columns:
        print(f"error: no 'cvss' column in '{in_df_filename}'!",
              file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"Adding CVSS Rankings to DataFrame...", file=sys.stderr)
    df = add_cvss_rankings(df)
    df = make_columns_categorical_sorted(df)

    print(f"Saving DataFrame to '{out_df_filename}'...", file=sys.stderr)
    df.to_parquet(out_df_filename)


if __name__ == '__main__':
    main()
