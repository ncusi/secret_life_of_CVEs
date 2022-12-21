#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <cve_df_filename> <published_cve_df_filename> <commits_with_published_cve_df_filename>

Requires result of cve_search_parser.py and retrieve_cve_dates.py
"""
import sys

import pandas as pd


def merge_published_cve(cve_df, published_cve_df):
    # find correct name of column
    time_column='commit_time'
    if time_column not in cve_df.columns:
        time_column = 'commiter_time'
    if time_column not in cve_df.columns:
        time_column = 'committer_time'
    # TODO: further error checking

    time_df = cve_df[['commit', 'commit_cves', time_column, 'project_names']]
    exploded_time_df = time_df.explode('commit_cves')
    combined_df = exploded_time_df.merge(published_cve_df, left_on='commit_cves', right_on='cve')
    return combined_df


def main():
    cve_df_filename = sys.argv[1]
    published_cve_df_filename = sys.argv[2]
    project_cve_df_filename = sys.argv[3]
    cve_df = pd.read_parquet(cve_df_filename)

    published_cve_df = pd.read_parquet(published_cve_df_filename)
    commits_with_published_cve_df = merge_published_cve(cve_df, published_cve_df)

    commits_with_published_cve_df.to_parquet(project_cve_df_filename)


if __name__ == '__main__':
    main()
