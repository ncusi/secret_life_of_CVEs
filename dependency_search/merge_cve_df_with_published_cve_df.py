#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <cve_df> <published_cve_df> <commits_with_published_cve_df>

Requires result of cve_search_parser.py and retrieve_cve_dates.py
"""
import sys

import pandas as pd


def merge_published_cve(cve_df, published_cve_df):
    ext_columns = []
    for column in list(cve_df.columns):
        if column[0:4] == 'ext_':
            ext_columns.append(column)
    selected_columns = ['commit', 'commit_cves', 'commiter_time', 'author_time', 'project_names',
                        'total_number_of_files'] + ext_columns
    time_df = cve_df[selected_columns]
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
