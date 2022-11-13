#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <search.CVE_in_commit_message.lstCmt_9.out> <parquet_result_file>

Creates pandas dataframe saved as parquet file with commits connected to cve from results of with_CVS_in_commit_message.sh
"""
import sys

import pandas as pd


def main():
    cve_search_filename = sys.argv[1]
    dataframe_filename = sys.argv[2]
    column_names = ['commit', 'tree', 'parent', 'author', 'commiter', 'author_time', 'commiter_time', 'author_timezone',
                    'commiter_timezone', 'commit_message']
    df = pd.read_csv(cve_search_filename, sep=";", encoding='latin-1', names=column_names)
    df = find_cve(df)
    extracted_df = df[['commit', 'cve']]
    # print(extracted_df.columns)
    # print(extracted_df.head())
    extracted_df.to_parquet(dataframe_filename)


def find_cve(df):
    """
    Extracts cve number from commit message
    :param df: dataframe with commit_message column
    :return: dataframe with extracted cve from commit message
    """
    pattern = r'(CVE-\d{4}-\d{4,7})'
    df['cve'] = df['commit_message'].str.extract(pattern)
    return df


if __name__ == '__main__':
    main()
