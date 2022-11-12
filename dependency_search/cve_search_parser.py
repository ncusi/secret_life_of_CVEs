#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <search.CVE_in_commit_message.lstCmt_9.out> <parquet_result_file>

"""
import sys

import pandas as pd


def main():
    cve_search_filename = sys.argv[1]
    dataframe_filename = sys.argv[2]
    column_names = ['commit', 'tree', 'parent', 'author', 'commiter', 'author_time', 'commiter_time', 'author_timezone',
                    'commiter_timezone', 'commit_message']
    df = pd.read_csv(cve_search_filename, sep=";",  encoding='latin-1', names=column_names)
    commit_messages_df = df[['commit', 'commit_message']]
    print(commit_messages_df.columns)
    print(commit_messages_df.head())
    # commit_messages_df.to_parquet(dataframe_filename)


if __name__ == '__main__':
    main()
