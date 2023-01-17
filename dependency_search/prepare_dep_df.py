#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <cve_df> <dep_df>

Requires result of cve_search_parser.py
"""
import sys

import pandas as pd


def prepare_dep_df(cve_df):
    dep_df = cve_df[['commit',
                     'dep_requirements',
                     'dep_cargo',
                     'dep_go_mod',
                     'dep_ivy',
                     'dep_maven',
                     'dep_npm',
                     'dep_nuget',
                     ]]
    dep_df['used_dep_manager'] = dep_df.any(axis='columns', bool_only=True)
    return dep_df


def main():
    cve_df_filename = sys.argv[1]
    dep_df_filename = sys.argv[2]

    cve_df = pd.read_parquet(cve_df_filename)

    dep_df = prepare_dep_df(cve_df)

    dep_df.to_parquet(dep_df_filename)


if __name__ == '__main__':
    main()
