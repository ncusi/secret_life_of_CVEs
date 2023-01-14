#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <unique_cve_info-cvss_cwe_etc.parquet> <cve_lifespan_language_df> <cve_lifespan_df> <cve_survival_input_df> <cve_survival_input_most_used_language_df>

Requires result of add_cvss_ranking.py and calculate_cve_lifespan_per_project.py
"""
import sys

import pandas as pd


def merge_dfs(unique_cve_info_cvss_cwe_etc_df, cve_lifespan_language_df_df):
    commits_df = cve_lifespan_language_df_df.drop_duplicates()
    commits_df['cve'] = commits_df['commit_cves']
    merged_df = commits_df.merge(unique_cve_info_cvss_cwe_etc_df.drop('error', axis='columns'), on='cve')
    return merged_df


def main():
    unique_cve_info_cvss_cwe_etc_df_filename = sys.argv[1]
    cve_lifespan_language_df_filename = sys.argv[2]
    cve_lifespan_df_filename = sys.argv[3]
    cve_survival_input_df_filename = sys.argv[4]
    cve_survival_input_most_used_language_df_filename = sys.argv[5]

    unique_cve_info_cvss_cwe_etc_df = pd.read_parquet(unique_cve_info_cvss_cwe_etc_df_filename)
    cve_lifespan_language_df = pd.read_parquet(cve_lifespan_language_df_filename)
    cve_lifespan_df = pd.read_parquet(cve_lifespan_df_filename)

    cve_survival_input_df = merge_dfs(unique_cve_info_cvss_cwe_etc_df, cve_lifespan_language_df)
    cve_survival_input_most_used_language_df = merge_dfs(unique_cve_info_cvss_cwe_etc_df, cve_lifespan_df)

    cve_survival_input_df.to_parquet(cve_survival_input_df_filename)
    cve_survival_input_most_used_language_df.to_parquet(cve_survival_input_most_used_language_df_filename)

if __name__ == '__main__':
    main()
