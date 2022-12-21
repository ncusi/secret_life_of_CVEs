#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <cleaned_df> <languages_to_class.json> <cve_lifespan_df>

Requires result of clean_data_before_cve_lifespan_calculation.py and prepare_language_to_class_dict.py
"""
import json
import sys

import numpy as np
import pandas as pd


def main():
    cleaned_df_filename = sys.argv[1]
    language_to_class_dict_filename = sys.argv[2]
    cve_lifespan_df_filename = sys.argv[3]
    language_to_class = read_language_to_class_dict(language_to_class_dict_filename)
    language_columns = language_to_class['language_columns']

    cleaned_df = pd.read_parquet(cleaned_df_filename)
    lifespan_df = calculate_lifespan(cleaned_df)


def read_language_to_class_dict(language_to_class_dict_filename):
    with open(language_to_class_dict_filename, 'r') as f:
        language_to_class = json.load(f)
    return language_to_class


def calculate_lifespan(df):
    gb = df.groupby(by=['commit_cves', 'published_date'])
    aggregated_df = gb.agg(min_commiter_time=('commiter_time', np.min), max_commiter_time=('commiter_time', np.max),
                           min_author_time=('author_time', np.min), max_author_time=('author_time', np.max))
    aggregated_df = aggregated_df.reset_index()
    aggregated_df['cve_birth_commiter_time'] = aggregated_df[['published_date', 'min_commiter_time']].min(axis=1)
    aggregated_df['cve_birth_author_time'] = aggregated_df[['published_date', 'min_author_time']].min(axis=1)
    birth_df = aggregated_df[['commit_cves', 'cve_birth_commiter_time', 'cve_birth_author_time']]
    commits_with_cve_birth = pd.merge(left=df, right=birth_df, left_on='commit_cves', right_on='commit_cves')
    commits_with_cve_birth_with_projects = commits_with_cve_birth[
        commits_with_cve_birth['project_names'].isna() == False]
    commits_with_cve_birth_one_project = commits_with_cve_birth_with_projects[
        commits_with_cve_birth_with_projects['project_names'].map(lambda x: len(x.split(";"))) == 1]
    project_names_gb = commits_with_cve_birth_one_project.groupby(['project_names', 'commit_cves'])
    project_names_gb_agg = project_names_gb.agg(cve_death_min_commiter_time=('commiter_time', np.min),
                                                cve_death_max_commiter_time=('commiter_time', np.max),
                                                cve_death_min_author_time=('author_time', np.min),
                                                cve_death_max_author_time=('author_time', np.max))
    cve_lifespan_per_project_df = project_names_gb_agg.reset_index()
    return cve_lifespan_per_project_df


if __name__ == '__main__':
    main()
