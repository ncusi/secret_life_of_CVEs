#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <cleaned_cve_df> <cve_lifespan_df>

Calculates and saves dataframe with cve lifespan for each project
Requires result of clean_data_before_cve_lifespan_calculation.py
"""
import sys

import numpy as np
import pandas as pd


def main():
    cleaned_df_filename = sys.argv[1]
    cve_lifespan_df_filename = sys.argv[2]

    cleaned_df = pd.read_parquet(cleaned_df_filename)
    lifespan_df = calculate_lifespan(cleaned_df)
    cve_lifespan_df = pd.merge(left=cleaned_df, right=lifespan_df,
                               left_on=['commit_cves', 'project_names'], right_on=['commit_cves', 'project_names'])
    cve_lifespan_df.to_parquet(cve_lifespan_df_filename)


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
    cve_death_df = project_names_gb_agg.reset_index()
    cve_lifespan_per_project_df = pd.merge(left=birth_df, right=cve_death_df, left_on=['commit_cves'],
                                           right_on=['commit_cves'])
    cve_lifespan_per_project_df['cve_lifespan_commiter_time'] = pd.to_datetime(
        cve_lifespan_per_project_df['cve_death_max_commiter_time']) - pd.to_datetime(
        cve_lifespan_per_project_df['cve_birth_commiter_time'])
    cve_lifespan_per_project_df['cve_lifespan_author_time'] = pd.to_datetime(
        cve_lifespan_per_project_df['cve_death_max_author_time']) - pd.to_datetime(
        cve_lifespan_per_project_df['cve_birth_author_time'])
    return cve_lifespan_per_project_df


if __name__ == '__main__':
    main()
