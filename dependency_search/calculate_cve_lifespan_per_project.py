#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <cleaned_cve_df> <language_to_class.json> <cve_lifespan_language_df> <cve_lifespan_df>

Calculates and saves dataframe with cve lifespan for each project, assigns groups per each programming language class
Requires result of clean_data_before_cve_lifespan_calculation.py
"""
import json
import sys

import numpy as np
import pandas as pd


def main():
    cleaned_df_filename = sys.argv[1]
    language_to_class_dict_filename = sys.argv[2]
    cve_lifespan_language_df_filename = sys.argv[3]
    cve_lifespan_df_filename = sys.argv[3]

    language_to_class_dict = read_language_to_class_dict(language_to_class_dict_filename)

    cleaned_df = pd.read_parquet(cleaned_df_filename)
    cve_lifespans_language_df, cve_lifespans_df = prepare_cve_lifespans_with_language_classes_df(cleaned_df,
                                                                                            language_to_class_dict)
    cve_lifespans_language_df.to_parquet(cve_lifespan_language_df_filename)
    cve_lifespans_df.to_parquet(cve_lifespan_df_filename)


def prepare_cve_lifespans_with_language_classes_df(cleaned_df, language_to_class_dict):
    lifespan_df = calculate_lifespan(cleaned_df)
    lifespan_df.drop_duplicates(inplace=True)

    embargo_df = calculate_embargo(cleaned_df)
    embargo_df.drop_duplicates(inplace=True)

    projects_cve_lifespan_df = pd.merge(left=cleaned_df, right=lifespan_df,
                                        left_on=['commit_cves', 'project_names'],
                                        right_on=['commit_cves', 'project_names'])
    projects_cve_lifespan_embargo_df = pd.merge(left=projects_cve_lifespan_df, right=embargo_df,
                                                left_on=['commit_cves', 'project_names'],
                                                right_on=['commit_cves', 'project_names'])

    language_columns = language_to_class_dict['language_columns']
    selected_columns = ['commit_cves', 'project_names',
                        'cve_lifespan_commiter_time', 'cve_lifespan_author_time', 'embargo_min',
                        'embargo_max'] + language_columns
    projects_cve_lifespan_embargo_df = projects_cve_lifespan_embargo_df[selected_columns]
    projects_cve_lifespan_embargo_most_used_language_df = projects_cve_lifespan_embargo_df.copy()

    cve_lifespans_language_df = assign_programming_language_classes(projects_cve_lifespan_embargo_df,
                                                                    language_to_class_dict)

    projects_cve_lifespan_embargo_most_used_language_df['most_common_language'] = \
    projects_cve_lifespan_embargo_most_used_language_df[language_columns].idxmax(axis=1)
    projects_cve_lifespan_embargo_most_used_language_df.drop(columns=language_columns, inplace=True)
    cve_lifespans_df = assign_programming_language_class(projects_cve_lifespan_embargo_most_used_language_df, language_to_class_dict)

    return cve_lifespans_language_df, cve_lifespans_df


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


def calculate_embargo(df):
    embargo_df = df.groupby(['project_names', 'commit_cves', 'published_date']) \
        .aggregate({'commiter_time': ['min', 'max']})
    embargo_df.columns = ['_'.join(col) for col in embargo_df.columns]
    embargo_df = embargo_df.reset_index()
    embargo_df['embargo_min'] = embargo_df['published_date'] > embargo_df['commiter_time_min']
    embargo_df['embargo_max'] = embargo_df['published_date'] > embargo_df['commiter_time_max']
    embargo_df = embargo_df[['project_names', 'commit_cves', 'embargo_min', 'embargo_max']]
    return embargo_df


def assign_programming_language_classes(df, language_to_class_dict):
    language_columns = language_to_class_dict['language_columns']
    programming_paradigm = language_to_class_dict['programming_paradigm']
    compilation_class = language_to_class_dict['compilation_class']
    type_class = language_to_class_dict['type_class']
    memory_model = language_to_class_dict['memory_model']

    df.replace(0, np.nan, inplace=True)
    df[language_columns] = df[language_columns].where(pd.isna, language_columns, axis=1)
    cve_lifespans_language_df = df.melt(id_vars=['commit_cves', 'project_names',
                                                 'cve_lifespan_commiter_time', 'cve_lifespan_author_time',
                                                 'embargo_min', 'embargo_max']).dropna()
    cve_lifespans_language_df['programming_paradigm'] = cve_lifespans_language_df['value'] \
        .apply(lambda language: programming_paradigm[language])
    cve_lifespans_language_df['compilation_class'] = cve_lifespans_language_df['value'] \
        .apply(lambda language: compilation_class[language])
    cve_lifespans_language_df['type_class'] = cve_lifespans_language_df['value'] \
        .apply(lambda language: type_class[language])
    cve_lifespans_language_df['memory_model'] = cve_lifespans_language_df['value'] \
        .apply(lambda language: memory_model[language])
    return cve_lifespans_language_df


def assign_programming_language_class(df, language_to_class_dict):
    programming_paradigm = language_to_class_dict['programming_paradigm']
    compilation_class = language_to_class_dict['compilation_class']
    type_class = language_to_class_dict['type_class']
    memory_model = language_to_class_dict['memory_model']

    cve_lifespans_df = df.copy()
    cve_lifespans_df['programming_paradigm'] = cve_lifespans_df['most_common_language'] \
        .apply(lambda language: programming_paradigm[language])
    cve_lifespans_df['compilation_class'] = cve_lifespans_df['most_common_language'] \
        .apply(lambda language: compilation_class[language])
    cve_lifespans_df['type_class'] = cve_lifespans_df['most_common_language'] \
        .apply(lambda language: type_class[language])
    cve_lifespans_df['memory_model'] = cve_lifespans_df['most_common_language'] \
        .apply(lambda language: memory_model[language])
    return cve_lifespans_df


def read_language_to_class_dict(language_to_class_dict_filename):
    with open(language_to_class_dict_filename, 'r') as f:
        language_to_class = json.load(f)
    return language_to_class


if __name__ == '__main__':
    main()
