#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <combined_df> <languages_to_class.json> <cleaned_cve_df>

Removes each row (commit and cve) matching following criteria:
1) Commit is in not supported language
2) Commit or cve has wrong date (either too far in past or in future)
3) Commit has multiple projects assigned in c2P map
Requires result of find_programming_language_for_cve.py and prepare_language_to_class_dict.py
"""
import json
import sys

import pandas as pd


def main():
    combined_df_filename = sys.argv[1]
    language_to_class_dict_filename = sys.argv[2]
    cleaned_data_df_filename = sys.argv[3]
    language_to_class = read_language_to_class_dict(language_to_class_dict_filename)
    language_columns = language_to_class['language_columns']

    combined_df = pd.read_parquet(combined_df_filename)

    selected_languages_df = combine_additional_languages(combined_df, language_columns)
    corrected_dates_df = remove_incorrect_dates(selected_languages_df)
    corrected_projects_df = handle_more_than_one_project(corrected_dates_df)
    corrected_projects_df.drop_duplicates(inplace=True)
    corrected_projects_df.to_parquet(cleaned_data_df_filename)


def read_language_to_class_dict(language_to_class_dict_filename):
    with open(language_to_class_dict_filename, 'r') as f:
        language_to_class = json.load(f)
    return language_to_class


def combine_additional_languages(df, language_columns):
    selected_language_columns = language_columns + ['lang_Shell']
    all_language_columns = [lang_column for lang_column in df if lang_column.startswith('lang_')]
    other_languages_columns = [language_column for language_column in all_language_columns if
                               language_column not in selected_language_columns]
    df['other_languages'] = df[other_languages_columns].sum(axis=1)
    columns = ['commit', 'commit_cves', 'commiter_time', 'author_time',
               'project_names', 'total_number_of_files',
               'published_date', 'error', 'other_languages'
               ] + selected_language_columns
    selected_languages_df = df[columns]
    selected_languages_df = selected_languages_df.dropna(subset=columns, how='all')
    return selected_languages_df


def remove_incorrect_dates(df):
    corrected_dates_df = df
    corrected_dates_df = corrected_dates_df[corrected_dates_df['commiter_time'] < '2023-01-01']
    corrected_dates_df = corrected_dates_df[corrected_dates_df['commiter_time'] > '1999-01-01']
    corrected_dates_df = corrected_dates_df[corrected_dates_df['author_time'] < '2023-01-01']
    corrected_dates_df = corrected_dates_df[corrected_dates_df['author_time'] > '1999-01-01']
    corrected_dates_df = corrected_dates_df[corrected_dates_df['error'].isnull()]
    corrected_dates_df = corrected_dates_df[corrected_dates_df['published_date'] < '2023-01-01']
    corrected_dates_df = corrected_dates_df[corrected_dates_df['published_date'] > '1999-01-01']
    return corrected_dates_df


def handle_more_than_one_project(df):
    non_empty_projects_df = df[df['project_names'].isna() == False]
    df_commits_with_one_project = non_empty_projects_df[
        non_empty_projects_df['project_names'].map(lambda x: len(x.split(";"))) == 1]

    df_g = df_commits_with_one_project.groupby(['project_names']).agg({'project_names': 'count'})
    project_idx = df_g['project_names'].sort_index().to_dict()

    def max_project(projects):
        project_list = projects.split(';')
        max_p = project_list[0]
        max_v = 0

        for x in project_list:
            if x in project_idx and project_idx[x] > max_v:
                max_p = x
                max_v = project_idx[x]
        return max_p

    df_multiple_projects = non_empty_projects_df[
        non_empty_projects_df['project_names'].map(lambda x: len(x.split(";"))) > 1]
    df_multiple_projects['project_names'] = df_multiple_projects['project_names'].map(max_project)

    one_project_per_each_cve_df = pd.concat([df_commits_with_one_project, df_multiple_projects])
    return one_project_per_each_cve_df


if __name__ == '__main__':
    main()
