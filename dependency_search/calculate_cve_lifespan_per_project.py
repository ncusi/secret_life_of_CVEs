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
    cve_lifespan_df_filename = sys.argv[4]

    language_to_class_dict = read_language_to_class_dict(language_to_class_dict_filename)

    cleaned_df = pd.read_parquet(cleaned_df_filename)
    cve_lifespans_language_df, cve_lifespans_df = prepare_cve_lifespans_with_language_classes_df(cleaned_df,
                                                                                                 language_to_class_dict)
    cve_lifespans_language_df.to_parquet(cve_lifespan_language_df_filename)
    cve_lifespans_df.to_parquet(cve_lifespan_df_filename)


def prepare_cve_lifespans_with_language_classes_df(cleaned_df, language_to_class_dict):
    language_columns = language_to_class_dict['language_columns']
    extended_language_columns = language_columns + ['lang_Shell', 'other_languages']

    commits_per_cve_df = calculate_commits_per_cve(cleaned_df)

    language_files_per_cve_df = calculate_language_files_per_cve(cleaned_df, extended_language_columns)

    dep_per_cve_df = calculate_dep_per_cve_df(cleaned_df)

    lifespan_df = calculate_lifespan(cleaned_df)

    embargo_df = calculate_embargo(cleaned_df)

    result_df = pd.merge(left=commits_per_cve_df, right=language_files_per_cve_df,
                         left_on=['commit_cves', 'project_names'],
                         right_on=['commit_cves', 'project_names'])
    result_df = pd.merge(left=result_df, right=dep_per_cve_df,
                         left_on=['commit_cves', 'project_names'],
                         right_on=['commit_cves', 'project_names'])
    result_df = pd.merge(left=result_df, right=lifespan_df,
                         left_on=['commit_cves', 'project_names'],
                         right_on=['commit_cves', 'project_names'])
    result_df = pd.merge(left=result_df, right=embargo_df,
                         left_on=['commit_cves', 'project_names'],
                         right_on=['commit_cves', 'project_names'])

    selected_columns = ['commits', 'commit_cves', 'project_names', 'used_dep_manager',
                        'cve_lifespan_commiter_time', 'cve_lifespan_author_time', 'embargo_min',
                        'embargo_max'] + extended_language_columns

    all_used_language_df = result_df[selected_columns]
    most_used_language_df = prepare_cve_most_used_language_df(all_used_language_df, language_columns)

    cve_lifespans_language_df = assign_programming_language_classes(all_used_language_df, language_to_class_dict)
    cve_lifespans_df = assign_programming_language_class(most_used_language_df, language_to_class_dict)

    return cve_lifespans_language_df, cve_lifespans_df


def prepare_cve_most_used_language_df(cve_all_used_language_df, language_columns):
    projects_cve_lifespan_embargo_most_used_language_df = cve_all_used_language_df.copy()
    projects_cve_lifespan_embargo_most_used_language_df['most_common_language'] = \
        projects_cve_lifespan_embargo_most_used_language_df[language_columns].idxmax(axis=1)
    projects_cve_lifespan_embargo_most_used_language_df['most_common_language_number_of_files'] = \
        projects_cve_lifespan_embargo_most_used_language_df[language_columns].max(axis=1)
    projects_cve_lifespan_embargo_most_used_language_df.drop(columns=language_columns, inplace=True)
    return projects_cve_lifespan_embargo_most_used_language_df


def calculate_dep_per_cve_df(df):
    dep_per_cve_df = df.groupby(['project_names', 'commit_cves'])[['used_dep_manager']].sum()
    dep_per_cve_df.reset_index(inplace=True)
    dep_per_cve_df.drop_duplicates(inplace=True)
    return dep_per_cve_df


def calculate_commits_per_cve(df):
    """
    Calculates number of commits per cve in selected project
    :param df: dataframe with project (project_names column), cve (commit_cves column) and commit in each row
    :return: dataframe with project, cve and number of unique commits in each row
    """
    commits_per_cve_df = df.groupby(['project_names', 'commit_cves']).agg(commits=('commit', 'nunique'))
    commits_per_cve_df.reset_index(inplace=True)
    commits_per_cve_df.drop_duplicates(inplace=True)
    return commits_per_cve_df


def calculate_language_files_per_cve(df, language_columns):
    """
    Calculates sum of files in each programming language per cve in selected project
    :param df: dataframe with project (project_names column), cve (commit_cves column) and commit in each row
    :param language_columns: list which columns contain programming languages
    :return: dataframe with project, cve and sum of files modified in each programming language in each row
    """
    language_files_per_cve_df = df.groupby(['project_names', 'commit_cves'])[language_columns].sum()
    language_files_per_cve_df.reset_index(inplace=True)
    language_files_per_cve_df.drop_duplicates(inplace=True)
    return language_files_per_cve_df


def calculate_lifespan(df):
    """
    Calculates cve lifespan
    :param df: dataframe with project (project_names column), cve (commit_cves column) and commit in each row
    :return: dataframe with project, cve and cve lifespans in each row
    """
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
    cve_lifespan_per_project_df.drop_duplicates(inplace=True)
    return cve_lifespan_per_project_df


def calculate_embargo(df):
    """
    Calculates cve embargo
    :param df: dataframe with project (project_names column), cve (commit_cves column) and commit in each row
    :return: dataframe with project, cve and cve embargo in each row
    """
    embargo_df = df.groupby(['project_names', 'commit_cves', 'published_date']) \
        .aggregate({'commiter_time': ['min', 'max']})
    embargo_df.columns = ['_'.join(col) for col in embargo_df.columns]
    embargo_df = embargo_df.reset_index()
    embargo_df['embargo_min'] = embargo_df['published_date'] > embargo_df['commiter_time_min']
    embargo_df['embargo_max'] = embargo_df['published_date'] > embargo_df['commiter_time_max']
    embargo_df = embargo_df[['project_names', 'commit_cves', 'embargo_min', 'embargo_max']]
    embargo_df.drop_duplicates(inplace=True)
    return embargo_df


def assign_programming_language_classes(df, language_to_class_dict):
    """
    Assigns programming language classes based on each of languages used in each fix
    :param df: dataframe with project (project_names column), cve (commit_cves column), and programming language
    :param language_to_class_dict: dictionaries from programming language to selected class of languages
    :return: dataframe with project, cve and programming language classes in each row
    """
    programming_paradigm = language_to_class_dict['programming_paradigm']
    compilation_class = language_to_class_dict['compilation_class']
    type_class = language_to_class_dict['type_class']
    memory_model = language_to_class_dict['memory_model']
    extended_programming_paradigm = language_to_class_dict['extended_programming_paradigm']

    cve_lifespans_language_df = df.melt(id_vars=['commit_cves', 'project_names', 'commits', 'used_dep_manager',
                                                 'cve_lifespan_commiter_time', 'cve_lifespan_author_time',
                                                 'embargo_min', 'embargo_max', 'lang_Shell',
                                                 'other_languages'])
    cve_lifespans_language_df = cve_lifespans_language_df[cve_lifespans_language_df['value'] > 0]
    cve_lifespans_language_df['programming_paradigm'] = cve_lifespans_language_df['variable'] \
        .apply(lambda language: programming_paradigm[language])
    cve_lifespans_language_df['Programming paradigm'] = pd.Categorical(
        cve_lifespans_language_df['programming_paradigm'].map({
            1: 'procedural',
            2: 'scripting',
            3: 'functional'
        }),
        categories=['procedural', 'functional', 'scripting'],
        ordered=True
    )
    cve_lifespans_language_df['compilation_class'] = cve_lifespans_language_df['variable'] \
        .apply(lambda language: compilation_class[language])
    cve_lifespans_language_df['Compilation class'] = pd.Categorical(
        cve_lifespans_language_df['compilation_class'].map({
            1: 'static', 2: 'dynamic'
        }),
        categories=['static', 'dynamic'],
        ordered=True
    )
    cve_lifespans_language_df['type_class'] = cve_lifespans_language_df['variable'] \
        .apply(lambda language: type_class[language])
    cve_lifespans_language_df['Type class'] = pd.Categorical(
        cve_lifespans_language_df['type_class'].map({
            1: 'strong', 2: 'weak'
        }),
        categories=['weak', 'strong'],
        ordered=True
    )
    cve_lifespans_language_df['memory_model'] = cve_lifespans_language_df['variable'] \
        .apply(lambda language: memory_model[language])
    cve_lifespans_language_df['Memory model'] = pd.Categorical(
        cve_lifespans_language_df['memory_model'].map({
            1: 'managed', 2: 'non managed'
        }),
        categories=['managed', 'non managed'],
        ordered=True
    )
    cve_lifespans_language_df['extended_programming_paradigm'] = cve_lifespans_language_df['variable'] \
        .apply(lambda language: extended_programming_paradigm[language])
    cve_lifespans_language_df['Programming paradigm (extended)'] = pd.Categorical(
        cve_lifespans_language_df['extended_programming_paradigm'].map({
            1: 'functional',
            2: 'object oriented',
            3: 'procedural',
            4: 'scripting'
        }),
        categories=[
            'object oriented',
            'procedural',
            'functional',
            'scripting'
        ],
        ordered=True
    )
    return cve_lifespans_language_df


def assign_programming_language_class(df, language_to_class_dict):
    """
    Assigns programming language classes based on most used language used in each fix
    :param df: dataframe with project (project_names column), cve (commit_cves column), and programming language
    :param language_to_class_dict: dictionaries from programming language to selected class of languages
    :return: dataframe with project, cve and programming language classes in each row
    """
    programming_paradigm = language_to_class_dict['programming_paradigm']
    compilation_class = language_to_class_dict['compilation_class']
    type_class = language_to_class_dict['type_class']
    memory_model = language_to_class_dict['memory_model']
    extended_programming_paradigm = language_to_class_dict['extended_programming_paradigm']

    cve_lifespans_df = df.copy().drop_duplicates()
    cve_lifespans_df = cve_lifespans_df[cve_lifespans_df['most_common_language_number_of_files'] > 0]

    cve_lifespans_df['programming_paradigm'] = cve_lifespans_df['most_common_language'] \
        .apply(lambda language: programming_paradigm[language])
    cve_lifespans_df['Programming paradigm'] = pd.Categorical(
        cve_lifespans_df['programming_paradigm'].map({
            1: 'procedural',
            2: 'scripting',
            3: 'functional'
        }),
        categories=['procedural', 'functional', 'scripting'],
        ordered=True
    )
    cve_lifespans_df['compilation_class'] = cve_lifespans_df['most_common_language'] \
        .apply(lambda language: compilation_class[language])
    cve_lifespans_df['Compilation class'] = pd.Categorical(
        cve_lifespans_df['compilation_class'].map({
            1: 'static', 2: 'dynamic'
        }),
        categories=['static', 'dynamic'],
        ordered=True
    )
    cve_lifespans_df['type_class'] = cve_lifespans_df['most_common_language'] \
        .apply(lambda language: type_class[language])
    cve_lifespans_df['Type class'] = pd.Categorical(
        cve_lifespans_df['type_class'].map({
            1: 'strong', 2: 'weak'
        }),
        categories=['weak', 'strong'],
        ordered=True
    )
    cve_lifespans_df['memory_model'] = cve_lifespans_df['most_common_language'] \
        .apply(lambda language: memory_model[language])
    cve_lifespans_df['Memory model'] = pd.Categorical(
        cve_lifespans_df['memory_model'].map({
            1: 'managed', 2: 'non managed'
        }),
        categories=['managed', 'non managed'],
        ordered=True
    )
    cve_lifespans_df['extended_programming_paradigm'] = cve_lifespans_df['most_common_language'] \
        .apply(lambda language: extended_programming_paradigm[language])
    cve_lifespans_df['Programming paradigm (extended)'] = pd.Categorical(
        cve_lifespans_df['extended_programming_paradigm'].map({
            1: 'functional',
            2: 'object oriented',
            3: 'procedural',
            4: 'scripting'
        }),
        categories=[
            'object oriented',
            'procedural',
            'functional',
            'scripting'
        ],
        ordered=True
    )
    return cve_lifespans_df


def read_language_to_class_dict(language_to_class_dict_filename):
    with open(language_to_class_dict_filename, 'r') as f:
        language_to_class = json.load(f)
    return language_to_class


if __name__ == '__main__':
    main()
