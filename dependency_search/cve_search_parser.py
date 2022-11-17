#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <search.CVE_in_commit_message.lstCmt_9.out> <parquet_result_file>

Creates pandas dataframe saved as parquet file with commits connected to cve from results of with_CVS_in_commit_message.sh
"""
import sys

import pandas as pd

from oscar import Commit


def main():
    cve_search_filename = sys.argv[1]
    dataframe_filename = sys.argv[2]
    column_names = ['commit', 'tree', 'parent', 'author', 'commiter', 'author_time', 'commiter_time', 'author_timezone',
                    'commiter_timezone', 'commit_message']
    df = pd.read_csv(cve_search_filename, sep=";", encoding='latin-1', names=column_names)
    df.set_index('commit')
    extracted_df = find_cve(df)
    extracted_df = extracted_df[['commit', 'cve']]
    # extracted_df = find_project_names(extracted_df.head())
    extracted_df = add_dependency_files(extracted_df)
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


def add_dependency_files(df):
    files_df = pd.DataFrame.from_records(df['commit'].apply(find_dependency_files))
    files_df.set_index('commit')
    result_df = pd.concat([df, files_df], axis=1)
    return result_df


def find_dependency_files(commit_sha):
    commit = Commit(commit_sha)
    modifies_requirements_txt = False
    modifies_cargo_toml = False
    modifies_go_mod = False
    modifies_ivy_xml = False
    modifies_maven_pom_xml = False
    modifies_npm_package_json = False
    modifies_nuget_nuspec_xml = False

    for changed_file_name in commit.changed_file_names:
        if b'requirements.txt' in changed_file_name:
            modifies_requirements_txt = True
        if b'Cargo.toml' in changed_file_name:
            modifies_cargo_toml = True
        if b'go_mod' in changed_file_name:
            modifies_go_mod = True
        if b'ivy.xml' in changed_file_name:
            modifies_ivy_xml = True
        if b'pom.xml' in changed_file_name:
            modifies_maven_pom_xml = True
        if b'package.json' in changed_file_name:
            modifies_npm_package_json = True
        if b'nuspec.xml' in changed_file_name:
            modifies_nuget_nuspec_xml = True

    result = {'commit': commit_sha,
              'requirements': modifies_requirements_txt,
              'cargo': modifies_cargo_toml,
              'go_mod': modifies_go_mod,
              'ivy': modifies_ivy_xml,
              'maven': modifies_maven_pom_xml,
              'npm': modifies_npm_package_json,
              'nuget': modifies_nuget_nuspec_xml}
    return result


def find_project_names(df):
    print(df)
    df['project_name'] = df['commit'].apply(find_project_name)
    return df


def find_project_name(commit_sha):
    print(commit_sha)
    commit = Commit(commit_sha)
    tmp = []
    for project_name in commit.project_names:
        tmp.append(project_name)
    print(tmp)
    return tmp[0]


if __name__ == '__main__':
    main()
