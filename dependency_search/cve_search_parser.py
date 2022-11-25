#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <search.CVE_in_commit_message.lstCmt_9.out> <projects_with_CVE_fix.txt> <parquet_result_file>

Creates pandas dataframe saved as parquet file with commits connected to cve from results of with_CVS_in_commit_message.sh
"""
import re
import sys
from os.path import splitext

import pandas as pd
from oscar import Commit


def main():
    cve_search_df_filename = sys.argv[1]
    project_name_df_filename = sys.argv[2]
    dataframe_filename = sys.argv[3]
    column_names = ['commit', 'tree', 'parent', 'author', 'commiter', 'author_time', 'commiter_time', 'author_timezone',
                    'commiter_timezone', 'commit_message']
    df = pd.read_csv(cve_search_df_filename, sep=";", encoding='latin-1', names=column_names)
    project_name_df = pd.read_fwf(project_name_df_filename, header=None)
    split_result = project_name_df[0].str.split(';', n=1, expand=True)
    project_name_df['commit'] = split_result[0]
    project_name_df['project_names'] = split_result[1]

    # df = df.head()

    extracted_df = add_cve(df)
    extracted_df = pd.merge(left=extracted_df, right=project_name_df, how='left', left_on=['commit'],
                            right_on=['commit']).drop([0], axis=1)
    extracted_df = add_dependency_files(extracted_df)
    # print(extracted_df.columns)
    # print(extracted_df)
    extracted_df.to_parquet(dataframe_filename)


def add_cve(df):
    """
    Prepares new dataset with extracted cve
    :param df: dataframe with 'commit' and 'commit_message' columns
    :return: new dataframe with 'commit' and 'cves' column
    """
    cve_df = pd.DataFrame.from_records(df.apply(lambda row: extract_cve(row['commit'], row['commit_message'],
                                                                        int(row['commiter_time'])), axis=1))
    return cve_df


def extract_cve(commit_sha, commit_message, commit_time):
    """
    Extracts cve numbers from commit message
    :param commit_sha: unique id of commit
    :param commit_message: message containing cve
    :param commit_time: time of commit
    :return: record with list of extracted cves and commit time
    """
    pattern = r'(CVE-\d{4}-\d{4,7})'
    cve_entries = re.findall(pattern, commit_message)
    from datetime import datetime
    return {'commit': commit_sha, 'commit_cves': cve_entries, 'commit_time': str(datetime.fromtimestamp(commit_time))}


def add_dependency_files(df):
    files_df = pd.DataFrame.from_records(df['commit'].apply(find_dependency_files))
    result_df = pd.concat([df, files_df], axis=1)
    return result_df


def find_dependency_files(commit_sha):
    commit = Commit(commit_sha)

    total_number_of_files = 0
    for changed_file_name in commit.changed_file_names:
        total_number_of_files += 1

    dependencies = find_file_name_dependencies(commit.changed_file_names)
    documentation = find_file_name_documentation(commit.changed_file_names)
    extensions = find_file_name_extensions(commit.changed_file_names)
    result = {
        'total_number_of_files': total_number_of_files
    }
    result.update(dependencies)
    result.update(documentation)
    result.update(extensions)
    return result


def find_file_name_dependencies(changed_files):
    modifies_requirements_txt = False
    modifies_cargo_toml = False
    modifies_go_mod = False
    modifies_ivy_xml = False
    modifies_maven_pom_xml = False
    modifies_npm_package_json = False
    modifies_nuget_nuspec_xml = False

    for changed_file_name in changed_files:
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

    dependencies = {
        'dep_requirements': modifies_requirements_txt,
        'dep_cargo': modifies_cargo_toml,
        'dep_go_mod': modifies_go_mod,
        'dep_ivy': modifies_ivy_xml,
        'dep_maven': modifies_maven_pom_xml,
        'dep_npm': modifies_npm_package_json,
        'dep_nuget': modifies_nuget_nuspec_xml
    }
    return dependencies


def find_file_name_documentation(changed_file_names):
    readme_regexp = re.compile(r'README.md')
    documentation_readme_count = 0
    for changed_file_name in changed_file_names:
        if readme_regexp.match(changed_file_name.decode()):
            documentation_readme_count += 1

    documentation = {'doc_readme_count': documentation_readme_count}
    return documentation


def find_file_name_extensions(changed_files):
    extensions = {}
    for changed_file in changed_files:
        file, raw_extension = splitext(changed_file)
        extension = 'ext_' + raw_extension.decode()
        if extension in extensions:
            extensions[extension] += 1
        else:
            extensions[extension] = 1
    return extensions


if __name__ == '__main__':
    main()
