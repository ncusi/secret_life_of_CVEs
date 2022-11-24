#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <search.CVE_in_commit_message.lstCmt_9.out> <parquet_result_file>

Creates pandas dataframe saved as parquet file with commits connected to cve from results of with_CVS_in_commit_message.sh
"""
import re
import sys
from os.path import splitext

import pandas as pd
from oscar import Commit


def main():
    cve_search_filename = sys.argv[1]
    dataframe_filename = sys.argv[2]
    column_names = ['commit', 'tree', 'parent', 'author', 'commiter', 'author_time', 'commiter_time', 'author_timezone',
                    'commiter_timezone', 'commit_message']
    df = pd.read_csv(cve_search_filename, sep=";", encoding='latin-1', names=column_names)
    extracted_df = add_cve(df)
    # extracted_df = find_project_names(extracted_df)
    extracted_df = add_dependency_files(extracted_df)
    # print(extracted_df.columns)
    # print(extracted_df.head())
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


def find_project_names(df):
    print(df)
    df['project_name'] = df['commit'].apply(find_project_name)
    return df


def find_project_name(commit_sha):
    # print(commit_sha)

    from subprocess import run, PIPE
    # args = ["~/lookup/getValues c2P"]
    args = "echo " + commit_sha + " | ~/lookup/getValues c2P"
    process = run(args, stdout=PIPE, shell=True)
    # print(process)
    output = process.stdout.decode()
    # print(output)
    return output.split(';')[1:]


if __name__ == '__main__':
    main()
