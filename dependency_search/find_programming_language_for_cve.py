#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <commits_with_published_cve_df> <extension_to_language.json> <combined_df>

Requires result of merge_cve_df_with_published_cve_df.py and prepare_extension_to_language_dict.py
"""
import json
import sys

import pandas as pd


def main():
    commits_with_published_cve_df_filename = sys.argv[1]
    extension_to_language_dict_filename = sys.argv[2]
    combined_df_filename = sys.argv[3]

    commits_df = pd.read_parquet(commits_with_published_cve_df_filename)
    extension_to_language = load_extension_to_language_dict(extension_to_language_dict_filename)
    result_df = add_language_to_df(commits_df, extension_to_language)
    result_df_lang_columns = result_df.filter(regex=("lang_.*"))
    result_df['most_common_language'] = result_df_lang_columns.idxmax(axis=1)
    result_df.to_parquet(combined_df_filename)


def load_extension_to_language_dict(extension_to_language_dict_filename):
    with open(extension_to_language_dict_filename, 'r') as file:
        extension_to_language = json.load(file)
        return extension_to_language


def add_language_to_df(commits_df, extension_to_language):
    """
    Adds columns with number of files connected to programing languages based on columns with file extensions.
    :param commits_df: dataframe with commits fixing cve,
     with "ext_.file_extension" columns representing modified number of files for selected extension
    :param extension_to_language: dictionary from file extensions to programming languages
    :return: expanded dataframe with columns representing files with given programming language
    """
    columns = commits_df.columns
    ext_columns = list(filter(lambda col: col[0:5] == 'ext_.', columns))

    lang_df = commits_df.apply(lambda row: map_row(row, ext_columns, extension_to_language),
                               axis='columns', result_type='expand')
    result_df = pd.concat([commits_df, lang_df], axis=1)
    return result_df


def get_languages(column_name, extension_to_language):
    """
    Assigns programming language for selected extension column
    :param column_name: column name in format "ext_.file_extension"
    :param extension_to_language: dictionary from file extensions to programming languages
    :return: list of programming languages matching provided extension form column name
    """
    extension = column_name[4:]
    if extension in extension_to_language:
        return extension_to_language[extension]
    else:
        return []


def map_row(row, ext_columns, extension_to_language):
    """
    Prepares dictionary of used languages for file extension columns in selected df row
    :param row: current dataframe row
    :param ext_columns: columns with filename extensions
    :param extension_to_language: dictionary from file extensions to programming languages
    :return: dictionary with used programming languages
    """
    result = {}
    for column in ext_columns:
        if pd.notna(row[column]):
            languages = get_languages(column, extension_to_language)
            for language in languages:
                language_key = "lang_" + language
                if language_key in result:
                    result[language_key] += row[column]
                else:
                    result[language_key] = row[column]
    return result


if __name__ == '__main__':
    main()
