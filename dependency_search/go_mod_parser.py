#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <go_mod_file> <parquet_result_file>

Specification https://go.dev/ref/mod#go-mod-file
"""
import sys

import pandas as pd


def main():
    requirements_filename = sys.argv[1]
    dataframe_filename = sys.argv[2]
    with open(requirements_filename) as f:
        lines = f.read()
        result = extract_dependencies(lines)
        data = pd.DataFrame(result, columns=['library', 'version'])
        data.to_parquet(dataframe_filename)


def find_end_index(lines, start_index):
    """
    Finds end index of closing ")" of requirements section
    :param lines: all lines in file
    :param start_index: start index of requirements section
    :return: end index of requirements section
    """
    end = start_index
    for line in lines[start_index:]:
        end += 1
        if ')' in line:
            break
    return end


def retrieve_dependencies_section(lines, start_index, end_index):
    """
    Parses requirements section
    :param lines: all lines in file
    :param start_index: start of requirements section "requirements ("
    :param end_index: end of requirements section ")"
    :return: list of library names and corresponding versions
    """
    requirement_lines = lines[start_index + 1:end_index - 1]
    dependencies = []
    for line in requirement_lines:
        content = line.strip().split(' ')
        library_name = content[0]
        library_version = content[1]
        dependencies.append((library_name, library_version))
    return dependencies


def retrieve_dependency(line):
    """
    Parses single dependency
    :param line: string in format "require golang.org/x/net v1.2.3"
    :return: library name and version
    """
    content = line.split(' ')
    library_name = content[1]
    library_version = content[2]
    return library_name, library_version


def extract_dependencies(file_content):
    """
    Takes go mod file contents as string
    Returns list of libraries with versions
    """
    lines = file_content.splitlines()
    dependencies = []
    for index, line in enumerate(lines):
        if 'require' in line:
            if 'require (' in line:
                start_index = index
                end_index = find_end_index(lines, start_index)
                multiple_dependencies = retrieve_dependencies_section(lines, start_index, end_index)
                dependencies.extend(multiple_dependencies)
            else:
                single_dependency = retrieve_dependency(line)
                dependencies.append(single_dependency)

    return dependencies


if __name__ == '__main__':
    main()
