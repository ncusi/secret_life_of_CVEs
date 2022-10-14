#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) requirements.txt <parquet_result_file>
"""
import sys

import pandas as pd


def main():
    requirements_filename = sys.argv[1]
    dataframe_filename = sys.argv[2]
    with open(requirements_filename) as f:
        lines = f.readlines()
        result = extract_dependencies(lines)
        data = pd.DataFrame(result, columns=['library', 'version'])
        data.to_parquet(dataframe_filename)


def extract_dependencies(lines):
    """
    Takes requirements.txt contents as list of lines
    Returns list of libraries with versions
    """
    dependencies = []
    for line in lines:
        if line.strip() != '':
            split_result = line.strip().split("==")
            if len(split_result) == 2:
                library, version = split_result
                dependencies.append((library, version))
    return dependencies


if __name__ == '__main__':
    main()
