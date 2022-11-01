#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) requirements.txt <parquet_result_file>

Specification https://pip.pypa.io/en/stable/reference/requirements-file-format/#requirements-file-format
"""
import requirements
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


def extract_dependencies(lines):
    """
    Takes requirements.txt contents as string
    Returns list of libraries with versions
    """
    dependencies = []
    for req in requirements.parse(lines):
        dependencies.append((req.name, req.specs))
    return dependencies


if __name__ == '__main__':
    main()
