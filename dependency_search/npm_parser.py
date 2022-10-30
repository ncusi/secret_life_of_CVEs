#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) package.json <parquet_result_file>

Specification https://docs.npmjs.com/cli/v8/configuring-npm/package-json#dependencies
"""
import json
import sys

import pandas as pd


def main():
    pom_filename = sys.argv[1]
    dataframe_filename = sys.argv[2]
    with open(pom_filename) as f:
        content = f.read()
        dependencies = extract_dependencies(content)
        data = pd.DataFrame(dependencies, columns=['name', 'version'])
        data.to_parquet(dataframe_filename)


def extract_dependencies(content):
    """
    Takes package.json contents
    Usually this is json structure dependencies->dependency
    Returns list of libraries with versions
    """
    package = json.loads(content)
    package_dependencies = package['dependencies']
    dependencies = []
    for library_name in package_dependencies.keys():
        library_version = package_dependencies[library_name]
        dependencies.append((library_name, library_version))
    return dependencies


if __name__ == '__main__':
    main()
