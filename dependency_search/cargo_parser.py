#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) Cargo.toml <parquet_result_file>

Specification https://doc.rust-lang.org/cargo/guide/cargo-toml-vs-cargo-lock.html
https://toml.io/en/
"""
import sys

import pandas as pd
import tomli


def main():
    cargo_toml_filename = sys.argv[1]
    dataframe_filename = sys.argv[2]
    with open(cargo_toml_filename) as f:
        lines = f.read()
        result = extract_dependencies(lines)
        data = pd.DataFrame(result, columns=['library', 'version'])
        data.to_parquet(dataframe_filename)


def extract_dependencies(lines):
    """
    Takes Cargo.toml contents as string
    Returns list of libraries with versions
    """
    content = tomli.loads(lines)
    dependencies = []
    for dependency in content['dependencies'].keys():
        possible_version = content['dependencies'][dependency]
        if isinstance(possible_version, str):
            version = possible_version
        else:
            version = possible_version['version']
        dependencies.append((dependency, version))
    return dependencies


if __name__ == '__main__':
    main()
