#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) nuspec.xml <parquet_result_file>

Specification https://learn.microsoft.com/en-us/nuget/reference/nuspec#dependencies-element
"""
import sys
import xml.etree.ElementTree as ET

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
    Takes nuspec xml contents
    Usually this is xml structure package->metadata->dependencies->dependency
    Returns list of libraries with versions
    """
    xml_root = ET.fromstring(content)

    dependencies = []
    for child in xml_root:
        if 'metadata' in child.tag:
            dependencies.extend(extract_metadata_dependencies(child))
    return dependencies


def extract_metadata_dependencies(metadata):
    dependencies = []
    for child in metadata:
        if 'dependencies' in child.tag:
            xml_dependencies = child
            for xml_dependency in xml_dependencies:
                library_name, library_version = extract_dependency(xml_dependency)
                dependencies.append((library_name, library_version))
    return dependencies


def extract_dependency(xml_dependency):
    name = xml_dependency.attrib.get('id')
    version = xml_dependency.attrib.get('version')
    return name, version


if __name__ == '__main__':
    main()
