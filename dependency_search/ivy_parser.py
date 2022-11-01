#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) ivy.xml <parquet_result_file>
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
        data = pd.DataFrame(dependencies, columns=['group_id', 'artifact_id', 'version'])
        data.to_parquet(dataframe_filename)


def extract_dependency(dependency):
    """
    Takes single dependency node from ivy.xml file
    :param dependency: xml node for dependency
    :return: extracted library name and version
    """
    group_id = dependency.attrib.get('org')
    artifact_id = dependency.attrib.get('name')
    version = dependency.attrib.get('rev')
    return group_id, artifact_id, version


def extract_dependencies(content):
    """
    Takes ivy.xml contents, usually this is xml structure ivy-module->dependencies->dependency
    :param content: ivy.xml content
    :return: list of libraries with versions
    """
    xml_root = ET.fromstring(content)

    dependencies = []
    for child in xml_root:
        if 'dependencies' in child.tag:
            xml_dependencies = child
            for xml_dependency in xml_dependencies:
                group_id, artifact_id, library_version = extract_dependency(xml_dependency)
                dependencies.append((group_id, artifact_id, library_version))
    return dependencies


if __name__ == '__main__':
    main()
