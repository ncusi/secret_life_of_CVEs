#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) pom.xml <parquet_result_file>
"""
import sys
import xml.etree.ElementTree as ET

import pandas as pd


def main():
    pom_filename = sys.argv[1]
    dataframe_filename = sys.argv[2]
    with open(pom_filename) as f:
        content = f.read()
        dependencies = extract_dependencies_with_external_version(content)
        dependencies_with_url = prepare_artifact_url(dependencies)
        data = pd.DataFrame(dependencies_with_url, columns=['group_id', 'artifact_id', 'version', 'url'])
        data.to_parquet(dataframe_filename)


def extract_dependency(dependency):
    group_id = None
    artifact_id = None
    version = None
    for element in dependency:
        if 'groupId' in element.tag:
            group_id = element.text
        if 'artifactId' in element.tag:
            artifact_id = element.text
        if 'version' in element.tag:
            version = element.text
    return group_id, artifact_id, version


def extract_properties(content):
    """
    Takes pom.xml contents
    Usually this is xml structure project->properties->property
    Property might be a version of library
    Returns dictionary property name -> value
    """
    xml_root = ET.fromstring(content)
    properties = {}
    for child in xml_root:
        if 'properties' in child.tag:
            xml_properties = child
            for xml_property in xml_properties:
                xml_property_with_namespace = xml_property.tag
                namespace_end_index = xml_property_with_namespace.find('}')
                variable_name = xml_property_with_namespace[namespace_end_index + 1:]
                properties[variable_name] = xml_property.text
    return properties


def extract_dependencies(content):
    """
    Takes pom.xml contents
    Usually this is xml structure project->dependencies->dependency
    Returns list of libraries with versions
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


def extract_dependencies_with_external_version(content):
    """
    Takes pom.xml content, extracts used dependencies, resolving versions from present variables (properties)
    :param content: maven pom.xml
    :return: extracted dependencies
    """
    properties = extract_properties(content)
    dependencies = extract_dependencies(content)
    adjusted_dependencies = []
    for dependency in dependencies:
        group_id, artifact_id, library_version = dependency
        if "${" == library_version[0:2] and "}" == library_version[-1:]:
            lookup_variable_name = library_version[2:-1]
            if lookup_variable_name in properties:
                adjusted_library_version = properties[lookup_variable_name]
                adjusted_dependencies.append((group_id, artifact_id, adjusted_library_version))
        else:
            adjusted_dependencies.append(dependency)
    return adjusted_dependencies


def prepare_artifact_url(dependencies):
    """
    Takes maven dependencies, adds url to check vulnerability via mvnrepository site
    :param dependencies: list of (group_id, artifact_id, version)
    :return: list of (group_id, artifact_id, version, artifact_url)
    """
    dependencies_with_vulnerability_url = []
    for dependency in dependencies:
        group_id, artifact_id, version = dependency
        url = f"https://mvnrepository.com/artifact/{group_id}/{artifact_id}/{version}"
        dependencies_with_vulnerability_url.append((group_id, artifact_id, version, url))
    return dependencies_with_vulnerability_url


if __name__ == '__main__':
    main()
