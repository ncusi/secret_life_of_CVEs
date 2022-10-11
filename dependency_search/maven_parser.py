#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) pom.xml
"""
import sys

import xml.etree.ElementTree as ET


def main():
    filename = sys.argv[1]
    with open(filename) as f:
        content = f.read()
        result = extract_dependencies(content)
        print(result)


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
    library_name = group_id + ':' + artifact_id
    library_version = version
    return library_name, library_version


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
                library_name, library_version = extract_dependency(xml_dependency)
                dependencies.append((library_name, library_version))
    return dependencies


if __name__ == '__main__':
    main()
