#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) requirements.txt
"""
import sys


def main():
    filename = sys.argv[1]
    with open(filename) as f:
        lines = f.readlines()
        result = extract_dependencies(lines)
        print(result)


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
