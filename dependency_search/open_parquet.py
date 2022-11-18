#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) requirements.txt
"""
import sys

import pandas as pd


def main():
    filename = sys.argv[1]
    data = pd.read_parquet(filename)
    print(data.head())
    print(data.shape)
    print(data[data['requirements']==True].head())
    print(data[data['cve']=='CVE-2014-0472'])

if __name__ == '__main__':
    main()
