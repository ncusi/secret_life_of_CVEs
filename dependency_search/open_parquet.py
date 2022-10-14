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


if __name__ == '__main__':
    main()
