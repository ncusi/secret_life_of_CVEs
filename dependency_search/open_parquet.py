#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <df>

Debug util to check dataframe shape and head rows
"""
import sys

import pandas as pd


def main():
    filename = sys.argv[1]
    data = pd.read_parquet(filename)
    print(data.shape)
    print(data.head())


if __name__ == '__main__':
    main()
