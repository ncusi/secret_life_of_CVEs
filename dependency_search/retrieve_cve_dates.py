#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <cve_df_filename> <published_cve_df_filename>

Retrieves cve published date via rest api from instance of CVE-Search
Requires results of cve_search_parser.py
"""
import json
import sys
import urllib.request

import pandas as pd
from tqdm import tqdm


def main():
    dataframe_filename = sys.argv[1]
    cve_published_date_filename = sys.argv[2]
    df = pd.read_parquet(dataframe_filename)
    time_df = df[['commit', 'commit_cves', 'commit_time', 'project_names']]
    exploded_time_df = time_df.explode('commit_cves')
    unique_cves = exploded_time_df['commit_cves'].dropna().unique()
    result, errors = download_cve_published_date(unique_cves)
    cve_published_date_df = pd.DataFrame(result, columns=['cve', 'published_date'])
    cve_published_date_df.to_parquet(cve_published_date_filename)
    cve_published_date_errors_df = pd.DataFrame(result, columns=['cve', 'error'])
    cve_published_date_errors_df.to_parquet(cve_published_date_filename + '_errors')


def download_cve_published_date(unique_cves):
    result = []
    errors = []
    for cve in tqdm(unique_cves):
        try:
            cve_published_date = gather_cve_published_data(cve)
            result.append((cve, cve_published_date))
        except Exception as err:
            print(cve)
            print(err)
            errors.append((cve, err))
            pass
    return result, errors


def gather_cve_published_data(cve):
    url = 'http://158.75.112.151:5000/api/cve/'
    request_url = url + cve
    with urllib.request.urlopen(request_url) as result:
        data = json.load(result)
        return data['Published']


if __name__ == '__main__':
    main()
