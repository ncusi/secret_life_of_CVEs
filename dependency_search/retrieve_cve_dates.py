#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <cve_df_filename> <published_cve_df_filename>

Retrieves cve published date via rest api from instance of CVE-Search
Requires results of cve_search_parser.py
Saves dataframe with columns 'cve', 'published_date', 'error'
"""
import json
import sys
import urllib3

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def main():
    dataframe_filename = sys.argv[1]
    cve_published_date_filename = sys.argv[2]
    df = pd.read_parquet(dataframe_filename)
    time_df = df[['commit', 'commit_cves', 'commiter_time', 'author_time', 'project_names']]
    exploded_time_df = time_df.explode('commit_cves')
    unique_cves = exploded_time_df['commit_cves'].dropna().unique()
    result = download_cve_published_date(unique_cves)
    cve_published_date_df = pd.DataFrame(result, columns=['cve', 'published_date', 'error'])
    cve_published_date_df.to_parquet(cve_published_date_filename)


def download_cve_published_date(unique_cves):
    """
    Downloads each cve details via rest api from instance of CVE-Search
    :param unique_cves: list of unique cve
    :return: list of results (cve, published date, error)
    """
    http = urllib3.PoolManager()

    result = Parallel(n_jobs=6 * 12, backend="threading")(
        delayed(gather_cve_published_data)(http, cve) for cve in tqdm(unique_cves)
    )

    return result


def gather_cve_published_data(http, cve):
    """
    Retrieves cve details via rest api from instance of CVE-Search
    :param cve: Unique cve in format CVE-\d{4}-\d{4,7}, for example CVE-2014-2972
    :return: publish date for selected cve
    """
    url = 'http://158.75.112.151:5000/api/cve/'
    request_url = url + cve
    published = None
    error = None
    try:
        response = http.request("GET", request_url)
        data = json.loads(response.data.decode('utf-8'))
        published = data['Published']
    except Exception as exception:
        error = str(exception)
    return cve, published, error

if __name__ == '__main__':
    main()
