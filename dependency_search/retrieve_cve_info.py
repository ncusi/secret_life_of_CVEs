#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName)s <cve_df_filename> <cve_df_info_filename>

Retrieves CVE information via REST API from an instance of CVE-Search
Requires result of cve_search_parser.py, or any other dataframe with 'cve' column
Saves dataframe with columns 'cve', 'cvss', 'cwe'

Based on retrieve_cve_dates.py
"""
import json
import sys
import os
import urllib3

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def usage():
    """print script usage and exit program
    """
    # %(prog)s is argparse specifier for the program name
    print(__doc__ % {'scriptName': os.path.basename(sys.argv[0])})
    exit()


def main():
    if len(sys.argv) <= 2:
        usage()

    dataframe_filename = sys.argv[1]
    cve_dataframe_filename = sys.argv[2]

    df = pd.read_parquet(dataframe_filename)
    if 'cve' not in df.columns:
        cves_df = df[['commit_cves']]
        cve_s = cves_df.explode('commit_cves')['commit_cves']
    else:
        cve_s = df['cve']
    unique_cve = cve_s.dropna().unique()

    result = download_cve_published_date(unique_cve)

    cve_published_date_df = pd.DataFrame(result,
                                         columns=['cve', 'cvss', 'cwe', 'error']).convert_dtypes()
    cve_published_date_df.to_parquet(cve_dataframe_filename)


def download_cve_published_date(unique_cve):
    """
    Downloads each cve details via rest api from instance of CVE-Search
    :param unique_cve: list of unique cve
    :return: list of results (cve, cvss, cve)
    """
    http = urllib3.PoolManager()

    result = Parallel(n_jobs=6 * 12, backend="threading")(
        delayed(gather_cve_published_data)(http, cve) for cve in tqdm(unique_cve)
    )

    return result


def gather_cve_published_data(http, cve):
    """
    Retrieves cve details via rest api from instance of CVE-Search
    :param http: urllib-compatible object, with .request("GET", request_url) method
    :param cve: Unique cve in format CVE-\d{4}-\d{4,7}, for example CVE-2014-2972
    :return: (cve, cvss, cws, error) tuple
    """
    url = 'http://158.75.112.151:5000/api/cve/'
    request_url = url + cve
    cvss = None
    cwe = None
    error = None
    try:
        response = http.request("GET", request_url)
        data = json.loads(response.data.decode('utf-8'))
        cvss = data['cvss']
        cwe  = data['cwe']
    except Exception as exception:
        error = str(exception)
    return cve, cvss, cwe, error


if __name__ == '__main__':
    main()