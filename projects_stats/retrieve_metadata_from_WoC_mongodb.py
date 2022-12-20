#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName)s <cve_df_filename> <unique_project_info.parquet>

Expects <projects_df_filename> to be a file in the Parquet format, which
contains DataFrame with either 'project_name' column with project name,
or 'project_names' column where projects may be in the form of ';' separated
list of projects.

Saves result in <project_info_filename> in the Parquet format.

Based on add_project_metadata_from_WoC_mongodb.ipynb (not using nbdev).
"""
import gzip
import os
import sys
from pathlib import Path
# DEBUG
#import pprint

import pandas as pd
from tqdm import tqdm
from ssh_pymongo import MongoSession


def usage():
    """print script usage and exit program
    """
    # %(prog)s is argparse specifier for the program name
    print(__doc__ % {'scriptName': os.path.basename(sys.argv[0])})
    exit()


def extract_list_of_projects(projects_df_filename):
    r"""Extract list of unique projects from a given DataFrame-ish file

    If the `projects_df_filename` parameter points to CSV or gzipped CSV file
    (*.csv or *.csv.gz), treat is as single column dataframe, assuming that
    the CSV file does not include headers -- which means that the contents
    of this text file is simply the list of projects, one project per line.
    Example: 'projects_stats/interesting_projects_list.csv'

    Otherwise, assume that it is in Parquet format, and contains DataFrame.
    If this dataframe has column named "project_name", assume that it contains
    individual project names (each entry is a separate project name).
    Example: 'data/df_commits_with_one_project'

    If it doesn't have "project_name" column, check if it has "project_names"
    column (note the plural form).  Assume that this column contains semicolon
    (';') separated list of projects, to be split into list, and extracted.
    Example: 'data/cve_df_filename'

    Parameters
    ----------
    projects_df_filename : str
        Name of the file to extract projects list from,
        in CSV, gzipped CSV, or Parquet format
        (which stores pandas.DataFrame).

    Returns
    -------
    list
        List of project names
    """
    # Create new `pandas` methods which use `tqdm` progress
    # (can use tqdm_gui, optional kwargs, etc.)
    tqdm.pandas()

    projects_df_path = Path(projects_df_filename)
    # assume that file in .csv or .csv.gz format contains just a list of projects, one per line
    # with no header
    if projects_df_path.match("*.csv") or projects_df_path.match("*.csv.gz"):
        try:
            print(f"Reading list of projects from '{projects_df_path}'...", file=sys.stderr)
            projects_df = pd.read_csv(projects_df_path, header=None, names=["project_name"])
            return projects_df["project_name"].unique()

        except FileNotFoundError:
            print(f"File '{projects_df_path}' with the list of projects not found",
                  file=sys.stderr, flush=True)
            return []
        except gzip.BadGzipFile:
            print(f"File '{projects_df_path}' looks like *.gz, but is not a gzipped file",
                  file=sys.stderr, flush=True)
            return []

    print(f"Reading from '{projects_df_path}'...", file=sys.stderr)
    try:
        # try to use fastparquet engine, if it is available
        try:
            data_df = pd.read_parquet(projects_df_path, engine="fastparquet")
        except ImportError:
            data_df = pd.read_parquet(projects_df_path)

    except FileNotFoundError:
        print(f"File '{projects_df_path}' with the list of projects not found",
              file=sys.stderr, flush=True)
        return []

    if 'project_name' in data_df.columns:
        print("--> column 'project_name' found", file=sys.stderr)
        return data_df['project_name'].unique()
    elif 'project_names' in data_df.columns:
        print("--> column 'project_names' with lists of ';'-separated projects found", file=sys.stderr)
        projects_s = data_df['project_names'].str.split(';').explode(ignore_index=True)
        print(f"--- {projects_s.shape[0]} rows include {projects_s.count()} non-NA/null values; "
              f"include {projects_s.shape[0] - projects_s.count()} nulls", file=sys.stderr)
        return projects_s.dropna().unique()
    else:
        print("!!! did not find 'project_name' or 'project_names' column", file=sys.stderr)

    return []


def retrieve_metadata(projects_list, output_df_filename):
    r"""Retrieve metadata for each project on `projects_list`, save result to `output_df_filename`

    Project metadata is retrieved from the World of Code's MongoDB database,
    which stores metadata about authors and projects, see
    https://github.com/woc-hack/tutorial#mongo-database

    Only selected subset of "flat" data is retrieved, and then saved as
    DataFrame in `output_df_filename`.

    This script uses the 'P_metadata.U' collection in 'WoC' database.

    Parameters
    ----------
    projects_list : list
        Names of projects as they appear in the World of Code dataset
        (https://worldofcode.org/), e.g. ['buildroot_buildroot', 'xen-project_xen']
    output_df_filename : str
        Name of Parquet file to save pandas.DataFrame with results to

    Notes
    -----
    You need to be able to log in to World of Code (WoC) servers via SSH
    (https://github.com/woc-hack/tutorial#before-the-tutorial-access-to-da-servers)
    to be able to run this script.

    This script was creates as a part of MSR 2023 Mining Challenge
    https://conf.researchr.org/track/msr-2023/msr-2023-mining-challenge

    References
    ----------
    .. [1] Ma, Y., Dey, T., Bogart, C. et al. "World of code: enabling
       a research workflow for mining and analyzing the universe of open
       source VCS data". Empirical Software Eng. 26, 22 (2021).
       https://doi.org/10.1007/s10664-020-09905-9
       https://mockus.org/papers/WoC_EMSE.pdf

    .. [2] Mockus, A., Nolte, A., Herbsleb, A. "MSR Mining Challenge:
       World of Code", in Proceedings of the International Conference
       on Mining Software Repositories (MSR 2023)
    """
    # TODO: make those settings configurable
    user = os.environ['USER']
    home = os.getenv('HOME', '/home/' + user)
    woc_user = user
    woc_host = 'da0.eecs.utk.edu'
    woc_port = 443
    woc_key = home + '/.ssh/id_rsa_woc'
    woc_mongo_uri = 'mongodb://da5.eecs.utk.edu/'
    woc_ver = 'U'

    # TODO: error checking, e.g. that `woc_key` file exists
    # ...

    # using ssh_pymongo = sshtunnel + pymongo
    print(f"Connecting to {woc_mongo_uri} via SSH tunnel to {woc_host}...", file=sys.stderr)
    session = MongoSession(
        host=woc_host,
        port=woc_port,
        user=woc_user,
        key=woc_key,
        uri=woc_mongo_uri
    )
    client = session.connection
    database = client['WoC']
    collection = database[f'P_metadata.{woc_ver}']

    projects_info = []
    num_errors = 0

    # MAYBE: optimize using joblib, or other solution
    print(f"Processing {len(projects_list)} projects, sequentially...", file=sys.stderr)
    for project in tqdm(projects_list):
        info = collection.find_one(
            {'ProjectID': project},
            {'_id': 0, 'CommunitySize': 1, 'NumCore': 1,
             'EarliestCommitDate': 1, 'LatestCommitDate': 1,
             'NumActiveMon': 1,
             'NumAuthors': 1,
             'NumBlobs': 1,
             'NumCommits': 1,
             'NumFiles': 1,
             'NumForks': 1,
             'NumStars': 1,
             'ProjectID': 1,
             'RootFork': 1,
             })
        if info is not None:
            # TODO: make this conversion configurable (be able to turn it off)
            info['EarliestCommitDate'] = pd.to_datetime(info['EarliestCommitDate'], unit='s', utc=True)
            info['LatestCommitDate'] = pd.to_datetime(info['LatestCommitDate'], unit='s', utc=True)
            projects_info.append(info)
        else:
            num_errors += 1
            projects_info.append({'ProjectID': project})

    if num_errors > 0:
        print(f"No data for {num_errors} project (added as all-null rows)")

    print(f"Converting {len(projects_info)} results to DataFrame...", file=sys.stderr)
    #pprint.pprint(projects_info)
    projects_info_df = pd.DataFrame\
        .from_records(projects_info,
                      columns=['ProjectID', 'RootFork',
                               'EarliestCommitDate', 'LatestCommitDate', 'NumActiveMon',
                               'NumAuthors', 'NumCore', 'CommunitySize',
                               'NumCommits', 'NumBlobs', 'NumFiles',
                               'NumForks', 'NumStars'])\
        .set_index('ProjectID')\
        .convert_dtypes()

    print(f"Saving DataFrame to '{output_df_filename}'...", file=sys.stderr)
    projects_info_df.to_parquet(output_df_filename)


# MAYBE: move to argparse or click library for parsing command line parameters
# https://docs.python.org/3/library/argparse.html
# https://click.palletsprojects.com/
def main():
    if len(sys.argv) <= 2:
        usage()

    projects_df_filename = sys.argv[1]
    projects_list = extract_list_of_projects(projects_df_filename)

    print(f"... {len(projects_list)} individual projects found", file=sys.stderr)
    # corner case
    if len(projects_list) == 0:
        print("NO PROJECTS FOUND", file=sys.stderr, flush=True)
        sys.exit(1)

    output_df_filename = sys.argv[2]
    retrieve_metadata(projects_list, output_df_filename)


if __name__ == '__main__':
    main()
