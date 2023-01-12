#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName)s <cve_df> <unique_project_info.parquet> <cve_with_project_info.parquet>

Expects <cve_df> to be a file in the Parquet format, which
contains DataFrame with either 'project_name' column with project name,
or 'project_names' column where projects may be in the form of ';' separated
list of projects.  This might be 'data/cve_df_filename', or result of any
subsequent steps augmenting this file, like e.g. calculate_cve_lifespan_per_project.py

Expects <unique_project_info.parquet> to be a file in the Parquet format,
which contains DataFrame with project metadata, with 'ProjectID' as the index,
created by the retrieve_metadata_from_WoC_mongodb.py script (that is by
the "retrieve_project_info" DVC stage).

The <cve_with_project_info.parquet> will be the output file in Parquet format.
"""
import math
import sys
from pathlib import Path

import pandas as pd


def usage():
    """Print script usage and exit program
    """
    # %(prog)s is argparse specifier for the program name
    print(__doc__ % {'scriptName': Path(sys.argv[0]).name})
    exit()


def merge_df(input_df, project_info_df):
    if project_info_df.index.name != 'ProjectID':
        print("??? project info dataframe index name " +
              f"'{project_info_df.index.name}' != 'ProjectID'",
              file=sys.stderr)

    if 'project_name' in input_df.columns:
        # easy case
        print("--> column 'project_name' found in input_df", file=sys.stderr)
        return input_df.join(project_info_df, on='project_name', how='left')

    elif 'project_names' not in input_df.columns:
        # an error
        print("!!! did not find 'project_name' or 'project_names' column in input_df", file=sys.stderr)
        sys.exit(1)

    # complex case
    print("--> column 'project_names' with lists of ';'-separated projects found in input_df",
          file=sys.stderr)
    zero_projects_mask = input_df['project_names'].isna()
    many_projects_mask = input_df['project_names'].str.contains(';', na=False)
    single_project_mask = ~zero_projects_mask & ~many_projects_mask
    zero_or_single_project_df = input_df[~many_projects_mask]
    many_projects_df = input_df[many_projects_mask]

    width = math.ceil(math.log10(input_df.shape[0]))
    print(f"--- {input_df.shape[0]} rows include:", file=sys.stderr)
    print(f"    - {zero_projects_mask.sum():{width}} without project name (N/A)", file=sys.stderr)
    print(f"    - {single_project_mask.sum():{width}} with a single project name", file=sys.stderr)
    print(f"    - {many_projects_mask.sum():{width}} with multiple project names", file=sys.stderr)
    print(f"    * {zero_or_single_project_df.shape[0]:{width}} with zero or one project", file=sys.stderr)

    print("--> Joining dataframes...", file=sys.stderr)
    print("-----> zero projects or single project...", file=sys.stderr)
    zero_or_single_project_df = zero_or_single_project_df.join(project_info_df, on='project_names', how='left')

    print("-----> multiple projects...", file=sys.stderr)
    # DEBUG
    #print(many_projects_df.dtypes)
    #print(project_info_df.dtypes)
    # NOTE: instead of using `.max()` or `.agg('max')`, one can use `.progress_aggregate('max')`,
    #       added to pandas GroupBy by running `tqdm.pandas()`, to have some level of progress indicator.
    # NOTE: string-valued 'RootFork' column will incidentally be also aggregated with `.max()`
    #       if there are different values for it
    #many_projects_df['projects_list'] = many_projects_df['project_names'].str.split(';', expand=False)
    # alternative
    #many_projects_df.loc[:, 'projects_list'] = many_projects_df['project_names'].str.split(';', expand=False)
    # another alternative
    #many_projects_df = many_projects_df.assign(
    #    projects_list=many_projects_df['project_names'].str.split(';', expand=False)
    #)
    many_merge_df = pd.DataFrame(
        {'projects_list': many_projects_df['project_names'].str.split(';', expand=False)}
    )\
        .explode('projects_list', ignore_index=False)\
        .rename(columns={'projects_list': 'project'})\
        .join(project_info_df, on='project', how='left')\
        .groupby(level=0)\
        .max()\
        .drop(columns=['project'])
    many_projects_df = many_projects_df.join(many_merge_df)

    print("--> Concatenating dataframes and sorting by index... ", file=sys.stderr)
    output_df = pd.concat([zero_or_single_project_df, many_projects_df])
    return output_df.sort_index()


def main():
    if len(sys.argv) <= 3:
        usage()

    input_df_filename = Path(sys.argv[1])
    projects_info_df_filename = Path(sys.argv[2])
    output_df_filename = Path(sys.argv[3])

    print(f"Reading input files '{input_df_filename}' and '{projects_info_df_filename}'...",
          file=sys.stderr)
    input_df = pd.read_parquet(input_df_filename)
    project_info_df = pd.read_parquet(projects_info_df_filename)
    print(f"- '{input_df_filename}' has {input_df.shape} rows and columns", file=sys.stderr)
    print(f"- '{projects_info_df_filename}' has {project_info_df.shape} rows and columns", file=sys.stderr)

    print(f"Merging dataframes...", file=sys.stderr)
    output_df = merge_df(input_df, project_info_df)

    print(f"Writing '{output_df_filename}'...", file=sys.stderr)
    output_df.to_parquet(output_df_filename)


if __name__ == '__main__':
    main()
