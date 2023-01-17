#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName)s [OPTIONS] <cve_survival_input_df>

Run 'cve_surv_analysis.py' script for different risk factors, appending to group metrics
(Dxy with confidence intervals), and putting each result in a separate subdirectory.
"""
import pathlib
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import yaml
from tqdm import tqdm


def usage():
    """Print script usage and exit program"""
    # %(prog)s is argparse specifier for the program name
    print(__doc__ % {'scriptName': Path(sys.argv[0]).name})
    exit()


def read_params(params_file: pathlib.Path) -> dict:
    params = {}
    print(f"Reading parameters from '{params_file}'...", file=sys.stderr)
    with params_file.open(mode='r') as yaml_file:
        all_params = yaml.safe_load(yaml_file)
        if all_params is None:
            print(f"- no documents in parameters file", file=sys.stderr)
            all_params = {}
        else:
            print(f"- read {len(all_params)} sections from first document in parameters file",
                  file=sys.stderr)
        if (all_params is not None) and ("eval" in all_params):
            params = all_params["eval"]
            print(f"- read {len(params)} parameters from 'eval' section", file=sys.stderr)

    if 'short_name' not in params:
        print(f"Did not find required 'short_name' parameter, exiting!", file=sys.stderr)
        exit(3)

    return params


def enumerate_risks(exec_path: pathlib.Path, params: dict, input_df_filename: pathlib.Path):
    RiskFactor = namedtuple('RiskFactor', ['column_name', 'prefix_dir'])
    risk_factors = [
        RiskFactor._make(elem) for elem in [
            ('Programming paradigm', 'programming_paradigm'),
            ('Programming paradigm (extended)', 'extended_programming_paradigm'),
            ('Compilation class', 'compilation_class'),
            ('Type class', 'type_class'),
            ('Memory model', 'memory_model'),
            ('CVSS v3.1 Ratings', 'cvss_v3.1_ratings'),
            ('CVSS v2.0 Ratings', 'cvss_v2.0_ratings'),
        ]
    ] + [
        RiskFactor(column_name=name, prefix_dir=name) for name in [
            'embargo_min',
            'embargo_max',
            'access.authentication',
            'access.complexity',
            'access.vector',
            'impact.availability',
            'impact.confidentiality',
            'impact.integrity',
        ]
    ]

    eval_path = params['eval_path'] if 'eval_path' in params else 'eval'

    for risk in tqdm(risk_factors):
        subprocess.run([
            exec_path,
            f"--eval-path={eval_path}/{params['short_name']}",
            f"--risk-column={risk.column_name}",
            f"--path-prefix=details/{risk.prefix_dir}/",
            "--append-to-group-metrics",
            input_df_filename
        ], stderr=subprocess.DEVNULL)


def main():
    """Run as script"""
    if len(sys.argv) != 2:
        usage()

    # read options and arguments
    input_df_filename = Path(sys.argv[1])
    if not input_df_filename.exists():
        print(f"File '{input_df_filename}' does not exist, exiting!")
        exit(1)

    # read parameters
    params_file = Path('params.yaml')
    if not params_file.exists():
        print(f"Could not find YAML file '{params_file}' with parameters, exiting!")
        exit(2)
    params = read_params(params_file)

    # find exec path
    exec_path = Path(sys.argv[0]).resolve().parent / 'cve_surv_analysis.py'
    print(f"Using '{exec_path}' executable", file=sys.stderr)

    # gather data, by running script for different risk factors
    enumerate_risks(exec_path, params, input_df_filename)


if __name__ == '__main__':
    main()
