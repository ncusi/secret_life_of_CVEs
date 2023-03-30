# The Secret Life of CVEs - code and notebooks

This repository contains scripts to process and join data from the
World of Code dataset (see <https://arxiv.org/abs/2010.16196>) and
CVE (Common Vulnerabilities and Exposures) dataset 
(gathered using the [cve-search](https://www.cve-search.org/) project),
that were used in the _"The Secret Life of CVEs"_ paper submission,
accepted to MSR 2023 Challenge: <https://conf.researchr.org/track/msr-2023/msr-2023-mining-challenge>.

Results were analyzed with the help of Jupyter Notebooks,
available in the 'notebooks/' subdirectory.

The final dataset, along with the source code and notebooks
used to extract and analyze the data, are accessible on Figshare:
<https://doi.org/10.6084/m9.figshare.22007003>.


## Running the code

The code requires Python 3 to run.

Results of each script are saved in the `data/` directory.  Files in this
directory without any extension are pandas dataframes saved in parquet
file format.

The easiest way to run all scripts in order is to use DVC (Data Version Control)
command line tool.  To recreate data processing and filtering on your local machine,
use "`dvc repro`" in main directory, which will run all scripts according to
what is in the "_dvc.yaml_" file, replacing data folder content when needed.

The data is also available on DagsHub, in the connected repository:
<https://dagshub.com/ncusi/secret_life_of_CVEs>, from which you can get data
from with "`dvc pull`" (after configuring DagsHub as dvc remote).


## Replicating paper results

To replicate the results in the paper, after recreating data files, or
downloading them from Figshare, or from DagsHub, use Jupyter notebooks
from the "`notebooks/`" directory.


## World of Code data extraction

The code starts with data already extracted from World of Code.
To recreate data extraction from WoC servers:
- Run "projects_stats/with_CVS_in_commit_message_ignore_case.sh" on WoC servers
- Run "cat search.CVE_in_commit_message_ignore_case.lstCmt_9.out | cut -d';' -f1 | ~/lookup/getValues c2P 1 >projects_with_CVE_fix.txt" on WoC servers
- Run "cve_search_parser.py search.CVE_in_commit_message.lstCmt_9.out projects_with_CVE_fix.txt cve_df_filename" on WoC servers
- Copy the result 'cve_df_filename' to local machine, and replace 'cve_df_filename' in 'data/' folder.
