# The Secret Life of CVEs - code and notebooks

This repository contains scripts to process and join data from the
World of Code dataset (see <https://arxiv.org/abs/2010.16196>) and
CVE (Common Vulnerabilities and Exposures) dataset 
(gathered using the [cve-search](https://www.cve-search.org/) project),
that were used in the _"The Secret Life of CVEs"_ paper submission,
accepted to MSR 2023 Challenge: <https://conf.researchr.org/track/msr-2023/msr-2023-mining-challenge>,
and published as DOI:[10.1109/MSR59073.2023.00056](https://doi.org/10.1109/MSR59073.2023.00056).

Results were analyzed with the help of Jupyter Notebooks,
available in the 'notebooks/' subdirectory.

The final dataset, along with the source code and notebooks
used to extract and analyze the data, are accessible on Figshare:
<https://doi.org/10.6084/m9.figshare.22007003>.

## How to cite
* Paper https://doi.org/10.1109/MSR59073.2023.00056
```
@INPROCEEDINGS{10174215,
  author={Przymus, Piotr and Fejzer, Mikołaj and Narębski, Jakub and Stencel, Krzysztof},
  booktitle={2023 IEEE/ACM 20th International Conference on Mining Software Repositories (MSR)}, 
  title={The Secret Life of CVEs}, 
  year={2023},
  volume={},
  number={},
  pages={362-366},
  doi={10.1109/MSR59073.2023.00056}
}
```
* Dataset https://doi.org/10.6084/m9.figshare.22007003
```
@article{Przymus2023,
  author = "Piotr Przymus and Mikołaj Fejzer and Jakub Narębski and Krzysztof Stencel",
  title = "{The Secret Life of CVEs - code and dataset}",
  year = "2023",
  month = "3",
  url = "https://figshare.com/articles/dataset/The_Secret_Life_of_CVEs_-_code_and_dataset/22007003",
  doi = "10.6084/m9.figshare.22007003.v1"
}
```

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


### World of Code data extraction

The code starts with data already extracted from World of Code.
To recreate data extraction from WoC servers:
- Run "projects_stats/with_CVS_in_commit_message_ignore_case.sh" on WoC servers
- Run "cat search.CVE_in_commit_message_ignore_case.lstCmt_9.out | cut -d';' -f1 | ~/lookup/getValues c2P 1 >projects_with_CVE_fix.txt" on WoC servers
- Run "cve_search_parser.py search.CVE_in_commit_message.lstCmt_9.out projects_with_CVE_fix.txt cve_df_filename" on WoC servers
- Copy the result 'cve_df_filename' to local machine, and replace 'cve_df_filename' in 'data/' folder.

### CVE data extraction

Retrieving CVE information (with the help of 'cve_information/retrieve_cve_info.py'
script) requires an instance of [CVE-Search](https://www.cve-search.org/) running,
as the script makes use of its REST API.  Currently the instance URI is hardcoded,
and you need to change it to be able to use your local instance, or some public
instance. You would need to change the following line in `gather_cve_published_data()`
function:

```.py
    url = 'http://158.75.112.151:5000/api/cve/'
    request_url = url + cve
```

The data file is available on Figshare, and via DagsHub.
