#!/bin/sh

# Re-created from playing with shell and shell API

# Assumes that 'lookup' repository was cloned into home directory
# git clone https://bitbucket.org/swsc/lookup ~/lookup

# Keys for identifying contents of Tokyo Cabinet's databases (*.tch) and basemaps (*.s)
# according to the README.md in https://github.com/woc-hack/tutorial
# - f = File
# - c = Commit
# - p = Project
# - P = Forked/Root Project
#
# See also:
# - https://github.com/woc-hack/tutorial
# - https://bitbucket.org/swsc/lookup/src/master/tutorial.md

# find number of all commits that changed 'requirements.txt' file,
# either in top level of repository, or in some subdirectory,
# (assuming no adversarial filenames)
zgrep -c -E "^requirements\.txt;" /da?_data/basemaps/gz/f2c*.s \
      >search.top_level_requirements.count.out
zgrep -c -E "/requirements\.txt;" /da?_data/basemaps/gz/f2c*.s \
      >search.in_subdirectory_requirements.count.out

# find all commits that changed 'requirements.txt' file,
# to be later used to find which projects they belong to
zgrep -h -E "^requirements\.txt;" /da?_data/basemaps/gz/f2c*.s \
      >search.top_level_requirements.commits.out
zgrep -h -E "/requirements\.txt;" /da?_data/basemaps/gz/f2c*.s \
      >search.in_subdirectory_requirements.commits.out

# find the projects that those commits belong to,
# that is all projects that have 'requirements.txt' file
cat search.top_level_requirements.commits.out |
    cut -d ';' -f2 |
    ~/lookup/getValues c2P  >search.top_level_requirements.c2P.out
cat search.in_subdirectory_requirements.commits.out |
    cut -d ';' -f2 |
    ~/lookup/getValues c2P  >search.in_subdirectory_requirements.c2P.out


# extract set of individual projects,
# taking into account repetition, and multiple projects as result
#cat search.top_level_requirements.c2P.out | cut -d';' -f2- |
#    sort | uniq -c >search.top_level_requirements.c2P-P_counts.out
cat search.top_level_requirements.c2P.out |
    cut -d';' -f2- |
    sed 's/;/\n/' |
    sort | uniq -c  >search.top_level_requirements.c2P-P_counts.v2.out
cat search.in_subdirectory_requirements.c2P.out |
    cut -d';' -f2- |
    sed 's/;/\n/' |
    sort | uniq -c  >search.in_subdirectory_requirements.c2P-P_counts.v2.out

# find common projects, that is projects that have both top level 'requirements.txt'
# at some point of time, and 'requirememnts.txt' in a subdirectory at some possibly
# different point of time
cut -c9- search.top_level_requirements.c2P-P_counts.v2.out \
    >search.top_level_requirements.c2P-P_list.v2.out
cut -c9- search.in_subdirectory_requirements.c2P-P_counts.v2.out \
    >search.in_subdirectory_requirements.c2P-P_list.v2.out

comm -1 -2 \
     search.top_level_requirements.c2P-P_list.v2.out \
     search.in_subdirectory_requirements.c2P-P_list.v2.out \
     >search.comm_requirements.c2P-P_list.v2.out
