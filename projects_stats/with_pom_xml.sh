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

BASENAME="pom_xml.anywhere"
REGEXP="(^|/)pom\.xml;"

#zgrep -c -E "$REGEXP" /da?_data/basemaps/gz/f2c*.s \
#      >"search.${BASENAME}.count.out" &&
zgrep -h -E "$REGEXP" /da?_data/basemaps/gz/f2c*.s \
      >"search.$BASENAME.f2c.out" &&
cat "search.$BASENAME.f2c.out" |
    cut -d ';' -f2 |
    ~/lookup/getValues c2P  >"search.$BASENAME.c2P.out" &&
cat "search.$BASENAME.c2P.out" |
    cut -d';' -f2- |
    sed 's/;/\n/' |
    sort | uniq -c  >"search.$BASENAME.c2P-P_counts.v2.out"
