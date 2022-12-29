#!/bin/sh

# Re-created from playing with shell and shell API

# Assumes that 'lookup' repository was cloned into home directory
# git clone https://bitbucket.org/swsc/lookup ~/lookup

# find all commit messages that mention CVE,
# assuming here that the command below
# puts whole commit message in one line by replacing LF characters with \n
#
# NOTE that Perl script may require being in correct directory, and on correct host
# ssh da5                       # may be not requires
cd /da5_data/All.blobs        # may be not required
for i in {0..127}; do \
    ~/lookup/lstCmt.perl 9 $i; \
    done |
    grep -i -e 'CVE[ _+-]\?[0-9]\{4\}[ _+-]\?[0-9]\{4,\}' \
         >~/search.CVE_in_commit_message_ignore_case.lstCmt_9.out

