#!/bin/sh

# Re-created from playing with shell and shell API

# Assumes that 'lookup' repository was cloned into home directory
# git clone https://bitbucket.org/swsc/lookup ~/lookup

# find all commit messages that mention CVE,
# and that also touch given pathname
# two different ways

zgrep -F \
      -f ~/search.CVE_in_commit_message.commit_id.out \
      ~/search.pom_xml_anywhere.c2P.out.gz \
      >search.CVE_in_commit_message-AND-pom_xml_anywhere.c2P.out

# note that this specific example failed with errors:
#
# Argument "" isn't numeric in sprintf at /home/jnareb/lookup/woc.pm line 388, <STDIN> line 41252525.
# Argument "" isn't numeric in sprintf at /home/jnareb/lookup/woc.pm line 390, <STDIN> line 41252525.
# Use of uninitialized value $taz in substitution (s///) at /home/jnareb/lookup/woc.pm line 393, <STDIN> line 41252525.
# Use of uninitialized value $tcz in substitution (s///) at /home/jnareb/lookup/woc.pm line 394, <STDIN> line 41252525.
# Use of uninitialized value $taz in concatenation (.) or string at /home/jnareb/lookup/woc.pm line 402, <STDIN> line 41252525.
# Use of uninitialized value $tcz in concatenation (.) or string at /home/jnareb/lookup/woc.pm line 402, <STDIN> line 41252525.
#
zcat ~/search.pom_xml.anywhere.c2P.out.gz |
    cut -d';' -f1 | ~/lookup/showCnt commit 9 |
    grep -e 'CVE[ _+-]\?[0-9]\{4\}[ _+-]\?[0-9]\{4,\}' \
         >search.pom_xml_anywhere-AND-CVE_in_commit_message.commit_9.out
