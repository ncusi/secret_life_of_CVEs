# -*- mode: makefile; coding: utf-8-unix

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

REGEXP="__PLACEHOLDER__"
NICE=nice

# list all package managers of interest
PACKAGE_MANAGERS=conda cargo ivy go_mod npm

final_name=search.$(name)_anywhere.c2P-P_counts.v2.out.gz
FINAL_FILES=$(foreach name,$(PACKAGE_MANAGERS),$(final_name))

intermediate_name=search.$(name)_anywhere.c2P.out.gz
C2P_FILES=$(foreach name,$(PACKAGE_MANAGERS),$(intermediate_name))

# default target
all :: $(C2P_FILES) $(FINAL_FILES)

# don't remove intermediate files, i.e. *.c2P.out.gz
# https://www.gnu.org/software/make/manual/make.html#Chained-Rules
# NOTE: this doesn't seem to work
.NOTINTERMEDIATE :

# main configuration
search.conda_anywhere.f2c.out.gz : REGEXP="(^|/)info/index\.json;"
search.cargo_anywhere.f2c.out.gz : REGEXP="(^|/)Cargo\.toml;"
search.ivy_anywhere.f2c.out.gz : REGEXP="(^|/)ivy\.xml;"
search.go_mod_anywhere.f2c.out.gz : REGEXP="(^|/)go\.mod;"
search.npm_anywhere.f2c.out.gz : REGEXP="(^|/)package\.json;"


# chain of pattern rules
#%.f2c.out.gz:
#	$(NICE) zgrep -h -E $(REGEXP) /da?_data/basemaps/gz/f2c*.s | \
#	gzip -c -q - >"$@"
%.f2c.out.gz:
	$(NICE) zgrep -h -E $(REGEXP) /da?_data/basemaps/gz/f2c*.s \
	>"$(basename $@)"
	$(NICE) gzip -q "$(basename $@)"

%.c2P.out.gz: %.f2c.out.gz
	$(NICE) zcat "$<" | \
	cut -d ';' -f2 | \
	$(NICE) ~/lookup/getValues c2P >"$(basename $@)"
	$(NICE) gzip -q "$(basename $@)"

%.c2P-P_counts.v2.out.gz: %.c2P.out.gz
	$(NICE) zcat "$<" | \
	cut -d ';' -f2- | \
	sed 's/;/\n/' | \
	sort | uniq -c  >"$(basename $@)"
	$(NICE) gzip -q "$(basename $@)"
	date
