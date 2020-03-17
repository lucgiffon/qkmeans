.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = QK-means
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
install:
	$(PYTHON_INTERPRETER) -m pip install . --ignore-installed

## Make Dataset
data: million
	$(PYTHON_INTERPRETER) code/data/make_dataset.py all data/external

data_no_million: covtype kddcup99 kddcup04 census plants coil20_32

covtype: data/external/covtype.dat data/external/covtype.lab
data/external/covtype.dat data/external/covtype.lab:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py covtype data/external

kddcup99: data/external/kddcup99.dat data/external/kddcup99.lab
data/external/kddcup99.dat data/external/kddcup99.lab:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py kddcup99 data/external

kddcup04: data/external/kddcup04.dat data/external/kddcup04.lab
data/external/kddcup04.dat data/external/kddcup04.lab:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py kddcup04 data/external

census: data/external/census.dat data/external/census.lab
data/external/census.dat data/external/census.lab:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py census data/external

plants: data/external/plants.npz
data/external/plants.npz:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py plants data/external

coil20_32: data/external/coil20_32.npz
data/external/coil20_32.npz:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py coil20_32 data/external

caltech256_50: data/external/caltech256_50.npz
data/external/caltech256_50.npz:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py caltech256_50 data/external

caltech256_32: data/external/caltech256_32.npz
data/external/caltech256_32.npz:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py caltech256_32 data/external

caltech256_28: data/external/caltech256_28.npz
data/external/caltech256_28.npz:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py caltech256_28 data/external

million: blobs_1_million blobs_10_million blobs_5_million blobs_2_million blobs_3_million

blobs_1_million: data/external/blobs_1_million.dat
data/external/blobs_1_million.dat:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py blobs_1_million data/external

blobs_2_million: data/external/blobs_2_million.dat
data/external/blobs_2_million.dat:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py blobs_2_million data/external

blobs_3_million: data/external/blobs_3_million.dat
data/external/blobs_3_million.dat:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py blobs_3_million data/external

blobs_5_million: data/external/blobs_5_million.dat
data/external/blobs_5_million.dat:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py blobs_5_million data/external

blobs_10_million: data/external/blobs_10_million.dat
data/external/blobs_10_million.dat:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py blobs_10_million data/external



## Delete all compiled Python files and data
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
#	rm -rf data/raw/*
	rm -rf data/external/*
	rm -rf data/processed/*
	rm -rf models/external/*

## Run unittests in project
test:
	$(PYTHON_INTERPRETER) -m unittest discover code/test




#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
