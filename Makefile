.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = deepstrom_network
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
	$(PYTHON_INTERPRETER) -m pip install -e code/

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py all data/external

cifar100: data/external/cifar100fine.npz
data/external/cifar100fine.npz:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py cifar100 data/external

cifar10: data/external/cifar10.npz
data/external/cifar10.npz:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py cifar10 data/external

mnist: data/external/mnist.npz
data/external/mnist.npz:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py mnist data/external

svhn: data/external/svhn.npz
data/external/svhn.npz:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py svhn data/external


## Make Transforms
transforms: transform_vgg19 transform_lenet

transform_vgg19: transform_vgg19_cifar10_block5_pool transform_vgg19_svhn_block5_pool

transform_lenet: transform_lenet_mnist_conv_pool_2

transform_lenet_mnist_conv_pool_2: data/processed/lenet/conv_pool_2/mnist.npz
data/processed/lenet/conv_pool_2/mnist.npz: mnist lenet_mnist
	$(PYTHON_INTERPRETER) src/features/build_features.py mnist lenet conv_pool_2 mnist data/processed

## Download models

models: vgg19 lenet

vgg19:
	$(PYTHON_INTERPRETER) src/models/download_model.py vgg19 all models/external

vgg19_cifar10: models/external/vgg19/cifar10/1544802301.9379897_vgg19_Cifar10Dataset.h5
models/external/vgg19/cifar10/1544802301.9379897_vgg19_Cifar10Dataset.h5:
	$(PYTHON_INTERPRETER) src/models/download_model.py vgg19 cifar10 models/external


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
#	rm -rf data/raw/*
	rm -rf data/external/*
	rm -rf data/processed/*
	rm -rf models/external/*

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



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
