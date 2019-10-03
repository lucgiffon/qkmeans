# QuicK-means (QK-means) research project

This repository aims at providing base scripts to play with the qk-means algorithm.
You can find:
 
- easily installable package which provide all necessary functions to
play with qkmeans or the palm4msa algorithm;
- a minimum working exemple on how to use our qkmeans algorithm;
- the history and evolution of our experiment script until its final version;
- the history of tested parameters for this experiment script;
- the utility functions necessary to download or create the datasets 
used in experiments;
- we also provide the source to compile our report with latex;
- the vizualization scripts used to generate figures.

__Note:__ This is a view of the developpement repository. It doesn't contain Work In progress.
We have tried to make it as clear as possible in order to help the reader to reproduce results.
If you have any question or need any clearance on the repository content, do not hesitate to 
raise an issue.

__Note 2:__ Everything isn't clean and tidy everywhere. We have let the code in the `if __name__ == "__main__"` blocks
in each module so that you can inspire yourself from them but they are not guaranteed to work. There is certainly
a lot of non-working code but we choose to remain transparent in case it would be of any use to any one. If you 
have any question, please raise an issue.

## Installation

To install the package, simply go in the `code` directory and install using pip:

	make install
	
To verify that everything is working, you can use the test rule from the root directory:

	make test

## Minimum working example

Under the `scripts/examples` directory, you will find 2 minimum working examples:

- `example_hierarchical_palm4msa.py` shows how the heirarchical-palm4msa function can be called
- `example_qkmeans.py` show how the qkmeans function can be called


## Experiments

Running experiments can be done in few steps:

- Prepare data
- Prepare parameters
- Launch experiment
- Vizualize results

See the details of those steps in the following subsections.

__Note:__ We point to the last working versions of the experiments/parameters/vizualizations but you can look
at the other if you find it interesting. Note that there is some kind of match between the names of the experiments scripts, 
the parameter files and vizualization scripts.

### Preparing data

Look inside the `Makefile` to see what rules you can call for what dataset. You can also
create all datasets at once using the following command but this will take a lot of time and 
a lot of space on your disk. Be carefull then. The data will be stored under the `data/external`
directory.

	make data

If you want to remove all data, you can do (this can take some time too)

	make clean

### Prepare parameters

In the `parameters/08/aaai` directory, you will find some files called `lazyfile_*.yml`. These
files describe combination of parameters to test. You can interpret them using the `lazygrid`
command from the [scikit-luc](https://pypi.org/project/scikit-luc/) package 
(which should be isntalled if you have installed `qkmeans`). To print the command lines produced
by a lazyfile, do for example:

	cd parameters/08/aaai
	lazygrid -l lazyfile_qmeans_analysis_caltech_decoda2.yml

### Launch experiments

You can use the scripts `code/scripts/2019/08/3_4_qmeans_minibatch_hierarchical_init.py` with any
combination of the previously generated parameters. Find the usage help on top of the script, in the dosctring.

### Vizualize results

You can try to adapt the code in `code/visualization/2019/08/aaai_conference` to vizualize
the results. This code won't work in your case because of invalids paths but maybe you
can take inspiration from it. 

## Report

Latex makes it difficult to create a `make` rule for its compilation so you'll have to do it by yourself.
To compile the paper as pdf go to `reports/aaai_2020` then use `pdflatex` and `bibtex`:

	cd reports/aaai_2020
	pdflatex aaai2020_qmeans.tex -synctex=1 -interaction=nonstopmode
	bibtex aai2020_qmeans.aux
	pdflatex aaai2020_qmeans.tex -synctex=1 -interaction=nonstopmode
	pdflatex aaai2020_qmeans.tex -synctex=1 -interaction=nonstopmode


## References

For the implementation of PALM4MSA, we used the description from the PALM4MSA paper and we helped ourselves with their
open source matplotlib implementation available here: https://faust.inria.fr/ . Note that our implementation maybe sub-optimal
compared to their and you can get in touch with them to use their python version.

The Palm4MSA paper:
Le Magoarou, Luc, et Remi Gribonval. « Flexible Multilayer Sparse Approximations of Matrices and Applications ». IEEE Journal of Selected Topics in Signal Processing 10, nᵒ 4 (juin 2016): 688‑700. https://doi.org/10.1109/JSTSP.2016.2543461.

QKmeans paper:
Luc Giffon, Valentin Emiya, Liva Ralaivola, Hachem Kadri. QuicK-means: Acceleration of K-means by learning a fast transform. 2019. ⟨hal-02174845v2⟩
