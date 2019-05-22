# -*- coding: utf-8 -*-

import click
import tempfile
import numpy as np
import pandas
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
# from keras.datasets import mnist, fashion_mnist

# from skluc.main.utils import read_matfile, logger, download_data
from pyqalm.utils import download_data, logger


def load_kddcup04bio():
    data_url = "http://cs.joensuu.fi/sipu/datasets/KDDCUP04Bio.txt"

    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        matfile_path = download_data(data_url, d_tmp)
        data = pandas.read_csv(matfile_path, delim_whitespace=True)

    return data.values

def load_census1990():
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"

    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        matfile_path = download_data(data_url, d_tmp)
        data = pandas.read_csv(matfile_path)

    return data.values[1:] # remove the `caseId` attribute

MAP_NAME_DATASET = {
    "kddcup": load_kddcup04bio,
    # "mnist": mnist.load_data,
    # "fashion_mnist": fashion_mnist.load_data,
    "census": load_census1990
}

MAP_NAME_CLASSES_PRESENCE = {
    "kddcup": False,
    # "mnist": True,
    # "fashion_mnist": True,
    "census": False,
}

def _download_all_data(output_dirpath):
    for key, _ in MAP_NAME_DATASET.items():
        _download_single_dataset(output_dirpath, key)

def _download_single_dataset(output_dirpath, dataname):
    if MAP_NAME_CLASSES_PRESENCE[dataname]:
        (x_train, y_train), (x_test, y_test) = MAP_NAME_DATASET[dataname]()
        map_savez = {"x_train": x_train,
                     "y_train": y_train,
                     "x_test": x_test,
                     "y_test": y_test
                     }
    else:
        X = MAP_NAME_DATASET[dataname]()
        map_savez = {"x_train": X}

    output_path = project_dir / output_dirpath / dataname
    logger.info(f"Save {dataname} to {output_path}")
    np.savez(output_path, **map_savez)

@click.command()
@click.argument('dataset', default="all")
@click.argument('output_dirpath', type=click.Path())
def main(output_dirpath, dataset):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    if dataset == "all":
        _download_all_data(output_dirpath)
    else:
        _download_single_dataset(output_dirpath, dataset)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
