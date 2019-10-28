# -*- coding: utf-8 -*-
"""
This file contains functions for downloading and storing dataset (also perform basic preprocessing on datasets)
"""

import click
import os
import operator
import tempfile
import numpy as np
import pandas
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import tarfile
import re
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, fetch_kddcup99, fetch_covtype
from sklearn.model_selection import train_test_split
from qkmeans.utils import download_data, logger
import cv2
import pandas as pd


def load_kddcup04bio_no_classif():
    data_url = "http://cs.joensuu.fi/sipu/datasets/KDDCUP04Bio.txt"

    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        matfile_path = download_data(data_url, d_tmp)
        data = pandas.read_csv(matfile_path, delim_whitespace=True)

    return data.values

def load_kddcup04bio():
    input_path = project_dir / "data/raw" / "data_kddcup04" / "bio_train.dat"
    data = pandas.read_csv(input_path, header=None, delim_whitespace=True)
    X = data.values[:, 3:]
    y = data.values[:, 2]
    return X, y

def load_kddcup99():
    X, y = fetch_kddcup99(shuffle=True, return_X_y=True)
    df_X = pd.DataFrame(X)
    X = pd.get_dummies(df_X, columns=[1, 2, 3], prefix=['protocol_type', "service", "flag"]).values
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y.reshape(-1, 1))
    return X, y

def load_covtype():
    X, y = fetch_covtype(shuffle=True, return_X_y=True)
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y.reshape(-1, 1))
    return X, y


def load_census1990():
    """
    Meek, Thiesson, and Heckerman (2001), "The Learning Curve Method Applied to Clustering", to appear in The Journal of Machine Learning Research.

    Number of clusters: 25, 50, 100
    :return:
    """
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"

    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        matfile_path = download_data(data_url, d_tmp)
        data = pandas.read_csv(matfile_path)

    return data.values[:, 1:], None # remove the `caseId` attribute

def crop_center(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def load_caltech(final_size):
    data_url = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"

    lst_images = []
    lst_classes_idx = []


    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        tarfile_path = Path(download_data(data_url, d_tmp))

        dir_path = Path(d_tmp)

        tf = tarfile.open(tarfile_path)
        tf.extractall(dir_path / "caltech256")
        tf.close()
        for root, dirs, files in os.walk(dir_path / "caltech256"):
            print(root)
            label_class = root.split("/") [-1]
            splitted_label_class = label_class.split(".")
            if splitted_label_class[-1] == "clutter":
                continue
            if len(splitted_label_class) > 1:
                label_idx = int(splitted_label_class[0])
            else:
                continue

            for file in files:
                path_img_file = Path(root) / file
                try:
                    img = plt.imread(path_img_file)
                except:
                    continue
                aspect_ratio = max(final_size / img.shape[0], final_size / img.shape[1])
                new_img = cv2.resize(img, dsize=(0,0), fx=aspect_ratio, fy=aspect_ratio)
                new_img = crop_center(new_img, (final_size, final_size, 3))

                if new_img.shape == (final_size, final_size):
                    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)


                lst_images.append(new_img.flatten())
                lst_classes_idx.append(label_idx)

        X = np.vstack(lst_images)
        y = np.array(lst_classes_idx)

        print(X.shape)
        print(y.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42, stratify=y)

    return (X_train, y_train), (X_test, y_test)

def load_plants():
    """
    USDA, NRCS. 2008. The PLANTS Database ([Web Link], 31 December 2008). National Plant Data Center, Baton Rouge, LA 70874-4490 USA.

    :return:
    """
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/plants/plants.data"

    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        file_path = download_data(data_url, d_tmp)

        with open(file_path, 'r', encoding="ISO-8859-15") as f:
            plants = f.readlines()

    # get all the features in a set
    set_plants_attributes = set()
    lst_plants = []
    for plant_line in plants:
        plant_line_no_name = [v.strip() for v in plant_line.split(',')[1:]]
        lst_plants.append(plant_line_no_name)
        set_plants_attributes.update(plant_line_no_name)

    # give a code to each feature in a 1-hot fashion
    arr_plants_attributes = np.array([v for v in set_plants_attributes])
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    onehot_encoder.fit(arr_plants_attributes.reshape(-1, 1))

    # transform each plant with their code
    for i, plant_line_no_name in enumerate(lst_plants):
        plant_line_oh = np.sum(onehot_encoder.transform(np.array(plant_line_no_name).reshape(-1, 1)), axis=0)
        lst_plants[i] = plant_line_oh

    arr_lst_plants = np.array(lst_plants)

    return arr_lst_plants

def generator_blobs_data(data_size, size_batch, nb_features, nb_centers):
    total_nb_chunks = int(data_size // size_batch)
    init_centers = np.random.uniform(-10.0, 10.0, (nb_centers, nb_features))
    for i in range(total_nb_chunks):
        logger.info("Chunk {}/{}".format(i + 1, total_nb_chunks))
        X, y = make_blobs(size_batch, n_features=nb_features, centers=init_centers, cluster_std=12.)
        yield X, y

def generator_data(data_load_func, size_batch=10000):
    X, y = data_load_func()
    data_size = X.shape[0]
    total_nb_chunks = int(data_size // size_batch)
    remaining = int(data_size % size_batch)
    for i in range(total_nb_chunks):
        logger.info("Chunk {}/{}".format(i + 1, total_nb_chunks))
        if y is None:
            yield X[i*size_batch: (i+1)*size_batch], None
        else:
            yield X[i * size_batch: (i + 1) * size_batch], y[i * size_batch: (i + 1) * size_batch]
    if remaining > 0:
        if y is None:
            yield X[(i+1)*size_batch: ], None
        else:
            yield X[(i + 1) * size_batch:], y[(i + 1) * size_batch:]

def save_memmap_data(output_dirpath, dataname, data_size, nb_features, Xy_gen):
    output_path_obs = project_dir / output_dirpath / (dataname + ".dat")
    output_path_labels = project_dir / output_dirpath / (dataname + ".lab")
    fp_obs = np.memmap(output_path_obs, dtype='float32', mode='w+', shape=(data_size, nb_features))
    fp_labels = np.memmap(output_path_labels, mode='w+', shape=(data_size,))

    logger.info("{} Data will be created in file: {}; labels stored in file: {}".format(dataname, output_path_obs, output_path_labels))
    logger.info("About to create {}: Total {} examples.".format(dataname, data_size))

    curr_idx = 0
    for i, (batch_X, batch_y) in enumerate(Xy_gen):
        curr_batch_size = batch_X.shape[0]
        fp_obs[curr_idx:curr_idx + curr_batch_size] = batch_X
        if batch_y is not None:
            fp_labels[curr_idx:curr_idx + curr_batch_size] = batch_y
        curr_idx += curr_batch_size

    if batch_y is None:
        os.remove(str(output_path_labels))

MAP_NAME_DATASET_DD = {
    "kddcup04": lambda p_output_dirpath : save_memmap_data(p_output_dirpath, "kddcup04", 145751, 74, generator_data(load_kddcup04bio)),
    "kddcup99": lambda p_output_dirpath : save_memmap_data(p_output_dirpath, "kddcup99", 494021, 118, generator_data(load_kddcup99)),
    "census": lambda p_output_dirpath : save_memmap_data(p_output_dirpath, "census", 2458285, 68, generator_data(load_census1990)),
    "covtype": lambda p_output_dirpath : save_memmap_data(p_output_dirpath, "covtype", 581012, 54, generator_data(load_covtype))
}

MAP_NAME_DATASET_RAM = {

    "plants": load_plants,
    "caltech256_50": lambda: load_caltech(50),
    "caltech256_32": lambda: load_caltech(32),
    "caltech256_28": lambda: load_caltech(28)
}

MAP_NAME_CLASSES_PRESENCE_RAM = {
    "plants": False,
    "caltech256_50": True,
    "caltech256_32": True,
    "caltech256_28": True,
}

def _download_all_data(output_dirpath):
    for key in list(MAP_NAME_DATASET_RAM.keys()) + list(MAP_NAME_DATASET_DD.keys()):
        _download_single_dataset(output_dirpath, key)



def _download_single_dataset(output_dirpath, dataname):
    regex_million = re.compile(r"blobs_(\d+)_million")
    match = regex_million.match(dataname)
    if match:
        size_batch = 10000
        data_size = int(1e6) * int(match.group(1))
        nb_features = 2000
        nb_centers = 1000

        save_memmap_data(output_dirpath, dataname, data_size, nb_features, generator_blobs_data(data_size, size_batch, nb_features, nb_centers))

    else:
        if dataname in MAP_NAME_DATASET_DD.keys():
            MAP_NAME_DATASET_DD[dataname](output_dirpath)
            return

        elif MAP_NAME_CLASSES_PRESENCE_RAM[dataname]:
            (x_train, y_train), (x_test, y_test) = MAP_NAME_DATASET_RAM[dataname]()
            map_savez = {"x_train": x_train,
                         "y_train": y_train,
                         "x_test": x_test,
                         "y_test": y_test
                         }
        else:
            X = MAP_NAME_DATASET_RAM[dataname]()
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
