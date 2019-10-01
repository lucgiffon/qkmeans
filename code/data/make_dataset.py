# -*- coding: utf-8 -*-

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
# from keras.datasets import mnist, fashion_mnist
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
# from skluc.main.utils import read_matfile, logger, download_data
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from qkmeans.utils import download_data, logger
import cv2


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
        # tarfile_path = Path("/home/luc/Téléchargements/256_ObjectCategories.tar")

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
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/plants/plants.data"

    with tempfile.TemporaryDirectory() as d_tmp:
        logger.debug(f"Downloading file from url {data_url} to temporary directory {d_tmp}")
        file_path = download_data(data_url, d_tmp)

        with open(file_path, 'r', encoding="ISO-8859-15") as f:
            plants = f.readlines()

    set_plants_attributes = set()
    lst_plants = []
    for plant_line in plants:
        plant_line_no_name = [v.strip() for v in plant_line.split(',')[1:]]
        lst_plants.append(plant_line_no_name)
        set_plants_attributes.update(plant_line_no_name)

    arr_plants_attributes = np.array([v for v in set_plants_attributes])
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    onehot_encoder.fit(arr_plants_attributes.reshape(-1, 1))

    for i, plant_line_no_name in enumerate(lst_plants):
        plant_line_oh = np.sum(onehot_encoder.transform(np.array(plant_line_no_name).reshape(-1, 1)), axis=0)
        lst_plants[i] = plant_line_oh

    arr_lst_plants = np.array(lst_plants)

    return arr_lst_plants




MAP_NAME_DATASET = {
    "kddcup": load_kddcup04bio,
    # "mnist": mnist.load_data,
    # "fashion_mnist": fashion_mnist.load_data,
    "census": load_census1990,
    "plants": load_plants,
    "caltech256_50": lambda: load_caltech(50),
    "caltech256_32": lambda: load_caltech(32),
    "caltech256_28": lambda: load_caltech(28)
}

MAP_NAME_CLASSES_PRESENCE = {
    "kddcup": False,
    # "mnist": True,
    # "fashion_mnist": True,
    "census": False,
    "plants": False,
    "caltech256_50": True,
    "caltech256_32": True,
    "caltech256_28": True,
}

def _download_all_data(output_dirpath):
    for key, _ in MAP_NAME_DATASET.items():
        _download_single_dataset(output_dirpath, key)

def _download_single_dataset(output_dirpath, dataname):
    regex_million = re.compile(r"blobs_(\d+)_million")
    match = regex_million.match(dataname)
    if match:
        output_path_obs = project_dir / output_dirpath / (dataname + ".dat")
        output_path_labels = project_dir / output_dirpath / (dataname + ".lab")
        size_batch = 10000
        data_size = int(1e6) * int(match.group(1))
        nb_features = 2000
        nb_centers = 1000
        fp_obs = np.memmap(output_path_obs, dtype='float32', mode='w+', shape=(data_size, nb_features))
        fp_labels = np.memmap(output_path_labels, mode='w+', shape=(data_size,))

        total_nb_chunks = int(data_size // size_batch)
        logger.info("blobs_1_billion Data created in file: {}; labels stored in file: {}".format(output_path_obs, output_path_labels))
        logger.info("About to create 1 billion blobs dataset: {} chunks of {} examples dim {}. Total {} examples.".format(total_nb_chunks, size_batch, nb_features, data_size))
        init_centers = np.random.uniform(-10.0, 10.0, (nb_centers, nb_features))
        for i in range(total_nb_chunks):
            logger.info("Chunk {}/{}".format(i+1, total_nb_chunks))
            X, y = make_blobs(size_batch, n_features=nb_features, centers=init_centers, cluster_std=12.)
            fp_obs[i * size_batch:(i + 1) * size_batch] = X
            fp_labels[i * size_batch:(i + 1) * size_batch] = y

    else:
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
