# Import
import os
import re
import fnmatch
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score


# ************************************* #
# ********* GENERAL FUNCTIONS ********* #
# ************************************* #

# create a new directory
def makedir(dirpath):
    """
    Create a directory, given a path

    :param dirpath: directory location
    """
    # check if dir exists
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print("\n> Directory [{}] has been created successfully!\n".format(dirpath))


# Return text as int if possible, or "text" unchanged
def convert(data):
    """
    Convert text into integer if possible

    :param data: input data

    :return: integer data, else data unchanged
    """
    return int(data) if data.isdigit() else data


# Turn a string into a list of strings and number chunks.
def alphanum_key(data):
    """
    Transform a string into a list of strings and number chunks

    :param data: input data

    :return: transformed data string
    """
    return [convert(c.replace("_", "")) for c in re.split("([0-9]+)", data)]


# Sort filenames as expected
def sort_files(file):
    """
    Sorting file as expected by human observer

    :param file: input file

    Example: img_file_0001, img_file_0002, ..., img_file_NNNN
    """
    # convert = lambda text: int(text) if text.isdigit() else text
    # alphanum_key = lambda key: [convert(c.replace("_", "")) for c in re.split("([0-9]+)", key)]
    file.sort(key=alphanum_key)


def count_files(file_path):
    """
    Count the number of files with extensions in the specified directory.

    :param file_path: (str) The path to the directory for which file count is required.

    :return: (int) The number of files with extensions in the specified directory.

    Example: count_files("/path/to/directory") -> 12
    """
    counter = len(fnmatch.filter(os.listdir(file_path), "*.*"))

    return counter


# Load data from a path
def load_file(dir_path):
    """
    Load file from a given path.

    :param dir_path: Directory path.

    :return: A sorted array of file.
    """
    files = []
    for filename in os.listdir(dir_path):
        file = Image.open(os.path.join(dir_path, filename))
        files.append(file)
    return files


# define train and test dataframe from dataset
def define_dataframe(train_dir_path, test_dir_path):
    """
    Define two dataframe, one for the training set and the other for the test set.

    :param train_dir_path: Training directory path.
    :param test_dir_path: Test directory path.

    :return: Pandas.Dataframe (train_df, test_df)
    """
    # train data
    train_chihuahua = load_file(dir_path=os.path.join(train_dir_path, "chihuahua"))
    train_muffin = load_file(dir_path=os.path.join(train_dir_path, "muffin"))

    # define dataframe for the train set
    train_data = {
        "image": train_muffin + train_chihuahua,
        "label": ["chihuahua"] * len(train_chihuahua) + ["muffin"] * len(train_muffin)
    }
    train_df = pd.DataFrame(train_data)
    # train_df.to_csv(data_folder + "/train_data.csv", index=False)

    # test data
    test_chihuahua = load_file(dir_path=os.path.join(test_dir_path, "chihuahua"))
    test_muffin = load_file(dir_path=os.path.join(test_dir_path, "muffin"))

    # define dataframe for the test set
    test_data = {
        "image": test_muffin + test_chihuahua,
        "label": ["chihuahua"] * len(test_chihuahua) + ["muffin"] * len(test_muffin)
    }
    test_df = pd.DataFrame(test_data)

    # # Encode data as: Muffin = one | Chihuahua = zero
    # encoder = preprocessing.OrdinalEncoder()
    # train_df["label"] = encoder.fit_transform(train_df[["label"]])
    # test_df["label"] = encoder.fit_transform(test_df[["label"]])

    return train_df, test_df


def get_labels_from_dataset(tf_dataset):
    """
    Extract true labels from a TensorFlow dataset.

    :param tf_dataset: (tf.data.Dataset) The TensorFlow dataset containing data and labels.

    :return: (list) A list of true labels extracted from the dataset.
    """
    # True labels array
    true_labels = []

    # Get labels
    for _, labels in tf_dataset:
        for label in labels:
            true_labels.append(label.numpy())
    # return labels
    return true_labels
