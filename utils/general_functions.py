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
    Example: img_file_0001, img_file_0002, ..., img_file_NNNN
    :param file: input file
    """
    # convert = lambda text: int(text) if text.isdigit() else text
    # alphanum_key = lambda key: [convert(c.replace("_", "")) for c in re.split("([0-9]+)", key)]
    file.sort(key=alphanum_key)


def count_files(file_path):
    """
    Simple counter to count the number of files inside a directory
    :param file_path: the path to file directory
    :return: int value
    """
    counter = len(fnmatch.filter(os.listdir(file_path), "*.*"))
    return counter


# Load data from a path
def load_file(dir_path):
    """
    Load file from a given path
    :param dir_path: directory path
    :return: sorted array of file
    """
    files = []
    for filename in os.listdir(dir_path):
        file = Image.open(os.path.join(dir_path, filename))
        files.append(file)
    return files


# define train and test dataframe from dataset
def define_dataframe(train_dir_path, test_dir_path):
    """
    Define two dataframe, one for the training set and the other for the test set
    :param train_dir_path: training directory path
    :param test_dir_path: test directory path
    :return: pandas.Dataframe (train_df, test_df)
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
    Get the data labels of the dataset in input
    :param tf_dataset: tf.Dataset.data object in input
    :return: label's array
    """
    # True labels array
    true_labels = []

    # Get labels
    for _, labels in tf_dataset:
        for label in labels:
            true_labels.append(label.numpy())
    # return labels
    return true_labels
