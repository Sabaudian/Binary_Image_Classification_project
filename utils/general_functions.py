# Import
import os

import kaggle
import pandas as pd

# My import
import constants as const

from classifiers import build_mlp_model
from classifiers import build_cnn_model
from classifiers import build_mobilenet_model


# ************************************* #
# ********* GENERAL FUNCTIONS ********* #
# ************************************* #


# Download dataset from Kaggle website
def download_dataset_from_kaggle(dataset_id, dataset_path):
    """
    Download the muffin-vs-chihuahua-image-classification Dataset using Kaggle module

    :param dataset_id: identify the dataset to download.
        Format: dataset_owner_name/dataset_name
    :param dataset_path: location to save the dataset

    :return: None
    """

    # Download the dataset if not exist in the workplace
    if not os.path.exists(dataset_path):
        print("\n> Download the dataset from Kaggle...")
        # Download dataset and unzip it
        kaggle.api.dataset_download_files(dataset=dataset_id, path=dataset_path, quiet=False, unzip=True)


# Create a new directory
def makedir(dirpath):
    """
    Create a directory, given a path

    :param dirpath: directory location
    """

    # check if dir exists
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
        print("\n> Directory [{}] has been created successfully!".format(dirpath))


# Create a series of folder used for the project
def define_workspace_folders():
    """
    Create a series of folders if not already present in the workspace:
        - data: to store table '.csv' to collect data
        - models: to store the models weights and pre-trained models saves.
        - plot: to store the graphs generated to describe the processes.
    """
    # create data folder
    makedir(const.DATA_PATH)

    # create models folder
    makedir(const.MODELS_PATH)

    # create plot folder
    makedir(const.PLOT_FOLDER)


# Count the number of files
def count_files(file_path, extensions="jpg"):
    """
    Count the number of files with specified extensions in the specified directory.

    Example: count_files("/path/to/directory", extensions=["jpg", "png"]) -> 12

    :param file_path: (str) The path to the directory for which file count is required.
    :param extensions: (list or None) List of file extensions to count. If None, count all files.

    :return: (int) The number of files with specified extensions in the specified directory.
    """

    if extensions is None:
        extensions = ['']

    counter = 0
    with os.scandir(file_path) as entries:
        for entry in entries:
            if entry.is_file() and any(entry.name.lower().endswith(ext) for ext in extensions):
                counter += 1

    return counter


# Just a helper function
def print_file_counts(dataset_path):
    """
    A helper function t pint information about the number of files inside the directory.

    :param dataset_path: The path to training data
    """

    count_train_chihuahua = count_files(file_path=os.path.join(dataset_path, "train/chihuahua"))
    count_train_muffin = count_files(file_path=os.path.join(dataset_path, "train/muffin"))
    count_test_chihuahua = count_files(file_path=os.path.join(dataset_path, "test/chihuahua"))
    count_test_muffin = count_files(file_path=os.path.join(dataset_path, "test/muffin"))

    tot_number_file = count_train_chihuahua + count_train_muffin + count_test_chihuahua + count_test_muffin
    print("- Total Number of file: {}\n".format(tot_number_file) +
          "- Number of file in train/chihuahua: {}\n".format(count_train_chihuahua) +
          "- Number of file in train/muffin: {}\n".format(count_train_muffin) +
          "- Number of file in test/chihuahua: {}\n".format(count_test_chihuahua) +
          "- Number of file in test/muffin: {}\n".format(count_test_muffin))


# Define train and test dataframe
def define_dataframe(train_dir_path, test_dir_path):
    """
    Define two dataframe, one for the training set and the other for the test set.

    :param train_dir_path: Training directory path.
    :param test_dir_path: Test directory path.

    :returns: Pandas.Dataframe (train_df, test_df)
    """

    def load_and_construct_df(dir_path, label):
        """
        Load image files from the specified directory and construct a dataframe.

        :param dir_path: The path to the directory containing image files.
        :param label: The label to assign to the images in the dataframe.

        :returns: A list of dictionaries containing image paths and labels.
        """

        data = []
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    data.append({"image": entry.path, "label": label})
        return data

    # Define dataframe for the train set
    train_data = (
            load_and_construct_df(dir_path=os.path.join(train_dir_path, "chihuahua"), label="chihuahua") +
            load_and_construct_df(dir_path=os.path.join(train_dir_path, "muffin"), label="muffin")
    )
    train_df = pd.DataFrame(train_data)

    # Define dataframe for the test set
    test_data = (
            load_and_construct_df(dir_path=os.path.join(test_dir_path, "chihuahua"), label="chihuahua") +
            load_and_construct_df(dir_path=os.path.join(test_dir_path, "muffin"), label="muffin")
    )
    test_df = pd.DataFrame(test_data)

    return train_df, test_df


# Organize models in a dictionary
def get_classifier():
    """
    Retrieves a dictionary of pre-built classification models.

    :return: A dictionary containing the following classification models:
             - 'MLP': Multi-layer Perceptron model.
             - 'CNN': Convolutional Neural Network model.
             - 'MobileNet': MobileNet model.
    """

    # models dictionary
    models = {"MLP": [], "CNN": [], "MobileNet": []}

    # Multilayer Perceptron model
    mlp_model = build_mlp_model
    models.update({"MLP": mlp_model})

    # Convolutional Neural Network model
    cnn_model = build_cnn_model
    models.update({"CNN": cnn_model})

    # MobileNet model
    mobilenet_model = build_mobilenet_model
    models.update({"MobileNet": mobilenet_model})

    return models
