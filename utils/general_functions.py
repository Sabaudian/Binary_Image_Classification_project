# Import
import os
import gdown
import pandas as pd

from PIL import Image

# My import
from classifiers import build_mlp_model
from classifiers import build_cnn_model
from classifiers import build_vgg16_model
from classifiers import build_mobilenet_model


# ************************************* #
# ********* GENERAL FUNCTIONS ********* #
# ************************************* #


def download_models_save_from_drive(drive_url, root_dir):
    """
    Download from Google Drive the models folder, that contains the models saved from the previous run of the project.
    This will speed up the entire process.

    :param drive_url: Link to Google drive folder.
    :param root_dir: Root directory of the project, for saving the downloaded folder.
    :return: if successful, return the list of files downloaded.
    """
    # download model folder if not already present in the workspace
    if not os.path.exists("models"):
        # Checking folder
        check_input = input("\n> [SUGGESTED] The 'models' folder is not present in the workspace, "
                            "do you want to download it from Google Drive? [Y/N]: ")
        if check_input.upper() == "Y":
            # Download "models" folder
            gdown.download_folder(url=drive_url, quiet=False, output=root_dir)
    else:
        print("\n> The 'models' folder is in the workspace!")


# create a new directory
def makedir(dirpath):
    """
    Create a directory, given a path

    :param dirpath: directory location
    """
    # check if dir exists
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
        print("\n> Directory [{}] has been created successfully!\n".format(dirpath))


def count_files(file_path, extensions="jpg"):
    """
    Count the number of files with specified extensions in the specified directory.

    :param file_path: (str) The path to the directory for which file count is required.
    :param extensions: (list or None) List of file extensions to count. If None, count all files.

    :return: (int) The number of files with specified extensions in the specified directory.

    Example: count_files("/path/to/directory", extensions=["jpg", "png"]) -> 12
    """
    if extensions is None:
        extensions = ['']

    counter = 0
    with os.scandir(file_path) as entries:
        for entry in entries:
            if entry.is_file() and any(entry.name.lower().endswith(ext) for ext in extensions):
                counter += 1

    return counter


# Just a helper funtion
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

    def load_and_construct_df(dir_path, label):
        """
        Load image files from the specified directory and construct a dataframe.

        :param dir_path: The path to the directory containing image files.
        :param label: The label to assign to the images in the dataframe.

        :return: A list of dictionaries containing image paths and labels.
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
             - 'MLP': Multi-layer Perceptron model
             - 'CNN': Convolutional Neural Network model
             - 'VGG16': VGG16 model
             - 'MobileNet': MobileNet model
    """
    # models dictionary
    models = {"MLP": [], "CNN": [], "MobileNet": []}  # "VGG16": [],

    # Multi-layer Perceptron model
    mlp_model = build_mlp_model
    models.update({"MLP": mlp_model})

    # Convolutional Neural Network model
    cnn_model = build_cnn_model
    models.update({"CNN": cnn_model})

    # # VGG16 model
    # vgg16_model = build_vgg16_model
    # models.update({"VGG16": vgg16_model})

    # MobileNet model
    mobilenet_model = build_mobilenet_model
    models.update({"MobileNet": mobilenet_model})

    return models
