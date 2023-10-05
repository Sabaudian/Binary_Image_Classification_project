# Import
import os
import re
import shutil
import numpy as np
import pandas as pd

from PIL import Image


# ------------------------------------- #
# --------- GENERAL FUNCTIONS --------- #
# ------------------------------------- #

# create a new directory
def makedir(dirpath):
    """
    Create a directory, given a path
    :param dirpath: directory location
    """
    # check if dir exists
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print("\n> Directory [{}] has been crated successfully!\n".format(dirpath))


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


# copy file to a new path
def copy_file_to_new_location(old_file_path, new_file_path):
    """
    Copy a file to a new given location
    :param old_file_path: old location of the file
    :param new_file_path: where the file needs to be placed
    """
    # Copy it only if the file does not exist
    if not os.path.exists(new_file_path):
        # Copy file into new location
        shutil.copyfile(old_file_path, new_file_path)
        print("> [{}] ––> [{}]".format(old_file_path, new_file_path))


# resize file and change mode
def resize_and_change_color_mode(image, original_color_mode, new_color_mode, original_size, new_img_size, new_file_path,
                                 new_filename):
    """

    :param image: input image file
    :param original_color_mode: original file color mode (ex.: RGB, Grayscale, ...)
    :param new_color_mode: new color mode to set up
    :param original_size: original file size
    :param new_img_size: new file size to set up (128x128)
    :param new_file_path: file path
    :param new_filename: file name
    :return:
    """
    # Convert image in dataset into grayscale
    if (original_color_mode != new_color_mode) or (original_size != new_img_size):
        # Size and mode conversion
        grayscale_image = image.convert(new_color_mode)
        resized_grayscale_image = grayscale_image.resize(new_img_size)
        # Check modified info
        new_mode = grayscale_image.mode
        new_size = resized_grayscale_image.size
        # Save modified image to filepath
        resized_grayscale_image.save(new_file_path)
        # Print messages
        print(">> [{}]: ".format(new_filename)
              + "[size:{} | mode:{}]".format(original_size, original_color_mode)
              + " --> "
              + "[size:{} | mode:{}]".format(new_size, new_mode))


# transform an image file into arrays
def image_to_array(files):
    """
    Convert image file into Numpy.Array
    :param files: input files
    :return: Numpy.Array
    """
    return np.stack([np.array(file) for file in files])


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
