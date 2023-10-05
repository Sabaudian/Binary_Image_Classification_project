# Import
import os
import pandas as pd
import tensorflow as tf

from keras import layers
from keras.models import Sequential
from PIL import Image, UnidentifiedImageError

# My import
import plot_functions
from utils.general_functions import sort_files
from utils.general_functions import define_dataframe


# Check dataset: filter out possible corrupted files.
def scanning_dataset(dataset_path):
    """
    Checking dataset,
    verifying the presence of corrupted file and collect metadata
    :param dataset_path: the path to the dataset
    """
    # Initialize
    bad_file = []  # to store corrupted file
    count_bad_file = 0  # to store the number of corrupted files

    # Loop through all dataset subfolders
    for dirpath, _, filenames in os.walk(dataset_path):

        # Ensure we're processing a sub-folder level
        if dirpath is not dataset_path:

            # Properly sorted file as presented in the original dataset
            sort_files(filenames)

            # Loop through all files
            for filename in filenames:
                # Check the file extension
                if filename.endswith("jpg"):
                    # Get the file path
                    file_path = os.path.join(dirpath, filename)
                    # Get the file extension
                    try:
                        with Image.open(file_path) as image:
                            image.verify()
                    except UnidentifiedImageError:
                        bad_file.append(file_path)
                        count_bad_file += 1
                        print("\n> there are {} corrupted files: {}".format(count_bad_file, bad_file))


# Check dataset: collect metadata information.
# Save metadata of the dataset's files
def collect_metadata(dataset_path):
    """
    Collect metadata from the files inside the dataset located at the given path
    :param dataset_path: the path to the dataset
    :return: metadata dictionary
    """
    # Initialize dictionary to collect metadata
    metadata = {}

    # Loop through all dataset subfolders
    for dirpath, _, filenames in os.walk(dataset_path):

        # Ensure we're processing a sub-folder level
        if dirpath is not dataset_path:

            # Properly sorted file as presented in the original dataset
            sort_files(filenames)

            # Loop through all files
            for filename in filenames:
                # Check the file extension
                if filename.endswith("jpg"):
                    # Get the file path
                    file_path = os.path.join(dirpath, filename)
                    # Open image
                    image = Image.open(file_path)
                    # size
                    metadata[file_path] = image.size
    return metadata


# Displaying data as a clear plot
def view_data(train_dir_path, test_dir_path, show_histogram=True):
    """
    Display the amount of data per class of sets: train and test
    :param train_dir_path: the path to training data
    :param test_dir_path: the path to test data
    :param show_histogram: chooses whether to show plot
    """
    # function that defines dataset
    train_df, test_df = define_dataframe(train_dir_path=train_dir_path,
                                         test_dir_path=test_dir_path)

    if show_histogram:
        # plot histogram
        plot_functions.plot_img_class_histogram(train_data=train_df, test_data=test_df, show_on_screen=True,
                                                store_in_folder=True)


def checking_dataset(dataset_path, train_dir_path, test_dir_path):
    """
    Preliminary check on dataset
    :param dataset_path: the path to the dataset
    :param train_dir_path: the path to training data
    :param test_dir_path: the path to test data
    """
    print("\n> CHECKING DATASET: ")
    # check for corrupted file
    scanning_dataset(dataset_path=dataset_path)

    # structure of the metadata
    metadata = collect_metadata(dataset_path=dataset_path)
    metadata_df = ((pd.DataFrame.from_dict(metadata, orient="index").reset_index()
                    .rename(columns={"index": "image", 0: "width", 1: "height"})))
    # Compute aspect ratio
    metadata_df["aspect_ratio"] = round(metadata_df["width"] / metadata_df["height"], 2)
    # Add label column
    metadata_df["label"] = metadata_df.apply(lambda row: row.iloc[0].rsplit("/")[-2], axis=1)  # label
    # show metadata information
    print("\n> Metadata:\n {} \n".format(metadata_df.describe()))

    # displaying histogram
    view_data(train_dir_path, test_dir_path, show_histogram=True)
    print("\n> DATASET CHECK COMPLETE!")


# load data into keras dataset
def load_dataset(train_data_dir, test_data_dir, batch_size, img_size):
    """
    Load data as a Keras dataset and perform tuning and rescaling processes
    :param train_data_dir: the path to the training data
    :param test_data_dir: the path to the test data
    :param batch_size: defines batch size
    :param img_size: define the size of the images -> (size, size)
    :return: tf.Dataset.data object
    """
    # train dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=train_data_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        subset="training",
        smart_resize=True
    )

    # validation dataset
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=train_data_dir,
        validation_split=0.2,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        subset="validation",
        smart_resize=True
    )

    # test dataset
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=test_data_dir,
        color_mode="rgb",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        smart_resize=True
    )

    def resize_and_scaling(keras_ds, shuffle=False):
        """
        Resize and Scaling dataset
        :param keras_ds: tf.Dataset.data object
        :param shuffle: choose if shuffle data
        :return: a normalized tf.Dataset.data object
        """
        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        normalized_ds = keras_ds.map(lambda x, y: (normalization_layer(x), y))
        normalized_ds = normalized_ds.cache()

        if shuffle:
            normalized_ds = normalized_ds.shuffle(1000, seed=123)
        normalized_ds = normalized_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return normalized_ds

    train_ds = resize_and_scaling(train_dataset, True)
    validation_ds = resize_and_scaling(validation_dataset, False)
    test_ds = resize_and_scaling(test_dataset)

    return train_ds, validation_ds, test_ds


# Perform data augmentation
def perform_data_augmentation():
    """
    Perform data Augmentation:
     - RandomFlip
     - RandomRotation
     - RandomZoom
    :return: tf.keras.Sequential model instance
    """
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    return data_augmentation
