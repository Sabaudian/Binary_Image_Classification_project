# Import
import os

import imagehash
from PIL import Image, UnidentifiedImageError

# My import
import plot_functions
# from utils.general_functions import sort_files
from utils.general_functions import count_files
from utils.general_functions import define_dataframe
from utils.general_functions import print_file_counts


# ********************************** #
# ********* PRE-PROCESSING ********* #
# ********************************** #

# Check dataset: filter out possible corrupted files.
def corruption_filter(dataset_path):
    """
    Check dataset for corrupted files and delete them if requested.

    :param dataset_path: The path to the dataset.
    """
    # Initialize
    bad_files = []  # to store corrupted file

    # Loop through all dataset subfolders
    for dirpath, _, filenames in os.walk(dataset_path):

        # Ensure we're processing a sub-folder level
        if dirpath is not dataset_path:

            # # Properly sorted file as presented in the original dataset
            # sort_files(filenames)

            # Loop through all files
            for filename in filenames:
                # Check the file extension
                if filename.endswith("jpg"):
                    # Get the file path
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with Image.open(file_path) as image:
                            image.verify()
                    except UnidentifiedImageError:
                        bad_files.append(file_path)
                        print("\n> There are {} corrupted files: {}".format(len(bad_files), bad_files))

    if len(bad_files) != 0:
        doc_message = input("\n> Do you want to delete these {} file? [Y/N]: ".format(len(bad_files)))
        if doc_message.upper() == "Y":
            for bad_file in bad_files:
                # delete duplicate
                os.remove(os.path.join(dataset_path, bad_file))
                print("- {} Corrupted File Deleted Successfully!".format(bad_file))

            # Print count
            print("\n> Checking the Number of file after the application of the corruption filter:")
            print_file_counts(dataset_path=dataset_path)
    else:
        print("> No Corrupted File Found")


# Check dataset: control the presence of duplicate inside the training set
def find_out_duplicate(dataset_path, hash_size):
    """
    Find and delete Duplicates

    :param dataset_path: the path to dataset.
    :param hash_size: images will be resized to a matrix with size by given value.
    """
    # Initialize
    hashes = {}
    duplicates = []

    # loop through file
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)

        with Image.open(file_path) as image:
            tmp_hash = imagehash.average_hash(image, hash_size)

            if tmp_hash in hashes:
                print("- Duplicate [{}] found for Image [{}]".format(file, hashes[tmp_hash]))
                duplicates.append(file)
            else:
                hashes[tmp_hash] = file

    if len(duplicates) != 0:
        doc_message = input("\n> Do you want to delete these {} file? [Y/N]: ".format(len(duplicates)))

        if doc_message.upper() == "Y":
            for duplicate in duplicates:
                # Delete duplicate
                os.remove(os.path.join(dataset_path, duplicate))
                print("- {} Deleted Successfully!".format(duplicate))
    else:
        print("> No Duplicate Found")


def checking_dataset(dataset_path, train_dir_path, test_dir_path, show_plot, save_plot):
    """
    Preliminary check on dataset:
        Calling corruption_filter, find_out_duplicate, collect_metadata and view_data function
        to analyze and control the dataset.

    :param dataset_path: The path to the dataset.
    :param train_dir_path: The path to the train dataset.
    :param test_dir_path: The path to the test dataset.
    :param show_plot: Decide if show data on screen or not.
    :param save_plot: Decide if store data or not.
    """
    print("\n> CHECK THE DATASET")
    print("\n> Checking the Number of file before performing Pre-processing Task...")

    # Data path
    train_chihuahua_path = os.path.join(dataset_path, "train/chihuahua")
    train_muffin_path = os.path.join(dataset_path, "train/muffin")

    # Print count
    print_file_counts(dataset_path=dataset_path)

    # Check for corrupted file
    print("> Checking for corrupted files...")
    corruption_filter(dataset_path=dataset_path)

    # Check for duplicates in training dataset: train/chihuahua and train /muffin
    print("\n> Checking duplicates in train/chihuahua directory...[current num. of file: {}]"
          .format(count_files(file_path=train_chihuahua_path)))
    find_out_duplicate(dataset_path=train_chihuahua_path, hash_size=32)

    print("\n> Checking duplicates in train/muffin directory...[current num. of file: {}]"
          .format(count_files(file_path=train_muffin_path)))
    find_out_duplicate(dataset_path=train_muffin_path, hash_size=32)

    print("\n> Final check to confirm the total file count:")
    print_file_counts(dataset_path=dataset_path)

    # Defines a dataframe
    train_df, test_df = define_dataframe(train_dir_path=train_dir_path,
                                         test_dir_path=test_dir_path)
    # Plot class distribution
    plot_functions.plot_class_distribution(train_data=train_df, test_data=test_df, show_on_screen=show_plot,
                                           store_in_folder=save_plot)

    print("\n> DATASET CHECK COMPLETE!")
