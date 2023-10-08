# Import
import os
import imagehash
import pandas as pd

from PIL import Image, UnidentifiedImageError

# My import
import plot_functions
from utils.general_functions import makedir
from utils.general_functions import sort_files
from utils.general_functions import count_files
from utils.general_functions import define_dataframe


# Check dataset: filter out possible corrupted files.
def corruption_filter(dataset_path):
    """
    Checking dataset,
    verifying the presence of corrupted file and collect metadata
    :param dataset_path: the path to the dataset
    """
    # Initialize
    bad_files = []  # to store corrupted file

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
                        bad_files.append(file_path)
                        print("\n> There are {} corrupted files: {}".format(len(bad_files), bad_files))

    if len(bad_files) != 0:
        doc_message = input("\n> Do you want to delete these {} file? [Y/N]: ".format(len(bad_files)))
        if doc_message.upper() == "Y":
            for bad_file in bad_files:
                # delete duplicate
                os.remove(os.path.join(dataset_path, bad_file))
                print("- {} Corrupted File Deleted Successfully!".format(bad_file))
    else:
        print("> No Corrupted File Found")


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

    # structure of the metadata
    metadata_df = ((pd.DataFrame.from_dict(metadata, orient="index").reset_index()
                    .rename(columns={"index": "image", 0: "width", 1: "height"})))
    # Compute aspect ratio
    metadata_df["aspect_ratio"] = round(metadata_df["width"] / metadata_df["height"], 2)
    # Add label column
    metadata_df["label"] = metadata_df.apply(lambda row: row.iloc[0].rsplit("/")[-2], axis=1)  # label
    # save the metadata into file
    makedir(dirpath="data")
    metadata_df.describe().to_csv(path_or_buf="data/dataset_metadata.csv", float_format="%.2f")
    # show metadata information
    print("\n> Metadata:\n {}\n\n{}".format(metadata_df, round(metadata_df.describe(), 2)))


# Check dataset: control the presence of duplicate inside the training set
def find_out_duplicate(dataset_path, hash_size):
    """
    Find and delete Duplicates
    :param dataset_path: the path to dataset
    :param hash_size: images will be resized to a matrix with size by given value
    """
    # Initialize
    hashes = {}
    duplicates = []
    # loop through file
    for file in os.listdir(dataset_path):
        with Image.open(os.path.join(dataset_path, file)) as image:
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
                # delete duplicate
                os.remove(os.path.join(dataset_path, duplicate))
                print("- {} Deleted Successfully!".format(duplicate))
    else:
        print("> No Duplicate Found")


# Displaying data as a clear plot
def view_data(train_dir_path, test_dir_path, show_histogram):
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
    print("\n> CHECK THE DATASET")
    print("\n> Checking the Number of file before performing Pre-processing Task...")
    # Count data
    count_train_chihuahua = count_files(file_path=dataset_path + "/train/chihuahua")
    count_train_muffin = count_files(file_path=dataset_path + "/train/muffin")
    count_test_chihuahua = count_files(file_path=dataset_path + "/test/chihuahua")
    count_test_muffin = count_files(file_path=dataset_path + "/test/muffin")
    tot_number_file = count_train_chihuahua + count_train_muffin + count_test_chihuahua + count_test_muffin
    print("> Number of file in train/chihuahua: {}".format(count_train_chihuahua) +
          "\n> Number of file in train/muffin: {}".format(count_train_muffin) +
          "\n> Number of file in test/chihuahua: {}".format(count_test_chihuahua) +
          "\n> Number of file in test/muffin: {}".format(count_test_muffin) +
          "\n> Total Number of file: {}\n".format(tot_number_file))

    # check for corrupted file
    print("\n> Checking for corrupted files...")
    corruption_filter(dataset_path=dataset_path)
    print("> Total Number of file After Applying Corruption Filter: {}\n".format(tot_number_file))

    # check for duplicates in training dataset
    print("\n> Checking duplicates in CHIHUAHUA directory...[num. of file: {}]".format(count_train_chihuahua))
    find_out_duplicate(dataset_path=dataset_path + "/train/chihuahua", hash_size=8)
    print("> Number of file in train/chihuahua: {}".format(count_train_chihuahua))
    print("\n> Checking duplicates in MUFFIN directory...[num. of file: {}]".format(count_train_muffin))
    find_out_duplicate(dataset_path=dataset_path + "/train/muffin", hash_size=8)
    print("> Number of file in train/muffin: {}".format(count_train_muffin))

    # structure of the metadata
    collect_metadata(dataset_path=dataset_path)

    # displaying histogram
    view_data(train_dir_path, test_dir_path, show_histogram=False)
    print("\n> DATASET CHECK COMPLETE!")
