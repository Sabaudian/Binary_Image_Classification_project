# Import
import numpy as np
import tensorflow as tf

# My import
import constants as const


# load data into keras dataset
def load_dataset(train_data_dir, test_data_dir):
    """
     Load data into Keras datasets.
     Note:
        - The function uses TensorFlow's `image_dataset_from_directory` utility to load the datasets.
        - Training and validation datasets are split from the `train_data_dir` with a validation split of 0.2.
        - The datasets are shuffled for training purposes.
        - The image size and batch size are determined by constants defined in the `constant` module.

    :param train_data_dir: The directory path containing the training data.
    :param test_data_dir: The directory path containing the test data.

    :returns: A tuple containing three datasets:
            - The training dataset.
            - The validation dataset.
            - The test dataset.
    """
    # train and validation dataset
    print("\n> Training and Validation: ")
    train_ks_dataset, val_ks_dataset = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        color_mode="rgb",
        batch_size=const.BATCH_SIZE,
        image_size=const.IMG_SIZE,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset="both",
    )

    # test dataset
    print("\n> Test:")
    test_ks_dataset = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        batch_size=const.BATCH_SIZE,
        image_size=const.IMG_SIZE,
        shuffle=False,
    )
    return train_ks_dataset, val_ks_dataset, test_ks_dataset


# Data Augmentation
def perform_data_augmentation():
    """
    Perform data augmentation.

    Note:
        - It is performed using a TensorFlow Sequential model with the following transformations:
        - RandomFlip: str, Specifies the type of random flip to be applied.
        - RandomRotation: float, Specifies the maximum angle of rotation in degrees.
        - RandomZoom: float, Specifies the maximum zoom factor.

    :return: tf.keras.Sequential, A sequential model representing the data augmentation transformations.
    """

    # Perform data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.2),
    ])

    return data_augmentation


# Data Normalization
def data_normalization(tf_dataset, augment):
    """
    Normalize the data and optionally apply data augmentation.

    Note:
        - Standardizes the data by rescaling pixel values to the range [0, 1].
        - The dataset is configured for performance by prefetching data using TensorFlow's AUTOTUNE mechanism.

    :param tf_dataset: tf.data.Dataset, The dataset to be normalized.
    :param augment: bool, Whether to apply data augmentation.
        If True, augmentation is performed only on the training set.

    :return: tf.Dataset.data object.
    """

    # Standardize the data
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    ds = tf_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Use data augmentation only on the training set
    if augment:
        data_augmentation = perform_data_augmentation()
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Configure the dataset for performance
    ds = ds.prefetch(buffer_size=const.AUTOTUNE)

    return ds


# Convert tensorflow dataset object into an array
def image_to_array(tf_dataset):
    """
    Convert a TensorFlow dataset object into NumPy arrays.

    Note:
        - The function iterates through the TensorFlow dataset to extract images and labels.
        - Images and labels are appended to separate lists and then converted into NumPy arrays.

    :param tf_dataset: tf.Dataset.data object.

    :return: A tuple containing two NumPy arrays:
        - X_array: The input images.
        - y_array: The corresponding class labels.
    """

    X_array = []  # Images
    y_array = []  # Labels

    for image, label in tf_dataset.unbatch().map(lambda x, y: (x, y)):
        X_array.append(image)
        y_array.append(label)

    # Input values
    X_array = np.array(X_array)
    # Target values
    y_array = np.array(y_array)

    return X_array, y_array
