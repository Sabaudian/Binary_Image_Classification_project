# Import
import numpy as np
import tensorflow as tf

from tensorflow import keras

# My import
import constants as const


# load data into keras dataset
def load_dataset(train_data_dir, test_data_dir):
    """
    Load data from directory as a Keras dataset

    :param train_data_dir: the path to the training data
    :param test_data_dir: the path to the test data
    :return: tf.Dataset.data object
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
    Performs data augmentation using Keras Sequential model with specific layers.

    - RandomFlip: str, Specifies the type of random flip to be applied.

    - RandomRotation: float, Specifies the maximum angle of rotation in degrees.

    - RandomZoom: float, Specifies the maximum zoom factor.

    :return: Keras Sequential model representing the data augmentation operations.
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
    Scale the keras dataset and perform prefetch.
    Apply also data augmentation on the training set.

    :param tf_dataset: tf.Dataset.data object.
    :param augment: Boolean value, if True, performs data augmentation on dataset.

    :return: tf.Dataset.data object.
    """
    # Standardize the data
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    ds = tf_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Use data augmentation only on the training set.
    if augment:
        data_augmentation = perform_data_augmentation()
        ds = ds.map(lambda x, y: (data_augmentation(x), y))

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


# Convert tensorflow dataset object into an array
def image_to_array(tf_dataset):
    """
    Transform a tf.Dataset.data object into a split array form.

    :param tf_dataset: tf.Dataset.data object.

    :return: X (Input values), y (target values).
    """
    X_array = []  # Images
    y_array = []  # Class labels

    for image, label in tf_dataset.unbatch().map(lambda x, y: (x, y)):
        X_array.append(image)
        y_array.append(label)

    # Input values
    X_array = np.array(X_array)
    # Target values
    y_array = np.array(y_array)

    return X_array, y_array
