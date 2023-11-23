# Import
import keras
import numpy as np
import tensorflow as tf

# My import
import constants as const

# Perform data augmentation
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])


# load data into keras dataset
def load_dataset(train_data_dir, test_data_dir):
    """
    Load data as a Keras dataset and perform tuning and rescaling processes
    :param train_data_dir: the path to the training data
    :param test_data_dir: the path to the test data
    :return: tf.Dataset.data object
    """
    # train dataset
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
        shuffle=True,
    )
    return train_ks_dataset, val_ks_dataset, test_ks_dataset


def data_normalization(tf_dataset, augment):
    """
    Scale the keras dataset and perform prefetch.
    :param tf_dataset: tf.Dataset.data object.
    :param augment: Boolean value, if True, performs data augmentation on dataset.
    :return: tf.Dataset.data object.
    """
    # Standardize the data
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    ds = tf_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y))

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    # ds = ds.cache().prefetch(buffer_size=AUTOTUNE)

    return ds


def image_to_array(tf_dataset):
    """
    Transform a Tensorflow dataset object into a split array form
    :param tf_dataset: tf.Dataset.data object
    :return: X (Input values), y (target values)
    """
    X_array = []  # store images
    y_array = []  # store class labels

    for image, label in tf_dataset.unbatch().map(lambda x, y: (x, y)):
        X_array.append(image)
        y_array.append(label)

    X_array = np.array(X_array)
    y_array = np.array(y_array)

    return X_array, y_array

# # Test Main
# if __name__ == '__main__':
#     train_ks_ds, val_ks_ds, test_ks_ds = load_dataset(train_data_dir=const.TRAIN_DIR,
#                                                       test_data_dir=const.TEST_DIR)
#
#     plot_functions.plot_data_visualization(train_ds=train_ks_ds, show_on_screen=True, store_in_folder=True)
#
#     data_augmentation = Sequential([
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(0.1),
#         layers.RandomZoom(0.1),
#     ])
#
#     plot_functions.plot_data_augmentation(train_ds=train_ks_ds, data_augmentation=data_augmentation,
#                                           show_on_screen=True, store_in_folder=True)
#
#     X_train = np.asarray(list(train_ds.map(lambda x, y: x)))
#     y_train = np.asarray(list(train_ds.map(lambda x, y: y)))
#
#     X_val = np.asarray(list(val_ds.map(lambda x, y: x)))
#     y_val = np.asarray(list(val_ds.map(lambda x, y: y)))
