# Import
import numpy as np
import tensorflow as tf

# My import
import constants as const
import plot_functions


# load data into keras dataset
def load_dataset(train_data_dir, test_data_dir):
    """
    Load data as a Keras dataset and perform tuning and rescaling processes
    :param train_data_dir: the path to the training data
    :param test_data_dir: the path to the test data
    :return: tf.Dataset.data object
    """
    # train dataset
    print("\nTraining: ")
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
    print("\nTest:")
    test_ks_dataset = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        batch_size=const.BATCH_SIZE,
        image_size=const.IMG_SIZE,
        shuffle=False
    )
    return train_ks_dataset, val_ks_dataset, test_ks_dataset


def data_normalization(keras_ds):
    """
    Scale the keras dataset and perform prefetch
    :param keras_ds: tf.Dataset.data object
    :return: a scaled tf.Dataset.data object
    """
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    normalized_ds = keras_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_ds = normalized_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return normalized_ds


def image_to_array(keras_ds):
    X_array = []  # store data
    y_array = []  # store labels

    for image, label in keras_ds.unbatch().map(lambda x, y: (x, y)):
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
