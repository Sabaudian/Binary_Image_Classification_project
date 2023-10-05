# Import
import tensorflow as tf
from keras import layers
from keras.models import Sequential


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
