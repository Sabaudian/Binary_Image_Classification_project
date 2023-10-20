import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# My import
import constants as const
import utils.general_functions as general


# ************************************** #
# *********** PLOT FUNCTIONS *********** #
# ************************************** #


# Show the amount of data per class
def plot_img_class_histogram(train_data, test_data, show_on_screen=True, store_in_folder=True):
    """
    Plot a histogram that shows the amount of data per class of the dataset.
    The histogram shows the number of images that represent muffin and chihuahua for the training and test set.
    :param train_data: Pandas.Dataframe, training data
    :param test_data: Pandas.Dataframe, test data
    :param show_on_screen: boolean, decides if show plot on screen
    :param store_in_folder: boolean, decides if save plot on location
    """

    # plot dataframe, counting data in it
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    fig.suptitle("DATA VISUALIZATION CHART", fontsize=18, weight="bold")
    sns.countplot(data=train_data, x="label", ax=ax1, hue="label")
    sns.countplot(data=test_data, x="label", ax=ax2, hue="label")

    # settings first plot
    ax1.set_title("Train Set", fontsize=16, weight="bold")
    ax1.set_xlabel(xlabel="Label", fontsize=14)
    ax1.set_ylabel(ylabel="Count", fontsize=14)
    # settings second plot
    ax2.set_title("Test Set", fontsize=16, weight="bold")
    ax2.set_xlabel(xlabel="Label", fontsize=14)
    ax2.set_ylabel(ylabel="Count", fontsize=14)

    # plot the exact amount of train's data
    for data in ax1.patches:
        x = data.get_x() + data.get_width() / 2  # text centered
        y = data.get_y() + data.get_height()  # text placed at column height => number of images in that label
        value = int(data.get_height())  # get value
        ax1.text(x, y, value, ha="center", fontsize=12, weight="bold")

    # plot the exact amount of test's data
    for data in ax2.patches:
        x = data.get_x() + data.get_width() / 2  # text centered
        y = data.get_y() + data.get_height()  # text placed at column height => number of images in that label
        value = int(data.get_height())  # get value
        ax2.text(x, y, value, ha="center", fontsize=12, weight="bold")

    # store plot into a proper folder
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_dataset_visualization" + const.FILE_EXTENSION, dpi=300)
        if not show_on_screen:
            plt.close()

    # show on screen
    if show_on_screen:
        # plot the image
        plt.show()


# plot the value's distribution of the images in the dataset
def plot_img_aspect_ratio(image_metadata, show_on_screen=True, store_in_folder=True):
    """
    Value's distribution of the images in the dataset
    :param image_metadata:
    :param show_on_screen:
    :param store_in_folder:
    """
    # Initialize value
    width, height, label = image_metadata["width"], image_metadata["height"], image_metadata["label"]

    # plot the image
    plt.subplots(figsize=(16, 8))
    sns.scatterplot(x=width, y=height, hue=label)

    # choose if store image to a path
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_img_aspect_ratio" + const.FILE_EXTENSION, dpi=300)
        if not show_on_screen:
            plt.close()

    # show on screen
    if show_on_screen:
        # plot the image
        plt.show()


def plot_data_visualization(train_ds, show_on_screen=True, store_in_folder=True):
    plt.figure(figsize=(16, 8))
    for images, labels in train_ds.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            if labels[i] == 0:
                plt.title("chihuahua")
            else:
                plt.title("muffin")
            plt.axis("off")

    # if it is true, save the image
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_visual_data" + const.FILE_EXTENSION, dpi=300)
        if not show_on_screen:
            plt.close()
    # if it is true, show the image
    if show_on_screen:
        plt.show()


def plot_data_augmentation(train_ds, data_augmentation, show_on_screen=True, store_in_folder=True):
    plt.figure(figsize=(16, 8))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")

    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_data_augmentation" + const.FILE_EXTENSION, dpi=300)
        if not show_on_screen:
            plt.close()
    # show on screen
    if show_on_screen:
        # plot the image
        plt.show()


def plot_history(model_history, model_name, show_on_screen=True, store_in_folder=True):
    plt.figure(figsize=(16, 8))  # plot dim.

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(model_history.history["accuracy"], linewidth=3)
    plt.plot(model_history.history["val_accuracy"], linewidth=3)
    plt.title("Training and Validation Accuracy", fontsize=18)
    plt.ylabel("accuracy", fontsize=16)
    plt.xlabel("epoch", fontsize=16)
    plt.grid()
    plt.legend(["Train", "validation"], loc="best")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(model_history.history["loss"], linewidth=3)
    plt.plot(model_history.history["val_loss"], linewidth=3)
    plt.title("Training and Validation Loss", fontsize=18)
    plt.ylabel("Loss", fontsize=16)
    plt.xlabel("epoch", fontsize=16)
    plt.grid()
    plt.legend(["Train", "validation"], loc="best")

    # to store plot
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_" + model_name + "_history" + const.FILE_EXTENSION, dpi=300)
        if not show_on_screen:
            plt.close()
    # show on screen
    if show_on_screen:
        # plot the image
        plt.show()


def plot_prediction(model, model_name, test_data, show_on_screen=True, store_in_folder=True):
    predict = model.predict(test_data)
    prediction_labels = np.argmax(predict, axis=-1)
    predicted_class_labels = ["chihuahua" if pred_label == 0 else "muffin" for pred_label in prediction_labels]

    image, label = next(iter(test_data))

    plt.figure(figsize=(16, 8))
    for images, label in test_data.take(9):
        for i in range(9):
            plt.imshow(image[i].numpy().astype("uint8"))
            plt.title("Predicted: {}\n".format(predicted_class_labels[i]) +
                      "True Label: {}".format(label[i]))
            plt.axis("off")
        plt.tight_layout()

    # to store plot
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_" + model_name + "_label_prediction" + const.FILE_EXTENSION,
                    dpi=300)
        if not show_on_screen:
            plt.close()

    # to show plot
    if show_on_screen:
        plt.show()

# FUNZIONI DA DEFINIRE
# https://towardsdatascience.com/10-minutes-to-building-a-binary-image-classifier-by-applying-transfer-learning-to-mobilenet-eab5a8719525
# def plot_roc_curve(false_positive_rate, true_positive_rate, validation_data):
